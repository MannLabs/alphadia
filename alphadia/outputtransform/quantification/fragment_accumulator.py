import logging
import os
from collections.abc import Iterator

import pandas as pd
import pyarrow.parquet as pq

from alphadia.outputtransform.quantification.quant_builder import (
    ION_HASH_COLUMNS,
    precursor_idx_from_ion,
    prepare_df,
)
from alphadia.utils import log_lfq_step

logger = logging.getLogger()


class FragmentQuantLoader:
    """Load and accumulate fragment quantification data from multiple runs.

    This class handles reading fragment files from folders and accumulating
    them into unified intensity and correlation matrices.

    Parameters
    ----------
    psm_df : pd.DataFrame
        PSM dataframe to filter fragments by precursor_idx
    columns : list[str] | None, default=None
        Columns to extract from fragment data. Defaults to ["intensity", "correlation"]
    """

    def __init__(self, psm_df: pd.DataFrame, columns: list[str] | None = None):
        self.psm_df = psm_df
        self.columns = ["intensity", "correlation"] if columns is None else columns
        # the remaining columns of the fragment files are never used downstream,
        # reading them would roughly double the bytes read per file
        self._read_columns = list(dict.fromkeys(ION_HASH_COLUMNS + self.columns))

    def accumulate_from_folders(
        self, folder_list: list[str]
    ) -> dict[str, pd.DataFrame] | None:
        """Accumulate fragment data from a list of folders.

        Parameters
        ----------
        folder_list : list[str]
            List of folders containing frag.parquet files

        Returns
        -------
        dict[str, pd.DataFrame] | None
            Dictionary with column name as key and dataframe as value, where each dataframe
            has columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
            Returns None if no fragment files are found
        """
        df_iterable = self._get_frag_df_generator(folder_list)
        return self.accumulate(df_iterable)

    def accumulate(
        self, df_iterable: Iterator[tuple[str, pd.DataFrame]]
    ) -> dict[str, pd.DataFrame] | None:
        """Accumulate fragment data from an iterator of (run_name, dataframe) tuples.

        Parameters
        ----------
        df_iterable : Iterator[tuple[str, pd.DataFrame]]
            Iterator yielding (run_name, fragment_df) tuples

        Returns
        -------
        dict[str, pd.DataFrame] | None
            Dictionary with column name as key and dataframe as value, where each dataframe
            has columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
            Returns None if iterator is empty
        """
        logger.info("Accumulating fragment data")

        raw_name, df = next(df_iterable, (None, None))
        if df is None:
            logger.warning(f"No frag file found for {raw_name}")
            return None

        df = prepare_df(df, self.psm_df, columns=self.columns)

        # the ion hash already encodes the precursor_idx, so it is the only join key
        # needed here and is expanded again once all runs are merged
        df_list = []
        for col in self.columns:
            feat_df = df[["ion", col]].copy()
            feat_df.rename(columns={col: raw_name}, inplace=True)
            df_list.append(feat_df)

        for run_number, (raw_name, df) in enumerate(df_iterable, start=2):
            df = prepare_df(df, self.psm_df, columns=self.columns)

            for idx, col in enumerate(self.columns):
                df_list[idx] = df_list[idx].merge(
                    df[["ion", col]],
                    on="ion",
                    how="outer",
                )
                df_list[idx].rename(columns={col: raw_name}, inplace=True)

            log_lfq_step(
                f"merged run {run_number} ({raw_name}), "
                f"{len(df_list[0]):,} ions accumulated"
            )

        log_lfq_step("building precursor metadata")
        precursor_metadata_df = self.psm_df.groupby(
            "precursor_idx", as_index=False
        ).agg({"pg": "first", "mod_seq_hash": "first", "mod_seq_charge_hash": "first"})

        return {
            col: self._add_precursor_idx(df, precursor_metadata_df)
            for col, df in zip(self.columns, df_list)
        }

    def _get_frag_df_generator(
        self, folder_list: list[str]
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Generate (run_name, fragment_df) tuples from a list of folders.

        Parameters
        ----------
        folder_list : list[str]
            List of folders containing frag.parquet files

        Yields
        ------
        tuple[str, pd.DataFrame]
            Tuple of (run_name, fragment_dataframe)
        """
        for folder in folder_list:
            raw_name = os.path.basename(folder)
            frag_path = os.path.join(folder, "frag.parquet")

            if not os.path.exists(frag_path):
                logger.warning(f"no frag file found for {raw_name}")
            else:
                # checked before reading, as a schema mismatch affects all runs and
                # must not be swallowed by the warning below
                self._check_read_columns(frag_path)

                try:
                    logger.info(f"reading frag file for {raw_name}")
                    run_df = pd.read_parquet(frag_path, columns=self._read_columns)
                except Exception as e:
                    logger.warning(f"Error reading frag file for {raw_name}")
                    logger.warning(e)
                else:
                    log_lfq_step(f"read {len(run_df):,} fragments for {raw_name}")
                    yield raw_name, run_df

    def _check_read_columns(self, frag_path: str) -> None:
        """Raise if the fragment file does not contain all required columns."""
        missing_columns = set(self._read_columns) - set(pq.read_schema(frag_path).names)
        if missing_columns:
            raise ValueError(
                f"Fragment file {frag_path} is missing required columns: {sorted(missing_columns)}"
            )

    @staticmethod
    def _add_precursor_idx(
        df: pd.DataFrame, precursor_metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add precursor index and its metadata to fragment data.

        Parameters
        ----------
        df : pd.DataFrame
            Fragment data with ion
        precursor_metadata_df : pd.DataFrame
            Precursor metadata with precursor_idx, pg, mod_seq_hash, mod_seq_charge_hash

        Returns
        -------
        pd.DataFrame
            Fragment data with precursor_idx and precursor metadata columns added
        """
        df.fillna(0, inplace=True)
        df.insert(0, "precursor_idx", precursor_idx_from_ion(df["ion"].values))
        df = df.merge(precursor_metadata_df, on="precursor_idx", how="left")
        return df

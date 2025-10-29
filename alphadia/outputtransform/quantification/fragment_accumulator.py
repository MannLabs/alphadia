import logging
import os
from collections.abc import Iterator

import numpy as np
import pandas as pd

from alphadia.outputtransform.quantification.quant_builder import prepare_df

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

        df_list = []
        for col in self.columns:
            feat_df = df[["precursor_idx", "ion", col]].copy()
            feat_df.rename(columns={col: raw_name}, inplace=True)
            df_list.append(feat_df)

        for raw_name, df in df_iterable:
            df = prepare_df(df, self.psm_df, columns=self.columns)

            for idx, col in enumerate(self.columns):
                df_list[idx] = df_list[idx].merge(
                    df[["ion", col, "precursor_idx"]],
                    on=["ion", "precursor_idx"],
                    how="outer",
                )
                df_list[idx].rename(columns={col: raw_name}, inplace=True)

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
                try:
                    logger.info(f"reading frag file for {raw_name}")
                    run_df = pd.read_parquet(frag_path)
                except Exception as e:
                    logger.warning(f"Error reading frag file for {raw_name}")
                    logger.warning(e)
                else:
                    yield raw_name, run_df

    @staticmethod
    def _add_precursor_idx(
        df: pd.DataFrame, precursor_metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add precursor index metadata to fragment data.

        Parameters
        ----------
        df : pd.DataFrame
            Fragment data with precursor_idx
        precursor_metadata_df : pd.DataFrame
            Precursor metadata with precursor_idx, pg, mod_seq_hash, mod_seq_charge_hash

        Returns
        -------
        pd.DataFrame
            Fragment data with precursor metadata columns added
        """
        df.fillna(0, inplace=True)
        df["precursor_idx"] = df["precursor_idx"].astype(np.uint32)
        df = df.merge(precursor_metadata_df, on="precursor_idx", how="left")
        return df

import logging
import os
from collections.abc import Iterator

import directlfq.config as lfqconfig
import directlfq.normalization as lfqnorm
import directlfq.protein_intensity_estimation as lfqprot_estimation
import directlfq.utils as lfqutils
import numba as nb
import numpy as np
import pandas as pd

from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger()


def get_frag_df_generator(folder_list: list[str]):
    """Return a generator that yields a tuple of (raw_name, frag_df)

    Parameters
    ----------

    folder_list: List[str]
        List of folders containing the frag.tsv file

    Returns
    -------

    Iterator[Tuple[str, pd.DataFrame]]
        Tuple of (raw_name, frag_df)

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


@nb.njit(cache=USE_NUMBA_CACHING)
def _ion_hash(precursor_idx, number, type, charge, loss_type):
    # create a 64 bit hash from the precursor_idx, number and type
    # the precursor_idx is the lower 32 bits
    # the number is the next 8 bits
    # the type is the next 8 bits
    # the charge is the next 8 bits
    # the loss_type is the last 8 bits
    return (
        precursor_idx
        + (number << 32)
        + (type << 40)
        + (charge << 48)
        + (loss_type << 56)
    )


def prepare_df(
    df: pd.DataFrame, psm_df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    df = df[df["precursor_idx"].isin(psm_df["precursor_idx"])].copy()
    df["ion"] = _ion_hash(
        df["precursor_idx"].values,
        df["number"].values,
        df["type"].values,
        df["charge"].values,
        df["loss_type"].values,
    )
    return df[["precursor_idx", "ion"] + columns]


class QuantBuilder:
    def __init__(self, psm_df: pd.DataFrame, columns: list[str] | None = None):
        self.psm_df = psm_df
        self.columns = ["intensity", "correlation"] if columns is None else columns

    def accumulate_frag_df_from_folders(
        self, folder_list: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Accumulate the fragment data from a list of folders

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the frag.tsv file

        Returns
        -------
        dict
            Dictionary with column name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...
        """

        df_iterable = get_frag_df_generator(folder_list)
        return self.accumulate_frag_df(df_iterable)

    def accumulate_frag_df(
        self, df_iterable: Iterator[tuple[str, pd.DataFrame]]
    ) -> dict[str, pd.DataFrame]:
        """Consume a generator of (raw_name, frag_df) tuples and accumulate the data in a single dataframe

        Parameters
        ----------

        df_iterable: Iterator[Tuple[str, pd.DataFrame]]
            Iterator of (raw_name, frag_df) tuples

        Returns
        -------
        dict
            Dictionary with feature name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...
        (None, None) if df_iterable is empty
        """

        logger.info("Accumulating fragment data")

        raw_name, df = next(df_iterable, (None, None))
        if df is None:
            logger.warning(f"No frag file found for {raw_name}")
            return None, None

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

        # annotate protein group
        annotate_df = self.psm_df.groupby("precursor_idx", as_index=False).agg(
            {"pg": "first", "mod_seq_hash": "first", "mod_seq_charge_hash": "first"}
        )

        return {
            col: self.add_annotation(df, annotate_df)
            for col, df in zip(self.columns, df_list)
        }

    @staticmethod
    def add_annotation(df: pd.DataFrame, annotate_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add annotation to the fragment data, including protein group, mod_seq_hash, mod_seq_charge_hash

        Parameters
        ----------
        df: pd.DataFrame
            Fragment data

        Returns
        -------
        pd.DataFrame
            Fragment data with annotation
        """
        df.fillna(0, inplace=True)
        df["precursor_idx"] = df["precursor_idx"].astype(np.uint32)

        df = df.merge(annotate_df, on="precursor_idx", how="left")

        return df

    def filter_frag_df(
        self,
        intensity_df: pd.DataFrame,
        quality_df: pd.DataFrame,
        min_correlation: float = 0.5,
        top_n: int = 3,
        group_column: str = "pg",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter the fragment data by quality

        Parameters
        ----------
        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        min_correlation: float
            Minimum correlation to keep a fragment, if not below top_n

        top_n: int
            Keep the top n fragments per precursor

        Returns
        -------

        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        """

        logger.info("Filtering fragments by quality")

        run_columns = [
            c
            for c in intensity_df.columns
            if c
            not in ["precursor_idx", "ion", "pg", "mod_seq_hash", "mod_seq_charge_hash"]
        ]

        quality_df["total"] = np.mean(quality_df[run_columns].values, axis=1)
        quality_df["rank"] = quality_df.groupby(group_column)["total"].rank(
            ascending=False, method="first"
        )
        mask = (quality_df["rank"].values <= top_n) | (
            quality_df["total"].values > min_correlation
        )
        return intensity_df[mask], quality_df[mask]

    def lfq(
        self,
        intensity_df: pd.DataFrame,
        quality_df: pd.DataFrame,
        num_samples_quadratic: int = 50,
        min_nonan: int = 1,
        num_cores: int = 8,
        normalize: bool = True,
        group_column: str = "pg",
    ) -> pd.DataFrame:
        """Perform label-free quantification

        Parameters
        ----------

        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        Returns
        -------

        lfq_df: pd.DataFrame
            Dataframe with the label-free quantification data containing the columns precursor_idx, ion, intensity, protein

        """

        logger.info("Performing label-free quantification using directLFQ")

        # drop all other columns as they will be interpreted as samples
        columns_to_drop = list(
            {"precursor_idx", "pg", "mod_seq_hash", "mod_seq_charge_hash"}
            - {group_column}
        )
        _intensity_df = intensity_df.drop(columns=columns_to_drop)

        lfqconfig.set_global_protein_and_ion_id(protein_id=group_column, quant_id="ion")
        lfqconfig.set_compile_normalized_ion_table(
            compile_normalized_ion_table=False
        )  # save compute time by avoiding the creation of a normalized ion table
        lfqconfig.check_wether_to_copy_numpy_arrays_derived_from_pandas()  # avoid read-only pandas bug on linux if applicable
        lfqconfig.set_log_processed_proteins(
            log_processed_proteins=True
        )  # here you can chose wether to log the processed proteins or not

        _intensity_df.sort_values(by=group_column, inplace=True, ignore_index=True)

        lfq_df = lfqutils.index_and_log_transform_input_df(_intensity_df)
        lfq_df = lfqutils.remove_allnan_rows_input_df(lfq_df)

        if normalize:
            lfq_df = lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
                lfq_df,
                num_samples_quadratic=num_samples_quadratic,
                selected_proteins_file=None,
            ).complete_dataframe

        protein_df, _ = lfqprot_estimation.estimate_protein_intensities(
            lfq_df,
            min_nonan=min_nonan,
            num_samples_quadratic=num_samples_quadratic,
            num_cores=num_cores,
        )

        return protein_df

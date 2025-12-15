import logging
from dataclasses import dataclass

import directlfq.config as lfqconfig
import directlfq.normalization as lfqnorm
import directlfq.protein_intensity_estimation as lfqprot_estimation
import directlfq.utils as lfqutils
import numba as nb
import numpy as np
import pandas as pd
from quantselect.output import run_quantselect

from alphadia.constants.keys import NormalizationMethods
from alphadia.utils import USE_NUMBA_CACHING
from alphadia.workflow.config import Config

logger = logging.getLogger()


@dataclass
class LFQOutputConfig:
    """Configuration for label-free quantification output at a specific level.

    Parameters
    ----------
    quant_level : str
        Column name to use for grouping quantification (e.g., 'mod_seq_charge_hash', 'mod_seq_hash', 'pg')
    level_name : str
        Descriptive name for this quantification level (e.g., 'precursor', 'peptide', 'pg')
    intensity_column : str
        Name of the intensity column in the output
    aggregation_components : list[str]
        Columns which are shared within a group by quant level.
        e.g. if the quant level is precursr, all rows will have the same pg, sequence, mods, mod_sites and charge.
    should_process : bool, default=True
        Whether to process this quantification level
    save_fragments : bool, default=False
        Whether to save fragment-level quantification matrices
    normalization_method: str | None, default="directlfq"
        Normalization method to use (e.g., 'directlfq', 'quantselect')
    """

    quant_level: str
    level_name: str
    intensity_column: str
    aggregation_components: list[str]
    should_process: bool = True
    save_fragments: bool = False
    normalization_method: str = NormalizationMethods.DIRECTLFQ


@nb.njit(cache=USE_NUMBA_CACHING)
def _ion_hash(precursor_idx, number, type, charge, loss_type):
    """Create a 64-bit hash from fragment ion characteristics.

    Parameters
    ----------
    precursor_idx : array-like
        Precursor indices (lower 32 bits)
    number : array-like
        Fragment number (next 8 bits)
    type : array-like
        Fragment type (next 8 bits)
    charge : array-like
        Fragment charge (next 8 bits)
    loss_type : array-like
        Loss type (last 8 bits)

    Returns
    -------
    int64
        64-bit hash value
    """
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
    """Prepare fragment dataframe by filtering and adding ion hash.

    Parameters
    ----------
    df : pd.DataFrame
        Fragment dataframe
    psm_df : pd.DataFrame
        PSM dataframe with precursor_idx column
    columns : list[str]
        Columns to keep from fragment data

    Returns
    -------
    pd.DataFrame
        Filtered fragment dataframe with ion hash
    """
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
    """Build quantification results through filtering and label-free quantification.

    This class focuses on fragment quality filtering and directLFQ-based
    protein quantification. Fragment data accumulation is handled by
    FragmentQuantLoader.

    Parameters
    ----------
    psm_df : pd.DataFrame
        PSM dataframe with precursor information
    columns : list[str] | None, default=None
        Columns to use for quantification. Defaults to ["intensity", "correlation"]
    """

    def __init__(self, psm_df: pd.DataFrame, columns: list[str] | None = None):
        self.psm_df = psm_df
        self.columns = ["intensity", "correlation"] if columns is None else columns

    def filter_frag_df(
        self,
        intensity_df: pd.DataFrame,
        quality_df: pd.DataFrame,
        min_correlation: float = 0.5,
        top_n: int = 3,
        group_column: str = "pg",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter fragment data by quality metrics.

        Keeps fragments that meet either of these criteria:
        - Among top N fragments per group (by mean quality across runs)
        - Quality score above min_correlation threshold

        Parameters
        ----------
        intensity_df : pd.DataFrame
            Fragment intensity data with columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
        quality_df : pd.DataFrame
            Fragment quality/correlation data with same structure as intensity_df
        min_correlation : float, default=0.5
            Minimum quality score to keep fragment (if not in top N)
        top_n : int, default=3
            Number of top fragments to keep per group
        group_column : str, default='pg'
            Column to group fragments by (pg, mod_seq_hash, mod_seq_charge_hash)

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Filtered intensity and quality dataframes
        """
        logger.info("Filtering fragments by quality")

        # Extract sample/run columns (e.g., run_0, run_1, run_2)
        # These are all columns except metadata columns
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

    def direct_lfq(
        self,
        intensity_df: pd.DataFrame,
        lfq_config: LFQOutputConfig,
        config: Config,
    ) -> pd.DataFrame:
        """Perform label-free quantification using directLFQ.

        Parameters
        ----------
        intensity_df: pd.DataFrame
            Fragment intensity dataframe with columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
        lfq_config: LFQOutputConfig
            Configuration for this quantification level
        config: Config
            Global configuration object

        Returns
        -------
        pd.DataFrame
            Protein/peptide quantification results with columns: group_column, run1, run2, ...
        """
        logger.info(
            f"Performing label-free quantification with {lfq_config.normalization_method} normalization"
        )

        # drop all other columns as they will be interpreted as samples
        columns_to_drop = list(
            {"precursor_idx", "pg", "mod_seq_hash", "mod_seq_charge_hash"}
            - {lfq_config.quant_level}
        )
        intensity_df = intensity_df.drop(columns=columns_to_drop)

        lfqconfig.set_global_protein_and_ion_id(
            protein_id=lfq_config.quant_level, quant_id="ion"
        )
        lfqconfig.set_compile_normalized_ion_table(compile_normalized_ion_table=False)
        lfqconfig.check_wether_to_copy_numpy_arrays_derived_from_pandas()
        lfqconfig.set_log_processed_proteins(log_processed_proteins=True)

        intensity_df.sort_values(
            by=lfq_config.quant_level, inplace=True, ignore_index=True
        )

        lfq_df = lfqutils.index_and_log_transform_input_df(intensity_df)
        lfq_df = lfqutils.remove_allnan_rows_input_df(lfq_df)

        if config["search_output"]["normalize_directlfq"]:
            logger.info("Applying directLFQ normalization")
            lfq_df = lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
                lfq_df,
                num_samples_quadratic=config["search_output"]["num_samples_quadratic"],
                selected_proteins_file=None,
            ).complete_dataframe

        protein_df, _ = lfqprot_estimation.estimate_protein_intensities(
            lfq_df,
            min_nonan=config["search_output"]["min_nonnan"],
            num_samples_quadratic=config["search_output"]["num_samples_quadratic"],
            num_cores=config["general"]["thread_count"],
        )
        return protein_df

    def quantselect_lfq(
        self,
        feature_dfs_dict: dict[str, pd.DataFrame],
        lfq_config: LFQOutputConfig,
    ) -> pd.DataFrame:
        """Perform label-free quantification using QuantSelect.

        Parameters
        ----------
        feature_dfs_dict: dict[str, pd.DataFrame]
            Dictionary with feature name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...
        lfq_config: LFQOutputConfig
            Configuration for this quantification level

        Returns
        -------
        pd.DataFrame
            Protein/peptide quantification results with columns: group_column, run1, run2, ...
        """
        logger.info("Performing label-free quantification with QuantSelect")

        return run_quantselect(
            seed=42,
            psm_df=self.psm_df,
            feature_dfs_dict=feature_dfs_dict,
            lfq_config=lfq_config,
        )

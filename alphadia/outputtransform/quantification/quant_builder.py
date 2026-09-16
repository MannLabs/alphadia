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

from alphadia.constants.keys import QuantificationLevelKey
from alphadia.utils import USE_NUMBA_CACHING
from alphadia.workflow.config import Config

logger = logging.getLogger()

PRECURSOR_IDX_COLUMN = "precursor_idx"
# directLFQ's name for the lowest quantified unit: fragments in the precursor sum,
# precursors in the peptide and protein group estimation
ION_COLUMN = "ion"
# Steepness of the fragment weighting: 2 favours precision, 6-8 favour the accuracy
# of spiked-in ratios, 4 is the compromise
FRAGMENT_CORRELATION_POWER = 4
# Run columns of the accumulated fragment matrices are everything not listed here
FRAGMENT_METADATA_COLUMNS = [
    PRECURSOR_IDX_COLUMN,
    ION_COLUMN,
    *QuantificationLevelKey.get_values(),
]


def get_run_columns(df: pd.DataFrame) -> list[str]:
    """Run columns of an accumulated fragment matrix."""
    return [c for c in df.columns if c not in FRAGMENT_METADATA_COLUMNS]


def compute_mean_correlation(
    intensity_df: pd.DataFrame, correlation_df: pd.DataFrame
) -> np.ndarray:
    """Mean cross-run correlation of every fragment over the runs in which it was observed.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Fragment intensity data with columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
    correlation_df : pd.DataFrame
        Fragment correlation data with the same rows in the same order

    Returns
    -------
    np.ndarray
        Mean correlation per row of intensity_df, 0 for fragments never observed
    """
    if not np.array_equal(
        intensity_df[ION_COLUMN].to_numpy(), correlation_df[ION_COLUMN].to_numpy()
    ):
        raise ValueError(
            "Intensity and correlation tables must list the same fragments in the same order"
        )

    run_columns = get_run_columns(intensity_df)
    observed = intensity_df[run_columns].to_numpy() > 0
    correlation = correlation_df[run_columns].to_numpy()

    n_observed = observed.sum(axis=1)
    summed_correlation = np.einsum("ij,ij->i", correlation, observed)
    return np.divide(
        summed_correlation,
        n_observed,
        out=np.zeros(len(n_observed)),
        where=n_observed > 0,
    )


def compute_fragment_weights(
    mean_correlation: np.ndarray, precursor_hash: np.ndarray
) -> np.ndarray:
    """Weight of every fragment relative to the best fragment of its precursor.

    Parameters
    ----------
    mean_correlation : np.ndarray
        Mean cross-run correlation per fragment
    precursor_hash : np.ndarray
        Precursor of every fragment, aligned with mean_correlation

    Returns
    -------
    np.ndarray
        Weights in (0, 1], the best fragment of every precursor has weight 1
    """
    weights = np.clip(mean_correlation, 0.0, 1.0) ** FRAGMENT_CORRELATION_POWER
    best_weight = pd.Series(weights).groupby(precursor_hash).transform("max").to_numpy()
    # a precursor without any correlating fragment falls back to the plain sum
    return np.divide(
        weights, best_weight, out=np.ones_like(weights), where=best_weight > 0
    )


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
    """

    quant_level: str
    level_name: str
    intensity_column: str
    aggregation_components: list[str]
    should_process: bool = True


# explicit signature: a uint64 precursor_idx would make numba promote the hash
# to float64, silently rounding away the lower bits for large hashes.
@nb.njit(
    "int64[:](int64[:], int64[:], int64[:], int64[:], int64[:])",
    cache=USE_NUMBA_CACHING,
)
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
    df = df[df[PRECURSOR_IDX_COLUMN].isin(psm_df[PRECURSOR_IDX_COLUMN])].copy()
    df[ION_COLUMN] = _ion_hash(
        df[PRECURSOR_IDX_COLUMN].values.astype(np.int64),
        df["number"].values.astype(np.int64),
        df["type"].values.astype(np.int64),
        df["charge"].values.astype(np.int64),
        df["loss_type"].values.astype(np.int64),
    )
    return df[[PRECURSOR_IDX_COLUMN, ION_COLUMN] + columns]


class QuantBuilder:
    """Build quantification results through filtering and label-free quantification.

    Filters fragments by correlation, sums them to precursors with correlation
    weights and estimates peptides and protein groups from those precursors with
    directLFQ. Fragment data accumulation is handled by FragmentQuantLoader.

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
        correlation_df: pd.DataFrame,
        min_correlation: float = 0.5,
        top_n: int = 3,
        group_column: str = "pg",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter fragment data by cross-run correlation.

        Keeps fragments that meet either of these criteria:
        - Among top N fragments per group (by mean correlation across runs)
        - Mean correlation above min_correlation threshold

        Parameters
        ----------
        intensity_df : pd.DataFrame
            Fragment intensity data with columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
        correlation_df : pd.DataFrame
            Fragment correlation data with same structure as intensity_df
        min_correlation : float, default=0.5
            Minimum mean correlation to keep fragment (if not in top N)
        top_n : int, default=3
            Number of top fragments to keep per group
        group_column : str, default='pg'
            Column to group fragments by (pg, mod_seq_hash, mod_seq_charge_hash)

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Filtered intensity and correlation dataframes
        """
        logger.info("Filtering fragments by correlation")

        run_columns = get_run_columns(intensity_df)

        correlation_df["total"] = np.mean(correlation_df[run_columns].values, axis=1)
        correlation_df["rank"] = correlation_df.groupby(group_column)["total"].rank(
            ascending=False, method="first"
        )
        mask = (correlation_df["rank"].values <= top_n) | (
            correlation_df["total"].values > min_correlation
        )
        return intensity_df[mask], correlation_df[mask]

    def direct_lfq(
        self,
        precursor_df: pd.DataFrame,
        lfq_config: LFQOutputConfig,
        config: Config,
    ) -> pd.DataFrame:
        """Estimate peptide or protein group quantities with directLFQ from precursor quantities.

        Precursors take the role of directLFQ's ions. Sample normalization already
        happened on the fragment table, so there is no second pass here.

        Parameters
        ----------
        precursor_df: pd.DataFrame
            Precursor quantities with columns: mod_seq_charge_hash, mod_seq_hash, pg, run1, run2, ...
        lfq_config: LFQOutputConfig
            Configuration for this quantification level
        config: Config
            Global configuration object

        Returns
        -------
        pd.DataFrame
            Protein/peptide quantification results with columns: group_column, run1, run2, ...
        """
        logger.info("Performing label-free quantification with directLFQ")

        # a precursor listed under several protein groups is one ion within each group
        ion_df = (
            precursor_df.groupby(
                [QuantificationLevelKey.PRECURSOR, lfq_config.quant_level],
                as_index=False,
            )[get_run_columns(precursor_df)]
            .sum()
            .rename(columns={QuantificationLevelKey.PRECURSOR: ION_COLUMN})
        )
        lfq_df = self._prepare_ion_table(ion_df, lfq_config.quant_level)
        if lfq_df.empty:
            return pd.DataFrame(columns=[lfq_config.quant_level])

        protein_df, _ = lfqprot_estimation.estimate_protein_intensities(
            lfq_df,
            min_nonan=config["search_output"]["min_nonnan"],
            num_samples_quadratic=config["search_output"]["num_samples_quadratic"],
            num_cores=config["general"]["thread_count"],
        )
        return protein_df

    def sum_fragments_to_precursors(
        self,
        intensity_df: pd.DataFrame,
        correlation_df: pd.DataFrame,
        config: Config,
    ) -> pd.DataFrame:
        """Roll fragments up to precursors by a correlation-weighted sum.

        Every fragment gets one weight for all runs, so the ratio between two runs
        is preserved for the fragments observed in both. Interfered fragments barely
        count, where a median of fragment ratios would give them an equal vote.

        Parameters
        ----------
        intensity_df : pd.DataFrame
            Fragment intensity data with columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
        correlation_df : pd.DataFrame
            Fragment correlation data with the same rows in the same order
        config : Config
            Global configuration object

        Returns
        -------
        pd.DataFrame
            Precursor quantities with columns: mod_seq_charge_hash, mod_seq_hash, pg, run1, run2, ...
            Runs in which no fragment of a precursor was observed report 0.
        """
        logger.info("Summing fragments to precursors")

        run_columns = get_run_columns(intensity_df)
        group_columns = [
            QuantificationLevelKey.PRECURSOR,
            QuantificationLevelKey.PEPTIDE,
            QuantificationLevelKey.PROTEIN,
        ]
        weights = compute_fragment_weights(
            compute_mean_correlation(intensity_df, correlation_df),
            intensity_df[QuantificationLevelKey.PRECURSOR].to_numpy(),
        )

        observed = (intensity_df[run_columns].to_numpy() > 0).any(axis=1)
        linear_df = intensity_df.loc[observed].set_index(group_columns)[run_columns]
        if linear_df.empty:
            return linear_df.reset_index()

        if config["search_output"]["normalize_directlfq"]:
            log_df = np.log2(linear_df.replace(0, np.nan))
            linear_df = (2 ** self._normalize_ion_table(log_df, config)).fillna(0.0)

        return (
            linear_df.mul(weights[observed], axis=0)
            .groupby(level=group_columns)
            .sum()
            .reset_index()
        )

    def _prepare_ion_table(
        self, intensity_df: pd.DataFrame, quant_level: str
    ) -> pd.DataFrame:
        """Build the log2 ion table directLFQ operates on.

        Parameters
        ----------
        intensity_df: pd.DataFrame
            Ion table with an ion column, the quant_level column and one column per run.
            Other metadata columns are dropped.
        quant_level: str
            Column to group ions by (pg, mod_seq_hash, mod_seq_charge_hash)

        Returns
        -------
        pd.DataFrame
            Log2 intensities indexed by (quant_level, ion) with one column per run.
            Missing values are NaN and ions missing in every run are dropped.
        """
        # directLFQ treats every column except the group and ion id as a sample
        columns_to_drop = [
            c
            for c in intensity_df.columns
            if c in FRAGMENT_METADATA_COLUMNS and c not in (ION_COLUMN, quant_level)
        ]
        intensity_df = intensity_df.drop(columns=columns_to_drop)

        lfqconfig.set_global_protein_and_ion_id(
            protein_id=quant_level, quant_id=ION_COLUMN
        )
        lfqconfig.set_compile_normalized_ion_table(compile_normalized_ion_table=False)
        lfqconfig.check_wether_to_copy_numpy_arrays_derived_from_pandas()
        lfqconfig.set_log_processed_proteins(log_processed_proteins=True)

        intensity_df.sort_values(by=quant_level, inplace=True, ignore_index=True)

        lfq_df = lfqutils.index_and_log_transform_input_df(intensity_df)
        return lfqutils.remove_allnan_rows_input_df(lfq_df)

    def _normalize_ion_table(
        self, lfq_df: pd.DataFrame, config: Config
    ) -> pd.DataFrame:
        """Apply directLFQ sample normalization to a log2 ion table.

        Parameters
        ----------
        lfq_df: pd.DataFrame
            Log2 ion table as returned by _prepare_ion_table
        config: Config
            Global configuration object

        Returns
        -------
        pd.DataFrame
            Ion table with per-sample shifts removed
        """
        logger.info("Applying directLFQ normalization")
        return lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
            lfq_df,
            num_samples_quadratic=config["search_output"]["num_samples_quadratic"],
            selected_proteins_file=None,
        ).complete_dataframe

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

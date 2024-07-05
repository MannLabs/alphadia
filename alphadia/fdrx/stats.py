# make it an abc
import numpy as np
import pandas as pd


def get_pep(
    psm_df: pd.DataFrame,
    score_column: str = "decoy_proba",
    decoy_column: str = "decoy",
    score_std: float = 0.01,
    pep_granularity: int = 1000,
    kernel_size: int = 20,
):
    """Implementation of a very simple nonparametric PEP estimation using a gaussian kernel.

    Parameters
    ----------

    psm_df : pd.DataFrame
        The dataframe containing the PSMs.

    score_column : str, default='decoy_proba'
        The name of the column containing the score to use for the selection.

    decoy_column : str, default='decoy'
        The name of the column containing the decoy information.

    score_std : float, default=0.01
        The standard deviation of the gaussian kernel.

    pep_granularity : int, default=1000
        The number of bins to use for the score histogram.

    kernel_size : int, default=20
        The size of the kernel to use for the convolution.

    Returns
    -------

    np.ndarray
        The PEP values with same shape and order as the input dataframe.

    """

    score_bins = np.linspace(0, 1, pep_granularity)
    target_decoy = psm_df[decoy_column].values
    score = psm_df[score_column].values

    target_hist, _ = np.histogram(score[target_decoy == 0], bins=score_bins)
    decoy_hist, _ = np.histogram(score[target_decoy == 1], bins=score_bins)

    std_norm = score_std / (score_bins[1] - score_bins[0])
    # create a gaussian kernel of 0.01 width with 5 elements
    kernel_gaussian = np.exp(
        -(np.arange(-kernel_size, kernel_size + 1) ** 2) / (2 * std_norm**2)
    )

    # convolve the target and decoy histograms with the kernel
    target_hist = np.convolve(target_hist, kernel_gaussian, mode="same")
    decoy_hist = np.convolve(decoy_hist, kernel_gaussian, mode="same")

    # numerical stability
    EPSILON = 1e-6
    pep = decoy_hist / (target_hist + decoy_hist + EPSILON)

    return pep[np.digitize(score, score_bins) - 1]


def add_q_values(
    _df: pd.DataFrame,
    decoy_proba_column: str = "decoy_proba",
    decoy_column: str = "decoy",
    qval_column: str = "qval",
    r_target_decoy: float = 1.0,
):
    """Calculates q-values for a dataframe containing PSMs.

    Parameters
    ----------

    _df : pd.DataFrame
        The dataframe containing the PSMs.

    decoy_proba_column : str, default='proba'
        The name of the column containing the probability of being a decoy.
        Value should be between 0 and 1 with 1 being a decoy.

    decoy_column : str, default='_decoy'
        The name of the column containing the decoy information.
        Decoys are expected to be 1 and targets 0.

    qval_column : str, default='qval'
        The name of the column to store the q-values in.

    Returns
    -------

    pd.DataFrame
        The dataframe containing the q-values in column qval.

    """
    EPSILON = 1e-6
    _df = _df.sort_values([decoy_proba_column, decoy_proba_column], ascending=True)

    # translate the decoy probabilities to target probabilities
    target_values = 1 - _df[decoy_column].values
    decoy_cumsum = np.cumsum(_df[decoy_column].values)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum / (target_cumsum + EPSILON)
    _df[qval_column] = fdr_to_q_values(fdr_values) * r_target_decoy
    return _df


def fdr_to_q_values(fdr_values: np.ndarray):
    """Converts FDR values to q-values.
    Takes a ascending sorted array of FDR values and converts them to q-values.
    for every element the lowest FDR where it would be accepted is used as q-value.

    Parameters
    ----------
    fdr_values : np.ndarray
        The FDR values to convert.

    Returns
    -------
    np.ndarray
        The q-values.
    """
    fdr_values_flipped = np.flip(fdr_values)
    q_values_flipped = np.minimum.accumulate(fdr_values_flipped)
    q_vals = np.flip(q_values_flipped)
    return q_vals


def keep_best(
    df: pd.DataFrame,
    score_column: str = "decoy_proba",
    group_columns: list[str] | None = None,
):
    """Keep the best PSM for each group of PSMs with the same precursor_idx.
    This function is used to select the best candidate PSM for each precursor.
    if the group_columns is set to ['channel', 'elution_group_idx'] then its used for target decoy competition.

    Parameters
    ----------

    df : pd.DataFrame
        The dataframe containing the PSMs.

    score_column : str
        The name of the column containing the score to use for the selection.

    group_columns : list[str], default=['channel', 'precursor_idx']
        The columns to use for the grouping.

    Returns
    -------

    pd.DataFrame
        The dataframe containing the best PSM for each group.
    """
    if group_columns is None:
        group_columns = ["channel", "mod_seq_charge_hash"]
    df = df.reset_index(drop=True)
    df = df.sort_values(score_column, ascending=True)
    df = df.groupby(group_columns).head(1)
    df = df.sort_index().reset_index(drop=True)
    return df

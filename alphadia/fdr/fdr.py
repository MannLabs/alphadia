"""Module performing False Discovery Rate (FDR) control."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphadia.fdr.plotting import plot_fdr
from alphadia.fdr.utils import manage_torch_threads, train_test_split_
from alphadia.fragcomp.fragcomp import FragmentCompetition

if TYPE_CHECKING:
    from alphadia.fdr._fdrx.models.two_step_classifier import TwoStepClassifier
    from alphadia.fdr.classifiers import Classifier

max_dia_cycle_shape = 2

logger = logging.getLogger()


@manage_torch_threads(max_threads=2)
def perform_fdr(  # noqa: C901, PLR0913 # too complex, too many arguments
    classifier: Classifier | TwoStepClassifier,
    available_columns: list[str],
    df_target: pd.DataFrame,
    df_decoy: pd.DataFrame,
    *,
    competetive: bool = False,  # TODO: rename all occurrences to `competitive` (also in config -> breaking change)
    group_channels: bool = True,
    figure_path: str | None = None,
    df_fragments: pd.DataFrame | None = None,
    dia_cycle: np.ndarray = None,
    fdr_heuristic: float = 0.1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Performs FDR calculation on a dataframe of PSMs.

     Currently, it does not scale above 2 threads also for large problems, so thread number is limited to 2.

    Parameters
    ----------
    classifier : Classifier | TwoStepClassifier
        A classifier that implements the fit and predict_proba methods

    available_columns : list[str]
        A list of column names that are available for the classifier

    df_target : pd.DataFrame
        A dataframe of target PSMs

    df_decoy : pd.DataFrame
        A dataframe of decoy PSMs

    competetive : bool
        Whether to perform competetive FDR calculation where only the highest scoring PSM in a target-decoy pair is retained

    group_channels : bool
        Whether to group PSMs by channel before performing competetive FDR calculation

    figure_path : str, default=None
        The path to save the FDR plot to

    df_fragments : pd.DataFrame, default=None
        The fragment dataframe.

    dia_cycle : np.ndarray, default=None
        The DIA cycle as provided by alphatims. Required if df_fragments is provided.

    fdr_heuristic : float, default=0.1
        The FDR heuristic to use for the initial selection of PSMs before fragment competition

    random_state : int, optional
        The random state for train-test split reproducibility.

    Returns
    -------
    psm_df : pd.DataFrame
        A dataframe of PSMs with q-values and probabilities.
        The columns `qval` and `proba` are added to the input dataframes.

    """
    target_len, decoy_len = len(df_target), len(df_decoy)
    df_target.dropna(subset=available_columns, inplace=True)
    df_decoy.dropna(subset=available_columns, inplace=True)
    target_dropped, decoy_dropped = (
        target_len - len(df_target),
        decoy_len - len(df_decoy),
    )

    if target_dropped > 0:
        logger.warning(f"dropped {target_dropped} target PSMs due to missing features")

    if decoy_dropped > 0:
        logger.warning(f"dropped {decoy_dropped} decoy PSMs due to missing features")

    if (
        np.abs(len(df_target) - len(df_decoy)) / ((len(df_target) + len(df_decoy)) / 2)
        > 0.1  # noqa: PLR2004
    ):
        logger.warning(
            f"FDR calculation for {len(df_target)} target and {len(df_decoy)} decoy PSMs"
        )
        logger.warning(
            "FDR calculation may be inaccurate as there is more than 10% difference in the number of target and decoy PSMs"
        )

    if random_state is not None:
        logger.info(f"Using random state {random_state} for FDR calculation")

    X_target = df_target[available_columns].to_numpy()
    X_decoy = df_decoy[available_columns].to_numpy()
    y_target = np.zeros(len(X_target))
    y_decoy = np.ones(len(X_decoy))

    X = np.concatenate([X_target, X_decoy])
    y = np.concatenate([y_target, y_decoy])

    X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split_(
        X, y, test_size=0.2, random_state=random_state
    )

    classifier.fit(X_train, y_train)

    psm_df = pd.concat([df_target, df_decoy])
    psm_df["_decoy"] = y

    if competetive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]

    predicted_proba = classifier.predict_proba(X)[:, 1]

    psm_df["proba"] = predicted_proba
    psm_df.sort_values(
        ["proba", "precursor_idx"], ascending=True, inplace=True
    )  # last sort to break ties

    psm_df = get_q_values(psm_df, "proba", "_decoy")

    if dia_cycle is not None and dia_cycle.shape[2] <= max_dia_cycle_shape:
        # use a FDR of 10% as starting point
        # if there are no PSMs with a FDR < 10% use all PSMs
        start_idx = psm_df["qval"].searchsorted(fdr_heuristic, side="left")
        if start_idx == 0:
            start_idx = len(psm_df)

        # make sure fragments are not reused
        if df_fragments is not None:
            if dia_cycle is None:
                raise ValueError(
                    "dia_cycle must be provided if df_fragments is provided"
                )
            fragment_competition = FragmentCompetition()
            psm_df = fragment_competition(
                psm_df.iloc[:start_idx], df_fragments, dia_cycle
            )

    psm_df = keep_best(psm_df, group_columns=group_columns)
    psm_df = get_q_values(psm_df, "proba", "_decoy")

    if figure_path is not None:
        plot_fdr(
            y_train,
            y_test,
            predicted_proba[idxs_train],
            predicted_proba[idxs_test],
            psm_df["qval"],
            figure_path=figure_path,
        )

    return psm_df


def keep_best(
    df: pd.DataFrame,
    score_column: str = "proba",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
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
        group_columns = ["channel", "precursor_idx"]
    df = df.reset_index(drop=True)
    df = df.sort_values(
        [score_column, *group_columns], ascending=True
    )  # last sort to break ties
    df = df.groupby(group_columns).head(1)
    return df.sort_index().reset_index(drop=True)


def _fdr_to_q_values(fdr_values: np.ndarray) -> np.ndarray:
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
    return np.flip(q_values_flipped)


def get_q_values(
    df: pd.DataFrame,
    score_column: str = "proba",
    decoy_column: str = "_decoy",
    qval_column: str = "qval",
    extra_sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Calculates q-values for a dataframe containing PSMs.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the PSMs.

    score_column : str, default='proba'
        The name of the column containing the score to use for the selection.
        Ascending sorted values are expected.

    decoy_column : str, default='_decoy'
        The name of the column containing the decoy information.
        Decoys are expected to be 1 and targets 0.

    qval_column : str, default='qval'
        The name of the column to store the q-values in.

    extra_sort_columns : list[str], default=['precursor_idx']
        Additional columns to sort by after score_column and decoy_column to break ties.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the q-values in column qval.

    """
    if extra_sort_columns is None:
        extra_sort_columns = ["precursor_idx"]

    df = df.sort_values(
        [score_column, decoy_column, *extra_sort_columns], ascending=True
    )  # last sort to break ties
    target_values = 1 - df[decoy_column].to_numpy()
    decoy_cumsum = np.cumsum(df[decoy_column].to_numpy())
    target_cumsum = np.cumsum(target_values)
    fdr_values = (
        decoy_cumsum / target_cumsum
    )  # TODO: RuntimeWarning: divide by zero encountered in divide
    df[qval_column] = _fdr_to_q_values(fdr_values)
    return df

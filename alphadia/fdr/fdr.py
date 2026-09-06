"""Module performing False Discovery Rate (FDR) control."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphadia.exceptions import TooFewPSMError
from alphadia.fdr.plotting import plot_fdr
from alphadia.fdr.utils import manage_torch_threads, train_test_split_
from alphadia.fragcomp.fragcomp import compete_for_fragments

if TYPE_CHECKING:
    from alphadia.fdr.classifiers import Classifier
    from alphadia.fdr.prefilter import CascadePrefilter
    from alphadia.fdr.semisupervised import SelfTrainer, TrainingResult

max_dia_cycle_shape = 2

logger = logging.getLogger()

# Below this standard deviation the probability is almost constant. The classifier then
# cannot separate targets from decoys, and the FDR filter keeps all PSMs.
_PROBA_COLLAPSE_STD_THRESHOLD = 1e-4

_MAX_FDR_CLASSIFIER_REINITS = 3

# Fraction of the gap between the worst scored PSM and 1.0 left empty above the scored
# PSMs, so a dropped PSM with stage-1 probability 0 still ranks strictly behind them.
_DROPPED_PROBA_OFFSET = 0.5

_DECOY_WEIGHT_COLUMN = "_decoy_weight"


@manage_torch_threads(max_threads=2)
def perform_fdr(  # noqa: C901, PLR0912, PLR0913, PLR0915 # too complex, too many branches, too many statements, too many arguments
    classifier: Classifier,
    available_columns: list[str],
    df_target: pd.DataFrame,
    df_decoy: pd.DataFrame,
    *,
    competitive: bool = False,
    group_channels: bool = True,
    figure_path: str | None = None,
    df_fragments: pd.DataFrame | None = None,
    dia_cycle: np.ndarray | None = None,
    fdr_heuristic: float = 0.1,
    random_state: int | None = None,
    is_final: bool = False,
    prefilter: CascadePrefilter | None = None,
    trainer: SelfTrainer | None = None,
) -> pd.DataFrame:
    """Performs FDR calculation on a dataframe of PSMs.

     Currently, it does not scale above 2 threads also for large problems, so thread number is limited to 2.

    Parameters
    ----------
    classifier : Classifier
        A classifier that implements the fit and predict_proba methods

    available_columns : list[str]
        A list of column names that are available for the classifier

    df_target : pd.DataFrame
        A dataframe of target PSMs

    df_decoy : pd.DataFrame
        A dataframe of decoy PSMs

    competitive : bool
        Whether to perform competitive FDR calculation where only the highest scoring PSM in a target-decoy pair is retained

    group_channels : bool
        Whether to group PSMs by channel before performing competitive FDR calculation

    figure_path : str, default=None
        The path to save the FDR plot to

    df_fragments : pd.DataFrame, default=None
        The fragment dataframe.

    dia_cycle : np.ndarray, default=None
        The DIA cycle. Required if df_fragments is provided.

    fdr_heuristic : float, default=0.1
        The FDR heuristic to use for the initial selection of PSMs before fragment competition

    random_state : int, optional
        The random state for train-test split reproducibility.

    is_final : bool, default=False
        Whether this is the FDR round whose scores are reported, rather than one of the
        optimization rounds.

    prefilter : CascadePrefilter, default=None
        Gate that decides which PSMs the classifier is fitted on and scores. PSMs it
        drops are ranked behind every scored PSM, in the order of its own scores.

    trainer : SelfTrainer, default=None
        Fits the classifier of the final round so that the decoy count stays honest,
        by cross-fitting or by hiding a share of the decoys, and decides the weight of
        every decoy in the q-values. None fits the classifier on every PSM and counts
        every decoy.

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

    psm_df = pd.concat([df_target, df_decoy])
    psm_df["_decoy"] = y

    if competitive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]

    if prefilter is None:
        keep = np.ones(len(X), dtype=bool)
        X_kept = X
    else:
        keep, stage1_proba = prefilter.select(psm_df, y, is_final=is_final)
        X_kept = X[keep]
        if prefilter.reset_classifier and not keep.all():
            classifier.reset()

    decoy_weight_column = None
    if trainer is not None and is_final:
        # The optimization rounds fitted the classifier on PSMs of this round; a warm
        # start would carry what it memorized about them into the honest fit.
        classifier.reset()
        precursor_idx = psm_df["precursor_idx"].to_numpy()
        decoy_weight = trainer.prepare(y, precursor_idx)
        competition_group = psm_df.groupby(group_columns).ngroup().to_numpy()
        results: list[TrainingResult] = []

        def fit_and_score() -> np.ndarray:
            results.append(
                trainer.fit_predict(
                    classifier,
                    X_kept,
                    y[keep],
                    decoy_weight[keep],
                    competition_group[keep],
                    precursor_idx[keep],
                    is_final=is_final,
                )
            )
            return results[-1].proba

        predicted_proba = _fit_until_separated(fit_and_score, classifier)

        idxs_train, y_train = results[-1].train_idx, results[-1].y_train
        idxs_test = np.setdiff1d(np.arange(len(X_kept)), idxs_train)
        y_test = y[keep][idxs_test]

        psm_df[_DECOY_WEIGHT_COLUMN] = decoy_weight
        decoy_weight_column = _DECOY_WEIGHT_COLUMN
    else:
        try:
            X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split_(
                X_kept, y[keep], test_size=0.2, random_state=random_state
            )
        except TooFewPSMError:
            logger.warning(
                "Too few PSMs for FDR classification, assigning qval=1.0 and proba=1.0 to all PSMs."
            )
            psm_df["qval"] = 1.0
            psm_df["proba"] = 1.0
            return psm_df

        def fit_and_score() -> np.ndarray:
            classifier.fit(X_train, y_train, is_final=is_final)
            return classifier.predict_proba(X_kept)[:, 1]

        predicted_proba = _fit_until_separated(fit_and_score, classifier)

    proba = np.empty(len(X))
    proba[keep] = predicted_proba
    if prefilter is not None:
        # Dropped PSMs never reach the FDR threshold, so their exact scores do not matter,
        # only that every one of them ranks behind every scored PSM. Spreading them by
        # their stage-1 score keeps them distinct, so a tied block cannot form in the tail.
        worst_kept = predicted_proba.max()
        proba[~keep] = worst_kept + (1 - worst_kept) * (
            _DROPPED_PROBA_OFFSET + (1 - _DROPPED_PROBA_OFFSET) * stage1_proba[~keep]
        )

    psm_df["proba"] = proba
    psm_df.sort_values(
        ["proba", "precursor_idx"], ascending=True, inplace=True
    )  # last sort to break ties

    psm_df = get_q_values(
        psm_df, "proba", "_decoy", decoy_weight_column=decoy_weight_column
    )

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
            psm_df = compete_for_fragments(
                psm_df.iloc[:start_idx], df_fragments, dia_cycle
            )

    psm_df = keep_best(psm_df, group_columns=group_columns)
    psm_df = get_q_values(
        psm_df, "proba", "_decoy", decoy_weight_column=decoy_weight_column
    )

    if decoy_weight_column is not None:
        psm_df.drop(columns=[decoy_weight_column], inplace=True)

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


def _fit_until_separated(
    fit_and_score: Callable[[], np.ndarray], classifier: Classifier
) -> np.ndarray:
    """Fit and score, starting over from fresh weights while the scores are near-constant.

    A collapse is usually an unlucky set of start weights, so a new fit recovers it.
    """
    predicted_proba = fit_and_score()

    n_reinit = 0
    while (
        float(np.std(predicted_proba)) < _PROBA_COLLAPSE_STD_THRESHOLD
        and n_reinit < _MAX_FDR_CLASSIFIER_REINITS
    ):
        n_reinit += 1
        logger.warning(
            f"FDR classifier collapsed to a near-constant probability "
            f"({np.unique(predicted_proba).size} unique value(s) over "
            f"{len(predicted_proba):,} PSMs); reinitializing from scratch and "
            f"retrying ({n_reinit}/{_MAX_FDR_CLASSIFIER_REINITS})."
        )
        classifier.reset()
        predicted_proba = fit_and_score()

    if float(np.std(predicted_proba)) < _PROBA_COLLAPSE_STD_THRESHOLD:
        logger.warning(
            "FDR classifier produced a near-constant probability; target/decoy "
            "separation failed and q-values will not filter PSMs."
        )

    return predicted_proba


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
    decoy_weight_column: str | None = None,
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

    decoy_weight_column : str, optional
        Column holding the weight each PSM adds to the decoy count, zero for targets.
        None counts every decoy once.

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
    decoy_cumsum = np.cumsum(
        df[decoy_column if decoy_weight_column is None else decoy_weight_column]
        .to_numpy()
        .astype(float)
    )
    target_cumsum = np.cumsum(target_values)
    fdr_values = np.divide(
        decoy_cumsum,
        target_cumsum,
        out=np.ones(len(df), dtype=float),
        where=target_cumsum > 0,
    )

    # The sort above puts every target of a tied-score block ahead of every decoy, so a
    # running ratio taken mid-block sees only the targets and reads far too low. With
    # many features the scores are near-continuous and blocks are 2-3 PSMs wide, but a
    # small feature subset emits few distinct probabilities and a single block can hold
    # most of the data, which collapses the q-values. Charge every member of a block the
    # ratio as it stands once the whole block is accepted.
    scores = df[score_column].to_numpy()
    block_end = np.searchsorted(scores, scores, side="right") - 1
    fdr_values = fdr_values[block_end]

    df[qval_column] = _fdr_to_q_values(fdr_values)
    return df

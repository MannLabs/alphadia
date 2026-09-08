"""Module performing False Discovery Rate (FDR) control."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphadia.exceptions import TooFewPSMError
from alphadia.fdr.plotting import plot_fdr
from alphadia.fdr.utils import manage_torch_threads, train_test_split_
from alphadia.fragcomp.fragcomp import compete_for_fragments

if TYPE_CHECKING:
    from alphadia.fdr.classifiers import Classifier
    from alphadia.fdr.cross_fitting import CrossFittedTrainer
    from alphadia.fdr.prefilter import CascadePrefilter

max_dia_cycle_shape = 2

_STAGE1_RANK_COLUMN = "_stage1_rank"

logger = logging.getLogger()

# Fraction of the gap between the worst scored PSM and 1.0 left empty above the scored
# PSMs, so a dropped PSM with stage-1 probability 0 still ranks strictly behind them.
_DROPPED_PROBA_OFFSET = 0.5

# The prefilter is judged by the identifications it let through: the share of them that
# sit in the worst-ranked tenth of what it kept. A gate with margin sees the
# identification density die out well before its cut; one that truncates the
# identifications is still dense at the cut. Final rounds on HeLa, with and without an
# entrapment library, land at 0.08-0.46 % under the shipped threshold, while plasma lands
# at 0.69-1.28 % and gains 21 % identifications once the cut is widened to the threshold
# below. Only the final round is judged: the optimization rounds identify a few thousand
# PSMs and their share is noise.
_RECALL_CHECK_FDR = 0.01
_RECALL_CHECK_TAIL_FRACTION = 0.1
_RECALL_WIDEN_TAIL_SHARE = 0.005
_WIDE_Q_VALUE_THRESHOLD = 0.5


@dataclass
class _Fit:
    """Scores of the kept PSMs and the split the classifier was fitted on."""

    proba: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@manage_torch_threads(max_threads=2)
def perform_fdr(  # noqa: C901, PLR0913, PLR0915 # too complex, too many arguments, too many statements
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
    trainer: CrossFittedTrainer | None = None,
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
        drops are ranked behind every scored PSM, in the order of its own scores. When
        the final round's identifications crowd the gate's cut, the cut is widened once
        and the classifier refitted on the wider set.

    trainer : CrossFittedTrainer, default=None
        Fits the classifier in the final round instead of a plain fit on a random split.
        The optimization rounds keep the plain fit, as their scores only steer the
        calibration.

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
        stage1_proba = np.zeros(len(X))
    else:
        keep, stage1_proba, n_passed = prefilter.select(psm_df, y, is_final=is_final)
        # the PSMs that passed the cut are a prefix of this order, see CascadePrefilter.select
        stage1_order = np.lexsort((psm_df["precursor_idx"].to_numpy(), y, stage1_proba))
        stage1_rank = np.empty(len(X), dtype=np.int64)
        stage1_rank[stage1_order] = np.arange(len(X))
        psm_df[_STAGE1_RANK_COLUMN] = stage1_rank

    precursor_idx = psm_df["precursor_idx"].to_numpy()
    cross_fitted = trainer is not None and is_final
    competition_group = (
        psm_df.groupby(group_columns).ngroup().to_numpy() if cross_fitted else None
    )

    def fit(keep: np.ndarray) -> _Fit:
        x_kept = X[keep]

        if cross_fitted:
            # The optimization rounds fitted the classifier on PSMs of this round; a
            # warm start would carry what it memorized about them into the out-of-fold
            # fit.
            classifier.reset()
            result = trainer.fit_predict(
                classifier,
                x_kept,
                y[keep],
                competition_group[keep],
                precursor_idx[keep],
                is_final=is_final,
            )
            test_idx = np.setdiff1d(np.arange(len(x_kept)), result.train_idx)
            return _Fit(
                result.proba,
                result.train_idx,
                test_idx,
                result.y_train,
                y[keep][test_idx],
            )

        X_train, _, y_train, y_test, train_idx, test_idx = train_test_split_(
            x_kept, y[keep], test_size=0.2, random_state=random_state
        )

        classifier.fit_separating(X_train, y_train, is_final=is_final)
        proba = classifier.predict_proba(x_kept)[:, 1]
        return _Fit(proba, train_idx, test_idx, y_train, y_test)

    def score(fit_result: _Fit, keep: np.ndarray) -> pd.DataFrame:
        proba = np.empty(len(X))
        proba[keep] = fit_result.proba
        if prefilter is not None:
            # Dropped PSMs never reach the FDR threshold, so their exact scores do not
            # matter, only that every one of them ranks behind every scored PSM.
            # Spreading them by their stage-1 score keeps them distinct, so a tied block
            # cannot form in the tail.
            worst_kept = fit_result.proba.max()
            proba[~keep] = worst_kept + (1 - worst_kept) * (
                _DROPPED_PROBA_OFFSET
                + (1 - _DROPPED_PROBA_OFFSET) * stage1_proba[~keep]
            )
        psm_df["proba"] = proba

        scored_df = get_q_values(psm_df, "proba", "_decoy")

        if dia_cycle is not None and dia_cycle.shape[2] <= max_dia_cycle_shape:
            # use a FDR of 10% as starting point
            # if there are no PSMs with a FDR < 10% use all PSMs
            start_idx = scored_df["qval"].searchsorted(fdr_heuristic, side="left")
            if start_idx == 0:
                start_idx = len(scored_df)

            # make sure fragments are not reused
            if df_fragments is not None:
                if dia_cycle is None:
                    raise ValueError(
                        "dia_cycle must be provided if df_fragments is provided"
                    )
                scored_df = compete_for_fragments(
                    scored_df.iloc[:start_idx], df_fragments, dia_cycle
                )

        scored_df = keep_best(scored_df, group_columns=group_columns)
        return get_q_values(scored_df, "proba", "_decoy")

    try:
        fit_result = fit(keep)
    except TooFewPSMError:
        logger.warning(
            "Too few PSMs for FDR classification, assigning qval=1.0 and proba=1.0 to all PSMs."
        )
        psm_df["qval"] = 1.0
        psm_df["proba"] = 1.0
        return psm_df.drop(columns=_STAGE1_RANK_COLUMN, errors="ignore")

    scored_df = score(fit_result, keep)

    if prefilter is not None and is_final and not keep.all():
        tail_share = _prefilter_tail_share(scored_df, n_passed)
        if tail_share > _RECALL_WIDEN_TAIL_SHARE:
            logger.warning(
                f"{100 * tail_share:.2f}% of the identifications sit in the worst "
                f"{_RECALL_CHECK_TAIL_FRACTION:.0%} of the PSMs that passed the prefilter, "
                f"widening the cut to stage-1 q-value <= {_WIDE_Q_VALUE_THRESHOLD} "
                f"and refitting"
            )
            keep, n_passed = prefilter.keep_from_scores(
                stage1_proba,
                y,
                precursor_idx,
                psm_df["elution_group_idx"].to_numpy(),
                _WIDE_Q_VALUE_THRESHOLD,
            )
            fit_result = fit(keep)
            scored_df = score(fit_result, keep)
            tail_share = _prefilter_tail_share(scored_df, n_passed)
        logger.info(
            f"Prefilter recall check: {100 * tail_share:.2f}% of the identifications at "
            f"{_RECALL_CHECK_FDR:.0%} FDR sit in the worst "
            f"{_RECALL_CHECK_TAIL_FRACTION:.0%} of the PSMs that passed the prefilter"
        )
    scored_df = scored_df.drop(columns=_STAGE1_RANK_COLUMN, errors="ignore")

    if figure_path is not None:
        plot_fdr(
            fit_result.y_train,
            fit_result.y_test,
            fit_result.proba[fit_result.train_idx],
            fit_result.proba[fit_result.test_idx],
            scored_df["qval"],
            figure_path=figure_path,
        )

    return scored_df


def _prefilter_tail_share(psm_df: pd.DataFrame, n_passed: int) -> float:
    """Share of the identifications that sit in the worst-ranked tail of the PSMs that
    passed the stage-1 cut.

    A gate cannot be told from inside whether it dropped PSMs the classifier would have
    identified; this is the next best thing: identifications that fill the tail of what
    passed mean the density was still high where it stopped. PSMs kept only for their
    elution group rank behind the cut and are not part of its tail.
    """
    identified = psm_df[(psm_df["_decoy"] == 0) & (psm_df["qval"] <= _RECALL_CHECK_FDR)]
    if len(identified) == 0:
        return 0.0
    stage1_rank = identified[_STAGE1_RANK_COLUMN]
    in_tail = (stage1_rank >= (1 - _RECALL_CHECK_TAIL_FRACTION) * n_passed) & (
        stage1_rank < n_passed
    )
    return float(in_tail.mean())


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

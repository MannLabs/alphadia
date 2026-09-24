"""Module performing False Discovery Rate (FDR) control."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from alphadia.exceptions import TooFewPSMError
from alphadia.fdr.plotting import plot_fdr
from alphadia.fdr.utils import manage_torch_threads, train_test_split_
from alphadia.fragcomp.fragcomp import compete_for_fragments

if TYPE_CHECKING:
    from alphadia.fdr.classifiers import Classifier
    from alphadia.fdr.prefilter import CascadePrefilter

max_dia_cycle_shape = 2

logger = logging.getLogger()

# Below this standard deviation the probability is almost constant. The classifier then
# cannot separate targets from decoys, and the FDR filter keeps all PSMs.
_PROBA_COLLAPSE_STD_THRESHOLD = 1e-4

_MAX_FDR_CLASSIFIER_REINITS = 3

# Folds of the cross-fitted classifier; each fold model is fitted on 80 % of the PSMs, as
# many as the single in-sample fit it replaces.
_CROSS_FIT_FOLDS = 5

# Torch threads of one fold's worker process, the cap the FDR task runs under anyway.
_CROSS_FIT_TORCH_THREADS = 2

# Fraction of the gap between the worst scored PSM and 1.0 left empty above the scored
# PSMs, so a dropped PSM with stage-1 probability 0 still ranks strictly behind them.
_DROPPED_PROBA_OFFSET = 0.5


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
    fragment_provider: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    dia_cycle: np.ndarray | None = None,
    fdr_heuristic: float = 0.1,
    random_state: int | None = None,
    prefilter: CascadePrefilter | None = None,
    cross_fit: bool = False,
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

    fragment_provider : Callable[[pd.DataFrame], pd.DataFrame], default=None
        Returns the fragment dataframe of the PSMs it is given. Used instead of
        `df_fragments` when the fragments are only quantified on demand; it is called
        with the PSMs that enter fragment competition.

    dia_cycle : np.ndarray, default=None
        The DIA cycle. Required if df_fragments is provided.

    fdr_heuristic : float, default=0.1
        The FDR heuristic to use for the initial selection of PSMs before fragment competition

    random_state : int, optional
        The random state for train-test split reproducibility.

    prefilter : CascadePrefilter, default=None
        Gate that decides which PSMs the classifier is fitted on and scores. PSMs it
        drops are ranked behind every scored PSM, in the order of its own scores.

    cross_fit : bool, default=False
        Score every PSM with a model that was fitted from fresh weights on the other folds,
        instead of with one model that also scores the PSMs it was fitted on. The passed
        classifier ends up holding the model of the first fold.

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

    if prefilter is None:
        keep = np.ones(len(X), dtype=bool)
        X_kept = X
    else:
        keep, stage1_proba = prefilter.select(psm_df, y)
        X_kept = X[keep]

    y_kept = y[keep]
    try:
        X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split_(
            X_kept, y_kept, test_size=0.2, random_state=random_state
        )
    except TooFewPSMError:
        logger.warning(
            "Too few PSMs for FDR classification, assigning qval=1.0 and proba=1.0 to all PSMs."
        )
        psm_df["qval"] = 1.0
        psm_df["proba"] = 1.0
        return psm_df

    if cross_fit:
        predicted_proba, fold = _cross_fitted_proba(
            classifier, X_kept, y_kept, random_state
        )
        # for the diagnostic plot fold 0 plays the test set; all probabilities are out-of-fold
        idxs_test = np.flatnonzero(fold == 0)
        idxs_train = np.flatnonzero(fold != 0)
        y_train, y_test = y_kept[idxs_train], y_kept[idxs_test]
    else:
        predicted_proba = _fit_predict(classifier, X_train, y_train, X_kept)

    if competitive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]

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
    # No sort here: get_q_values sorts by proba, _decoy and precursor_idx, and a stable
    # sort on those leaves tied PSMs in the order they came in either way.
    psm_df = get_q_values(psm_df, "proba", "_decoy")

    if dia_cycle is not None and dia_cycle.shape[2] <= max_dia_cycle_shape:
        # use a FDR of 10% as starting point
        # if there are no PSMs with a FDR < 10% use all PSMs
        start_idx = psm_df["qval"].searchsorted(fdr_heuristic, side="left")
        if start_idx == 0:
            start_idx = len(psm_df)

        if df_fragments is None and fragment_provider is not None:
            df_fragments = fragment_provider(psm_df.iloc[:start_idx])

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


def _cross_fitted_proba(
    classifier: Classifier,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold decoy probability of every row, and the fold of every row."""
    fold = np.random.default_rng(random_state).permutation(len(X)) % _CROSS_FIT_FOLDS
    # Fresh weights: a warm start carries what earlier rounds learned from the labels of
    # these same precursors, so a fold model would not be blind to the rows it scores.
    fresh = deepcopy(classifier)
    fresh.reset()
    tasks = [
        (deepcopy(fresh), X[fold != fold_idx], y[fold != fold_idx], X[fold == fold_idx])
        for fold_idx in range(_CROSS_FIT_FOLDS)
    ]
    # The folds are fitted at the same time so that cross-fitting costs no wall time. The
    # training loop is bound by the GIL, so threads do not run in parallel, and forking a
    # process that has already run torch deadlocks, hence spawned worker processes.
    with ProcessPoolExecutor(
        max_workers=_CROSS_FIT_FOLDS, mp_context=get_context("spawn")
    ) as pool:
        results = list(pool.map(_fit_predict_in_worker, tasks))

    predicted_proba = np.empty(len(X))
    for fold_idx, (fold_proba, _) in enumerate(results):
        predicted_proba[fold == fold_idx] = fold_proba
    classifier.from_state_dict(results[0][1].to_state_dict())
    return predicted_proba, fold


def _fit_predict_in_worker(
    task: tuple[Classifier, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, Classifier]:
    """Fit one fold in a worker process; returns its probabilities and the fitted model."""
    classifier, X_train, y_train, X_score = task
    torch.set_num_threads(_CROSS_FIT_TORCH_THREADS)
    return _fit_predict(classifier, X_train, y_train, X_score), classifier


def _fit_predict(
    classifier: Classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_score: np.ndarray,
) -> np.ndarray:
    """Fit the classifier and return its decoy probability for `X_score`, refitting from scratch on a collapse."""
    classifier.fit(X_train, y_train)
    predicted_proba = classifier.predict_proba(X_score)[:, 1]

    # A collapse is usually an unlucky set of start weights, so a new fit recovers it.
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
        classifier.fit(X_train, y_train)
        predicted_proba = classifier.predict_proba(X_score)[:, 1]

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


def q_values_of(
    scores: np.ndarray, decoys: np.ndarray, decoy_offset: int = 0
) -> np.ndarray:
    """Calculates the q-value of every PSM, in the order the PSMs are given in.

    Parameters
    ----------
    scores : np.ndarray
        Score of every PSM, ascending, lower is better.

    decoys : np.ndarray
        Decoy information of every PSM, 1 for decoys and 0 for targets.

    decoy_offset : int, default=0
        Added to the running decoy count before dividing by the running target count.

    Returns
    -------
    np.ndarray
        The q-value of every PSM.

    """
    # Ordering PSMs of one score against each other would put every target of the block
    # ahead of every decoy, so a running ratio taken mid-block sees only the targets and
    # reads far too low. With many features the scores are near-continuous and blocks are
    # 2-3 PSMs wide, but a small feature subset emits few distinct probabilities and a
    # single block can hold most of the data, which collapses the q-values. Charging every
    # member of a block the ratio as it stands once the whole block is accepted makes the
    # q-value a property of the block, so the counting runs over the distinct scores
    # rather than over the PSMs, and no PSM has to be ordered against a tied one.
    _, block_of_psm = np.unique(scores, return_inverse=True)
    decoy_cumsum = np.cumsum(np.bincount(block_of_psm, weights=decoys))
    target_cumsum = np.cumsum(np.bincount(block_of_psm, weights=1 - decoys))
    fdr_values = np.divide(
        decoy_cumsum + decoy_offset,
        target_cumsum,
        out=np.ones(len(decoy_cumsum), dtype=float),
        where=target_cumsum > 0,
    )
    return _fdr_to_q_values(fdr_values)[block_of_psm]


def get_q_values(
    df: pd.DataFrame,
    score_column: str = "proba",
    decoy_column: str = "_decoy",
    qval_column: str = "qval",
    extra_sort_columns: list[str] | None = None,
    decoy_offset: int = 0,
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

    decoy_offset : int, default=0
        Added to the running decoy count before dividing by the running target count.

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
    df[qval_column] = q_values_of(
        df[score_column].to_numpy(), df[decoy_column].to_numpy(), decoy_offset
    )
    return df

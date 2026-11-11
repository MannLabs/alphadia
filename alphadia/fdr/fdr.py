"""Module performing False Discovery Rate (FDR) control."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from alphadia.exceptions import TooFewPSMError
from alphadia.fdr.plotting import plot_fdr
from alphadia.fdr.utils import manage_torch_threads, train_test_split_indices
from alphadia.fragcomp.fragcomp import compete_for_fragments

if TYPE_CHECKING:
    import pandas as pd

    from alphadia.fdr.classifiers import Classifier

max_dia_cycle_shape = 2

logger = logging.getLogger()

# PSMs below this FDR enter fragment competition, the others are dropped
FDR_HEURISTIC = 0.1

# Below this standard deviation the probability is almost constant. The classifier then
# cannot separate targets from decoys, and the FDR filter keeps all PSMs.
_PROBA_COLLAPSE_STD_THRESHOLD = 1e-4

# The classifier can also collapse only partially: it assigns the same, best probability to a large share of
# PSMs, while the remaining probabilities stay spread out. The standard deviation check does not detect this, but
# the PSMs sharing the best probability can no longer be ranked against each other. In a healthy fit, only a few
# PSMs share the best probability.
_MAX_NUM_PSMS_AT_BEST_PROBA = 500

_MAX_FDR_CLASSIFIER_REINITS = 3

# With `train_on_best_candidate`, the classifier trains on the best candidate of each of these groups.
BEST_CANDIDATE_GROUP_COLUMN = "precursor_idx"
_BEST_CANDIDATE_MAX_ITER = 100


def _best_candidate_positions(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, random_state: int | None
) -> np.ndarray:
    """Sorted positions of each group's row with the lowest decoy logit of a logistic regression fit on all rows."""
    logistic_regression = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=_BEST_CANDIDATE_MAX_ITER, random_state=random_state
        ),
    )
    logistic_regression.fit(x, y)
    decoy_logit = logistic_regression.decision_function(x)
    order = np.lexsort((decoy_logit, groups))
    is_first_of_group = np.ones(len(order), dtype=bool)
    is_first_of_group[1:] = groups[order][1:] != groups[order][:-1]
    return np.sort(order[is_first_of_group])


def _stratified_subsample(
    idxs: np.ndarray, y: np.ndarray, max_samples: int, random_state: int | None
) -> np.ndarray:
    """Sorted random subset of `idxs` of at most `max_samples` rows that keeps the share of decoys (`y` of the rows)."""
    if len(idxs) <= max_samples:
        return idxs
    rng = np.random.default_rng(random_state)
    fraction = max_samples / len(idxs)
    subsets = [
        rng.choice(
            idxs[y == label], int(round(fraction * np.sum(y == label))), replace=False
        )
        for label in (0, 1)
    ]
    return np.sort(np.concatenate(subsets))


def _is_collapsed(proba: np.ndarray) -> bool:
    """Whether the probabilities are near-constant, or too many PSMs share the best probability."""
    return (
        float(np.std(proba)) < _PROBA_COLLAPSE_STD_THRESHOLD
        or int(np.sum(proba == proba.min())) > _MAX_NUM_PSMS_AT_BEST_PROBA
    )


@manage_torch_threads(max_threads=2)
def perform_fdr(  # noqa: C901, PLR0912, PLR0913, PLR0915 # too complex, too many branches, too many statements, too many arguments
    classifier: Classifier,
    available_columns: list[str],
    psm_df: pd.DataFrame,
    decoy_mask: np.ndarray,
    *,
    competitive: bool = False,
    group_channels: bool = True,
    figure_path: str | None = None,
    df_fragments: pd.DataFrame | None = None,
    fragment_provider: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    dia_cycle: np.ndarray | None = None,
    train_on_best_candidate: bool = False,
    fdr_heuristic: float = FDR_HEURISTIC,
    max_train_samples: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Performs FDR calculation on a dataframe of target and decoy PSMs.

     Currently, it does not scale above 2 threads also for large problems, so thread number is limited to 2.
     `psm_df` is not modified: the PSMs are copied once, the targets ahead of the decoys.

    Parameters
    ----------
    classifier : Classifier
        A classifier that implements the fit and predict_proba methods

    available_columns : list[str]
        A list of column names that are available for the classifier

    psm_df : pd.DataFrame
        A dataframe of target and decoy PSMs

    decoy_mask : np.ndarray, dtype=bool
        Whether each row of `psm_df` is a decoy of the target-decoy competition

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

    train_on_best_candidate : bool, default=False
        Train the classifier only on the most target-like candidate of each precursor, as picked by a logistic
        regression on all training rows. All rows are still scored.

    fdr_heuristic : float, default=FDR_HEURISTIC
        The FDR heuristic to use for the initial selection of PSMs before fragment competition.
        PSMs above it are dropped when fragments compete.

    max_train_samples : int, optional
        Fit the classifier on a random subset of at most this many training rows, with the same share of decoys.
        All rows are still scored. By default, all training rows are used.

    random_state : int, optional
        The random state for train-test split reproducibility.

    Returns
    -------
    psm_df : pd.DataFrame
        A dataframe of PSMs with q-values and probabilities.
        The columns `qval` and `proba` are added.

    """
    complete = psm_df[available_columns].notna().all(axis=1).to_numpy()
    target_rows = np.flatnonzero(~decoy_mask & complete)
    decoy_rows = np.flatnonzero(decoy_mask & complete)
    target_dropped = int((~decoy_mask).sum()) - len(target_rows)
    decoy_dropped = int(decoy_mask.sum()) - len(decoy_rows)

    if target_dropped > 0:
        logger.warning(f"dropped {target_dropped} target PSMs due to missing features")

    if decoy_dropped > 0:
        logger.warning(f"dropped {decoy_dropped} decoy PSMs due to missing features")

    n_target, n_decoy = len(target_rows), len(decoy_rows)
    if np.abs(n_target - n_decoy) / ((n_target + n_decoy) / 2) > 0.1:  # noqa: PLR2004
        logger.warning(
            f"FDR calculation for {n_target} target and {n_decoy} decoy PSMs"
        )
        logger.warning(
            "FDR calculation may be inaccurate as there is more than 10% difference in the number of target and decoy PSMs"
        )

    if random_state is not None:
        logger.info(f"Using random state {random_state} for FDR calculation")

    psm_df = psm_df.take(np.concatenate([target_rows, decoy_rows]))
    X = psm_df[available_columns].to_numpy()
    y = np.concatenate([np.zeros(n_target), np.ones(n_decoy)])

    try:
        idxs_train, idxs_test = train_test_split_indices(
            len(X), test_size=0.2, random_state=random_state
        )
    except TooFewPSMError:
        logger.warning(
            "Too few PSMs for FDR classification, assigning qval=1.0 and proba=1.0 to all PSMs."
        )
        psm_df["qval"] = 1.0
        psm_df["proba"] = 1.0
        return psm_df

    y_train, y_test = y[idxs_train], y[idxs_test]
    # The other candidates of a precursor are mostly noise peaks whose target label is wrong.
    idxs_fit = idxs_train
    if train_on_best_candidate:
        idxs_fit = idxs_train[
            _best_candidate_positions(
                X[idxs_train],
                y_train,
                psm_df[BEST_CANDIDATE_GROUP_COLUMN].to_numpy()[idxs_train],
                random_state,
            )
        ]
        logger.info(
            f"training on the best candidate of each precursor: {len(idxs_fit):,} of {len(idxs_train):,} rows"
        )
    if max_train_samples is not None and len(idxs_fit) > max_train_samples:
        n_fit = len(idxs_fit)
        idxs_fit = _stratified_subsample(
            idxs_fit, y[idxs_fit], max_train_samples, random_state
        )
        logger.info(f"training on a random subset: {len(idxs_fit):,} of {n_fit:,} rows")
    classifier.fit(X[idxs_fit], y[idxs_fit])

    psm_df["_decoy"] = y

    if competitive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]

    predicted_proba = classifier.predict_proba(X)[:, 1]

    # A collapse is usually an unlucky set of start weights, so a new fit recovers it.
    n_reinit = 0
    while _is_collapsed(predicted_proba) and n_reinit < _MAX_FDR_CLASSIFIER_REINITS:
        n_reinit += 1
        logger.warning(
            f"FDR classifier collapsed ({np.unique(predicted_proba).size:,} unique "
            f"value(s) over {len(predicted_proba):,} PSMs, "
            f"{int(np.sum(predicted_proba == predicted_proba.min())):,} at the best "
            f"one); reinitializing from scratch and retrying "
            f"({n_reinit}/{_MAX_FDR_CLASSIFIER_REINITS})."
        )
        classifier.reset()
        classifier.fit(X[idxs_fit], y[idxs_fit])
        predicted_proba = classifier.predict_proba(X)[:, 1]
    del X

    if _is_collapsed(predicted_proba):
        logger.warning(
            "FDR classifier still collapsed after the retries; target/decoy "
            "separation failed and q-values may not filter PSMs."
        )

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


def keep_best(
    psm_df: pd.DataFrame,
    score_column: str = "proba",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Keep the best PSM for each group of PSMs with the same precursor_idx.

    This function is used to select the best candidate PSM for each precursor.
    if the group_columns is set to ['channel', 'elution_group_idx'] then its used for target decoy competition.

    Parameters
    ----------
    psm_df : pd.DataFrame
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
    psm_df = psm_df.reset_index(drop=True)
    psm_df = psm_df.sort_values(
        [score_column, *group_columns], ascending=True
    )  # last sort to break ties
    psm_df = psm_df.groupby(group_columns).head(1)
    return psm_df.sort_index().reset_index(drop=True)


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
    psm_df: pd.DataFrame,
    score_column: str = "proba",
    decoy_column: str = "_decoy",
    qval_column: str = "qval",
    extra_sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Calculates q-values for a dataframe containing PSMs.

    Parameters
    ----------
    psm_df : pd.DataFrame
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

    # Decoys sort before targets on tied scores. q-values come from a running decoy count,
    # so targets first would let a block of tied scores pass as decoy-free.
    sort_order = [True, False, *[True] * len(extra_sort_columns)]
    psm_df = psm_df.sort_values(
        [score_column, decoy_column, *extra_sort_columns], ascending=sort_order
    )
    target_values = 1 - psm_df[decoy_column].to_numpy()
    decoy_cumsum = np.cumsum(psm_df[decoy_column].to_numpy())
    target_cumsum = np.cumsum(target_values)
    fdr_values = (
        decoy_cumsum / target_cumsum
    )  # TODO: RuntimeWarning: divide by zero encountered in divide
    psm_df[qval_column] = _fdr_to_q_values(fdr_values)
    return psm_df

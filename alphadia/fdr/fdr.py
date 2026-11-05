"""Module performing False Discovery Rate (FDR) control."""

from __future__ import annotations

import copy
import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from alphadia.constants.keys import CalibCols
from alphadia.exceptions import TooFewPSMError
from alphadia.fdr import fragment_features, raw_fragment_features, riding
from alphadia.fdr.plotting import plot_fdr
from alphadia.fdr.utils import (
    MAX_FDR_TORCH_THREADS,
    manage_torch_threads,
    train_test_split_indices,
)
from alphadia.fragcomp.fragcomp import compete_for_fragments

if TYPE_CHECKING:
    from alphadia.fdr.classifiers import Classifier

max_dia_cycle_shape = 2
# Only the final round carries this feature.
FINAL_ROUND_FEATURE = "lib_explained"
# The final-round fits train on each precursor's best candidate only (classifier fit with `groups`).
BEST_PEAK_GROUP_COLUMN = "precursor_idx"
# The final-round fits are cross-fitted: each of CROSSFIT_FOLDS fits learns on the other folds of the competition groups
# and scores its own fold, so no PSM is scored by a fit that trained on it (the riding prescore and the final fit use the
# same folds). Scored in sample, the many final-round features let the fits rank their own training targets above the
# decoys: false targets beat their decoys 1.2x at q 1 % (paired entrapment, 5 ng), 1.0x cross-fitted.
CROSSFIT_FOLDS = 4
# Below CROSSFIT_SMALL_POOL_ROWS PSMs (the MBR step searches a library of pass-1 IDs: ~80k PSMs per 5 ng file) the folds
# train on every out-of-fold row instead of the out-of-fold part of the 80 % split (75 % of the rows instead of 60 %):
# the starved fits cost the MBR step 5 % of its targets at q 1 % against in-sample fits (paired 5 ng 2 and 4 min, same
# MBR library). These pools are also aligned by the decoy ECDF (`_aligned_proba`).
CROSSFIT_SMALL_POOL_ROWS = 500_000
# From CROSSFIT_PARALLEL_MIN_ROWS rows on, the fold fits run side by side in spawned processes, each with an equal share
# of the torch threads, reading the feature matrix from shared memory. One fit scales poorly with threads (1.8 M rows:
# 29 s on 1, 16 s on 3, 10 s on 12), so four processes of 3 threads fit the folds in 27 s instead of 40 s. Threads gain
# nothing at 12 (the training loop holds the GIL), and forked processes deadlock once torch has run.
CROSSFIT_PARALLEL_MIN_ROWS = 100_000
_TORCH_SEED_LIMIT = 2**31 - 1
# The fold fits put their logits on different scales. Each fold is aligned where q-values are cut: the last target at a
# competed q-value of CROSSFIT_ANCHOR_QVAL maps to 0 and the median competition-winning decoy to 1, then one sigmoid
# of slope CROSSFIT_SLOPE.
CROSSFIT_ANCHOR_QVAL = 0.01
CROSSFIT_SLOPE = 5.0
CROSSFIT_MIN_SCALE = 1e-6
CROSSFIT_MAX_ABS_Z = 700.0
# identity columns written next to the features by `dump_path`: enough to compete, stratify and label the rows offline
DUMP_ID_COLUMNS = [
    "precursor_idx",
    "elution_group_idx",
    "channel",
    "decoy",
    "rank",
    "sequence",
    "mods",
    "mod_sites",
    "charge",
    "proteins",
    "genes",
    CalibCols.RT_OBSERVED,
    CalibCols.RT_LIBRARY,
    CalibCols.MZ_LIBRARY,
]
# weight of the anchor-aligned probability that orders the rows within one step of the decoy ECDF
DECOY_ECDF_TIEBREAK = 1e-6

logger = logging.getLogger()

# Below this standard deviation the probability is almost constant. The classifier then
# cannot separate targets from decoys, and the FDR filter keeps all PSMs.
_PROBA_COLLAPSE_STD_THRESHOLD = 1e-4

_MAX_FDR_CLASSIFIER_REINITS = 3

# A collapse can also leave the probability spread out over the junk while every target-like PSM gets one
# and the same value: the last hidden layer is all zero for those inputs, so the output is its bias. The
# standard deviation stays high, but the q-value of that tied block is set by the decoys inside it. A
# healthy float64 probability almost never ties at its best value.
_MAX_TIED_BEST_PROBA = 500

# Upper peptide lengths of the strata whose q-values are computed separately. Short peptides have few fragments, so
# both false and decoy precursors pass more easily than long ones; with one pooled q-value the short stratum ran at a
# decoy-estimated FDR of ~2 % while the long ones stayed below 1 %, and its false targets beat their decoys more often
# still (paired entrapment, 5 ng HeLa: 2.9 % FDP for peptides up to 8 residues at a pooled 1 %).
PEPTIDE_LENGTH_STRATA = (8, 10)


def _is_collapsed(proba: np.ndarray) -> bool:
    """Whether the classifier output carries no ranking: near constant, or one value for the best PSMs."""
    return (
        float(np.std(proba)) < _PROBA_COLLAPSE_STD_THRESHOLD
        or int(np.sum(proba == proba.min())) > _MAX_TIED_BEST_PROBA
    )


def perform_fdr(
    classifier: Classifier,
    available_columns: list[str],
    df_target: pd.DataFrame,
    df_decoy: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Performs FDR calculation on separate dataframes of target and decoy PSMs.

    The PSMs of `df_decoy` are the decoys of the target-decoy competition, whatever their
    `decoy` column says. See `perform_fdr_on_psms` for the keyword arguments and the result.

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

    **kwargs
        Passed to `perform_fdr_on_psms`.

    Returns
    -------
    psm_df : pd.DataFrame
        A dataframe of PSMs with q-values and probabilities.

    """
    is_decoy = np.concatenate(
        [np.zeros(len(df_target), dtype=bool), np.ones(len(df_decoy), dtype=bool)]
    )
    return perform_fdr_on_psms(
        classifier,
        available_columns,
        pd.concat([df_target, df_decoy]),
        is_decoy,
        **kwargs,
    )


@manage_torch_threads(max_threads=MAX_FDR_TORCH_THREADS)
def perform_fdr_on_psms(  # noqa: C901, PLR0912, PLR0913, PLR0915 # too complex, too many branches, too many statements, too many arguments
    classifier: Classifier,
    available_columns: list[str],
    df: pd.DataFrame,
    is_decoy: np.ndarray,
    *,
    competitive: bool = False,
    group_channels: bool = True,
    figure_path: str | None = None,
    df_fragments: pd.DataFrame | None = None,
    fragment_provider: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    riding_provider: Callable[[pd.DataFrame], tuple] | None = None,
    raw_fragment_provider: Callable[[pd.DataFrame], dict[str, np.ndarray]]
    | None = None,
    ms2_tolerance_ppm: float | None = None,
    dia_cycle: np.ndarray | None = None,
    fdr_heuristic: float = 0.1,
    random_state: int | None = None,
    dump_path: str | None = None,
) -> pd.DataFrame:
    """Performs FDR calculation on a dataframe of target and decoy PSMs.

     The torch threads are limited to `MAX_FDR_TORCH_THREADS`, above which the classifier training gets no faster.
     `df` is not modified: the PSMs are copied once, the targets ahead of the decoys.

    Parameters
    ----------
    classifier : Classifier
        A classifier that implements the fit and predict_proba methods

    available_columns : list[str]
        A list of column names that are available for the classifier

    df : pd.DataFrame
        A dataframe of target and decoy PSMs

    is_decoy : np.ndarray, dtype=bool
        Whether each row of `df` is a decoy of the target-decoy competition

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

    riding_provider : Callable[[pd.DataFrame], tuple], default=None
        Returns, in the row order of the PSMs it is given, their top library fragments as (rows, fragments) arrays of
        library m/z, library intensity, observed intensity and correlation (see `riding.find_riding`). In the final round, PSMs whose fragment evidence a
        better co-eluting PSM explains are removed before the classifier is fitted; which PSM is better, and which are
        confident enough to explain others, comes from a fit of the classifier on a sample of the rows.

    raw_fragment_provider : Callable[[pd.DataFrame], dict[str, np.ndarray]], default=None
        Returns, in the row order of the PSMs it is given, the `raw_fragment_features.FEATURES` from their raw fragment
        traces. In the final round (with `riding_provider`) they are classifier features.

    ms2_tolerance_ppm : float, default=None
        The calibrated MS2 tolerance, from which the fragment sharing tolerance of the riding removal is derived.

    dia_cycle : np.ndarray, default=None
        The DIA cycle. Required if df_fragments is provided.

    fdr_heuristic : float, default=0.1
        The FDR heuristic to use for the initial selection of PSMs before fragment competition

    random_state : int, optional
        The random state for train-test split reproducibility.

    dump_path : str, optional
        In the final round, the feature matrix the classifier was fitted on (after the riding removal), the identity
        columns of its rows, their cross-fit fold and their cross-fitted probability are written to this parquet file.

    Returns
    -------
    psm_df : pd.DataFrame
        A dataframe of PSMs with q-values and probabilities.
        The columns `qval` and `proba` are added.

    """
    complete = df[available_columns].notna().all(axis=1).to_numpy()
    target_rows = np.flatnonzero(~is_decoy & complete)
    decoy_rows = np.flatnonzero(is_decoy & complete)
    target_dropped = int((~is_decoy).sum()) - len(target_rows)
    decoy_dropped = int(is_decoy.sum()) - len(decoy_rows)

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

    psm_df = df.take(np.concatenate([target_rows, decoy_rows]))
    X = psm_df[available_columns].to_numpy()
    x_columns = list(available_columns)
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
    initial_classifier = copy.deepcopy(classifier)
    final_round = FINAL_ROUND_FEATURE in available_columns
    remove_riding = (
        final_round
        and riding_provider is not None
        and ms2_tolerance_ppm is not None
        and dia_cycle is not None
        and dia_cycle.shape[2] == 1
    )

    psm_df["_decoy"] = y

    if competitive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]

    groups = psm_df[BEST_PEAK_GROUP_COLUMN].to_numpy() if final_round else None
    crossfit = None
    if final_round:
        competition = psm_df.groupby(group_columns, sort=False).ngroup().to_numpy()
        crossfit = (_crossfit_folds(competition, random_state), competition)
    keep = None
    if remove_riding:
        # quantified once for every PSM: the fragment features need all rows, the removal is served from them
        fragments = riding_provider(psm_df)
        values = fragment_features.compute(
            fragments,
            psm_df["rank"].to_numpy(),
            psm_df[CalibCols.MZ_OBSERVED].to_numpy(np.float64),
        )
        if raw_fragment_provider is not None:
            raw_values = raw_fragment_provider(psm_df)
            values |= {
                name: raw_values[name] for name in raw_fragment_features.FEATURES
            }
        psm_df = psm_df.assign(**values)
        score_columns = [
            c
            for c in available_columns
            if c not in fragment_features.UNUSED_SCORE_COLUMNS
        ]
        x_columns = [
            *score_columns,
            *fragment_features.FEATURES,
            *(name for name in raw_fragment_features.FEATURES if name in values),
        ]
        X = np.column_stack(
            [psm_df[score_columns].to_numpy()]
            + [values[name] for name in x_columns[len(score_columns) :]]
        )
        # the removal needs the m/z, library, observed and correlation arrays only
        riding_provider = fragment_features.serving_provider(
            psm_df["_candidate_idx"].to_numpy(), fragments[: riding.FRAGMENT_ARRAYS]
        )
        # the removal only needs a ranking of the candidates: a fit on a sample of the rows ranks them alike
        keep = _without_riding(
            psm_df,
            _prescore_proba(initial_classifier, X, y, random_state, groups, crossfit),
            group_columns,
            riding_provider,
            dia_cycle,
            ms2_tolerance_ppm,
        )
    if keep is not None:
        psm_df, X, y = psm_df.iloc[keep], X[keep], y[keep]
        groups = groups[keep]
        crossfit = (crossfit[0][keep], crossfit[1][keep])
        idxs_train, idxs_test = train_test_split_indices(
            len(X), test_size=0.2, random_state=random_state
        )
        y_train, y_test = y[idxs_train], y[idxs_test]
    if final_round:
        predicted_proba = _crossfit_predict(
            initial_classifier,
            X,
            y,
            np.arange(len(X)) if len(X) < CROSSFIT_SMALL_POOL_ROWS else idxs_train,
            groups,
            crossfit,
            first=classifier,
        )
        if dump_path is not None:
            _dump_feature_matrix(
                dump_path, psm_df, X, x_columns, crossfit[0], predicted_proba
            )
    else:
        predicted_proba = _fit_predict(classifier, X, y_train, idxs_train, groups)
    del X

    if _is_collapsed(predicted_proba):
        logger.warning(
            "FDR classifier produced a near-constant probability; target/decoy "
            "separation failed and q-values will not filter PSMs."
        )

    psm_df["proba"] = predicted_proba
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
    psm_df = get_q_values_by_length(psm_df, "proba", "_decoy")

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


def _dump_feature_matrix(  # noqa: PLR0913 # one row-aligned array per argument
    path: str,
    psm_df: pd.DataFrame,
    X: np.ndarray,  # noqa: N803
    x_columns: list[str],
    fold: np.ndarray,
    proba: np.ndarray,
) -> None:
    matrix = pd.DataFrame(X, columns=x_columns)
    for column in DUMP_ID_COLUMNS:
        if column in psm_df.columns and column not in matrix.columns:
            matrix[column] = psm_df[column].to_numpy()
    matrix["fold"] = fold
    matrix["proba"] = proba
    matrix.to_parquet(path, index=False)
    logger.info(
        f"Saved the final-round feature matrix ({len(matrix):,} rows, {len(x_columns)} features) to {path}"
    )


def _fit_predict(
    classifier: Classifier,
    X: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    idxs_train: np.ndarray,
    groups: np.ndarray | None = None,
    idxs_predict: np.ndarray | None = None,
) -> np.ndarray:
    """Fit `classifier` on `idxs_train` and predict `idxs_predict` (default all rows); a collapsed fit is repeated from
    new start weights.

    With `groups` (per row of X), each fit trains on the best row of each group among `idxs_train`.
    """
    train_groups = None if groups is None else groups[idxs_train]
    X_predict = X if idxs_predict is None else X[idxs_predict]  # noqa: N806
    classifier.fit(X[idxs_train], y_train, train_groups)
    predicted_proba = classifier.predict_proba(X_predict)[:, 1]

    # A collapse is usually an unlucky set of start weights, so a new fit recovers it.
    n_reinit = 0
    while _is_collapsed(predicted_proba) and n_reinit < _MAX_FDR_CLASSIFIER_REINITS:
        n_reinit += 1
        logger.warning(
            f"FDR classifier collapsed to a near-constant probability "
            f"({np.unique(predicted_proba).size} unique value(s) over "
            f"{len(predicted_proba):,} PSMs); reinitializing from scratch and "
            f"retrying ({n_reinit}/{_MAX_FDR_CLASSIFIER_REINITS})."
        )
        classifier.reset()
        classifier.fit(X[idxs_train], y_train, train_groups)
        predicted_proba = classifier.predict_proba(X_predict)[:, 1]
    return predicted_proba


def _crossfit_folds(competition: np.ndarray, random_state: int | None) -> np.ndarray:
    """The cross-fit fold of every row, drawn per competition group so that a target and its decoy share a fold."""
    _, inverse = np.unique(competition, return_inverse=True)
    rng = np.random.default_rng(random_state)
    return rng.integers(0, CROSSFIT_FOLDS, inverse.max() + 1 if len(inverse) else 0)[
        inverse
    ]


def _crossfit_predict(  # noqa: PLR0913 # the fold assignment comes with the competition groups
    classifier: Classifier,
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    idxs_train: np.ndarray,
    groups: np.ndarray | None,
    crossfit: tuple[np.ndarray, np.ndarray],
    first: Classifier | None = None,
) -> np.ndarray:
    """P(decoy) of every row from CROSSFIT_FOLDS copies of the unfitted `classifier`, each fitted on the rows of
    `idxs_train` outside one fold and predicting that fold, aligned across folds by `_aligned_proba` (by the decoy ECDF
    below CROSSFIT_SMALL_POOL_ROWS rows). From CROSSFIT_PARALLEL_MIN_ROWS rows on, the folds are fitted in parallel
    processes.

    `crossfit` holds the fold and the competition group of every row. `first`, if given, is fitted for the first fold
    instead of a copy, so that it ends up fitted.
    """
    folds, competition = crossfit
    fits = []
    for k in range(CROSSFIT_FOLDS):
        rows = np.flatnonzero(folds == k)
        if len(rows) == 0:
            continue
        member = first if k == 0 and first is not None else copy.deepcopy(classifier)
        fits.append((member, idxs_train[folds[idxs_train] != k], rows))

    if len(X) >= CROSSFIT_PARALLEL_MIN_ROWS and len(fits) > 1:
        fold_probas = _fit_predict_in_processes(fits, X, y, groups)
    else:
        fold_probas = [
            _fit_predict(member, X, y[train], train, groups, rows)
            for member, train, rows in fits
        ]

    decoy_ecdf = len(X) < CROSSFIT_SMALL_POOL_ROWS
    proba = np.empty(len(X))
    for (_, _, rows), fold_proba in zip(fits, fold_probas, strict=True):
        proba[rows] = _aligned_proba(
            fold_proba, y[rows] == 1, competition[rows], decoy_ecdf=decoy_ecdf
        )
    return proba


def _fit_predict_in_processes(
    fits: list[tuple[Classifier, np.ndarray, np.ndarray]],
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    groups: np.ndarray | None,
) -> list[np.ndarray]:
    """`_fit_predict` of every (classifier, training rows, predicted rows) of `fits` in its own spawned process; each
    classifier ends up fitted as if it had been fitted here."""
    n_threads = max(1, torch.get_num_threads() // len(fits))
    # the torch weights and dropout of each process are drawn from a seed of this process's torch generator
    seeds = torch.randint(_TORCH_SEED_LIMIT, (len(fits),)).tolist()
    memory = shared_memory.SharedMemory(create=True, size=max(X.nbytes, 1))
    try:
        np.ndarray(X.shape, X.dtype, buffer=memory.buf)[:] = X
        shared = (memory.name, X.shape, X.dtype)
        with ProcessPoolExecutor(
            len(fits), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    _fit_predict_shared,
                    member,
                    shared,
                    y[train],
                    train,
                    groups,
                    rows,
                    n_threads,
                    seed,
                )
                for (member, train, rows), seed in zip(fits, seeds, strict=True)
            ]
            results = [future.result() for future in futures]
    finally:
        memory.close()
        memory.unlink()
    for (member, _, _), (_, fitted) in zip(fits, results, strict=True):
        member.__dict__.update(fitted.__dict__)
    return [proba for proba, _ in results]


def _fit_predict_shared(  # noqa: PLR0913 # one process's fold: the classifier, its rows and its share of the threads
    classifier: Classifier,
    shared: tuple[str, tuple[int, ...], np.dtype],
    y_train: np.ndarray,
    idxs_train: np.ndarray,
    groups: np.ndarray | None,
    idxs_predict: np.ndarray,
    n_threads: int,
    seed: int,
) -> tuple[np.ndarray, Classifier]:
    """`_fit_predict` in a spawned process on the feature matrix in the shared memory block `shared` (name, shape,
    dtype); returns the probabilities and the fitted classifier."""
    torch.set_num_threads(n_threads)
    torch.manual_seed(seed)
    name, shape, dtype = shared
    memory = shared_memory.SharedMemory(name=name)
    try:
        X = np.ndarray(shape, dtype, buffer=memory.buf)  # noqa: N806
        proba = _fit_predict(classifier, X, y_train, idxs_train, groups, idxs_predict)
        # the shared buffer cannot be closed while an array still views it
        del X
    finally:
        memory.close()
    return proba, classifier


def _aligned_proba(
    proba: np.ndarray,
    decoy: np.ndarray,
    competition: np.ndarray,
    *,
    decoy_ecdf: bool = False,
) -> np.ndarray:
    """P(decoy) of one fold on the common cross-fit scale: the logit of the last target at a competed q-value of
    CROSSFIT_ANCHOR_QVAL maps to 0 and the median logit of the competition-winning decoys to 1. With `decoy_ecdf`, the
    share of the fold's competition-winning decoys that score better instead, ordered within a share by the former."""
    proba = np.clip(proba, 1e-300, 1 - 1e-16)
    logit = np.log(proba) - np.log1p(-proba)
    zero, one = _competition_anchors(logit, decoy, competition)
    z = np.clip(
        (logit - zero) / max(one - zero, CROSSFIT_MIN_SCALE),
        -CROSSFIT_MAX_ABS_Z,
        CROSSFIT_MAX_ABS_Z,
    )
    aligned = 1 / (1 + np.exp(-CROSSFIT_SLOPE * (z - 1)))
    if not decoy_ecdf:
        return aligned
    # The two anchors fail where a fold's fit outputs one logit for thousands of poor rows (6.7k of 7.3k decoys of one
    # MBR fold): the decoy median lands in that block, the fold's scale shrinks ~7x and its best decoys outrank most
    # targets of the other folds. The share of the fold's competition-winning decoys that score better is one scale
    # for every fold whatever the shape of its logits.
    winners = _competition_winners(logit, competition)
    decoy_logits = np.sort(logit[winners][decoy[winners]])
    share = np.searchsorted(decoy_logits, logit, side="right") / max(
        len(decoy_logits), 1
    )
    return (1 - DECOY_ECDF_TIEBREAK) * share + DECOY_ECDF_TIEBREAK * aligned


def _competition_winners(logit: np.ndarray, competition: np.ndarray) -> np.ndarray:
    """Positions of the lowest logit of every competition group."""
    order = np.lexsort((logit, competition))
    first = np.ones(len(order), dtype=bool)
    first[1:] = competition[order][1:] != competition[order][:-1]
    return order[first]


def _competition_anchors(
    logit: np.ndarray, decoy: np.ndarray, competition: np.ndarray
) -> tuple[float, float]:
    """The logit of the last target at a competed q-value <= CROSSFIT_ANCHOR_QVAL and the median logit of the
    competition-winning decoys; the best decoy instead of the target if no target reaches that q-value."""
    winners = _competition_winners(logit, competition)
    w_logit, w_decoy = logit[winners], decoy[winners]
    ranked = np.argsort(w_logit, kind="stable")
    d = np.cumsum(w_decoy[ranked])
    t = np.cumsum(~w_decoy[ranked])
    q = np.minimum.accumulate((d / np.maximum(t, 1))[::-1])[::-1]
    passing = np.flatnonzero((q <= CROSSFIT_ANCHOR_QVAL) & ~w_decoy[ranked])
    if not w_decoy.any():
        return float(w_logit.min()), float(w_logit.max())
    zero = w_logit[ranked][passing[-1]] if len(passing) else w_logit[w_decoy].min()
    return float(zero), float(np.median(w_logit[w_decoy]))


def _prescore_proba(
    initial_classifier: Classifier,
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    random_state: int | None,
    groups: np.ndarray | None = None,
    crossfit: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """P(decoy) of every row from one fit of a copy of `initial_classifier` on at most riding.PRESCORE_ROWS rows,
    cross-fitted with `crossfit` (fold and competition group of every row)."""
    rows = np.arange(len(X))
    if len(X) > riding.PRESCORE_ROWS:
        rng = np.random.default_rng(random_state)
        rows = np.sort(rng.choice(len(X), riding.PRESCORE_ROWS, replace=False))
    if crossfit is not None:
        return _crossfit_predict(initial_classifier, X, y, rows, groups, crossfit)
    return _fit_predict(copy.deepcopy(initial_classifier), X, y[rows], rows, groups)


def _without_riding(  # noqa: PLR0913 # the fragment source, the cycle and the tolerance come from the caller
    psm_df: pd.DataFrame,
    predicted_proba: np.ndarray,
    group_columns: list[str],
    riding_provider: Callable[[pd.DataFrame], tuple],
    dia_cycle: np.ndarray,
    ms2_tolerance_ppm: float,
) -> np.ndarray | None:
    """Positions of `psm_df` to keep after removing the PSMs that ride on a better co-eluting PSM.

    With riding.CLASS_BLIND, targets and decoys are removed by the same rule: a decoy rides on a real peptide's peak as
    often as a false target does, so removing both keeps the decoys a fair model of the false targets. Returns None if
    nothing is removed or the columns needed are missing.
    """
    columns = list(dict.fromkeys([*riding.REQUIRED_COLUMNS, *group_columns]))
    missing = [c for c in columns if c not in psm_df.columns]
    if missing:
        logger.warning(f"riding PSMs not removed: missing columns {missing}")
        return None

    rows = psm_df[columns].reset_index(drop=True)
    rows["proba"] = predicted_proba
    rows["_decoy"] = psm_df["_decoy"].to_numpy()
    rows["_row"] = np.arange(len(rows))

    competed = get_q_values(
        keep_best(rows, group_columns=group_columns), "proba", "_decoy"
    )
    confident = competed[
        (competed["_decoy"] == 0) & (competed["qval"] <= riding.WINNER_QVAL)
    ]
    if len(confident) == 0:
        return None
    partner = np.zeros(len(rows), dtype=bool)
    best = competed["_row"].to_numpy()
    partner[best[competed["proba"].to_numpy() <= confident["proba"].max()]] = True

    loser = np.ones(len(rows), dtype=bool)
    if riding.LOSER_POOLED_QVAL is not None:
        pooled = get_q_values(rows, "proba", "_decoy")
        loser[pooled["_row"].to_numpy()] = (
            pooled["qval"].to_numpy() <= riding.LOSER_POOLED_QVAL
        )
    needed = np.flatnonzero(loser | partner)
    fragments = riding_provider(psm_df.iloc[needed])
    riding_share, own_support = riding.find_riding(
        rows.iloc[needed].reset_index(drop=True),
        fragments,
        partner[needed],
        dia_cycle,
        ms2_tolerance_ppm,
    )
    removed = np.zeros(len(rows), dtype=bool)
    removed[needed] = (
        (riding_share >= riding.MIN_RIDING)
        & (own_support <= riding.MAX_OWN_SUPPORT)
        & loser[needed]
    )
    is_decoy = rows["_decoy"].to_numpy() == 1
    logger.info(
        f"removed {int((removed & ~is_decoy).sum()):,} target and {int((removed & is_decoy).sum()):,} decoy PSMs "
        f"riding on {int(partner.sum()):,} co-eluting partners"
    )
    if not removed.any():
        return None
    return np.flatnonzero(~removed)


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
    # the same selection as a stable sort by score and group followed by groupby(...).head(1), on
    # arrays: the pandas version copied the whole frame for every step
    keys = [df[column].to_numpy() for column in [score_column, *group_columns]]
    order = np.lexsort(keys[::-1])
    group_ids = df.groupby(group_columns, sort=False).ngroup().to_numpy()
    is_first = ~pd.Series(group_ids[order]).duplicated().to_numpy()
    keep = np.sort(order[is_first])
    keep = keep[group_ids[keep] >= 0]
    return df.iloc[keep].reset_index(drop=True)


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


def get_q_values_by_length(
    df: pd.DataFrame,
    score_column: str = "proba",
    decoy_column: str = "_decoy",
    qval_column: str = "qval",
) -> pd.DataFrame:
    """Calculate q-values separately within each peptide length stratum of `PEPTIDE_LENGTH_STRATA`.

    Falls back to pooled q-values when the dataframe has no `naa` column. The result is sorted by `score_column`.
    """
    if "naa" not in df.columns:
        return get_q_values(df, score_column, decoy_column, qval_column)

    strata = np.searchsorted(PEPTIDE_LENGTH_STRATA, df["naa"].to_numpy(), side="left")
    df = pd.concat(
        [
            get_q_values(stratum_df, score_column, decoy_column, qval_column)
            for _, stratum_df in df.groupby(strata)
        ]
    )
    return df.sort_values([score_column, decoy_column, "precursor_idx"], ascending=True)


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
    target_values = 1 - df[decoy_column].to_numpy()
    decoy_cumsum = np.cumsum(df[decoy_column].to_numpy())
    target_cumsum = np.cumsum(target_values)
    fdr_values = np.divide(
        decoy_cumsum + decoy_offset,
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

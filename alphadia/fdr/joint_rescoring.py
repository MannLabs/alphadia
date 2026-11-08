"""First-pass precursors rescored jointly over the runs of an experiment.

The per-run final-round classifier is data-limited at low input: half of a run's competition groups cost 7 % of its
IDs. Here every run is also scored by a classifier fitted on all final-round rows of the other runs (leave one run out;
above MAX_LOFO_RUNS runs, the runs fall into MAX_LOFO_RUNS groups and a group is left out), with the per-run settings,
quantile transform and best-peak training. Its probability is aligned like a cross-fit fold and averaged with the
run's own cross-fitted probability on the logit scale. A run's own rows, its decoys included, never enter its joint
model, so a peptide is never scored by a fit that saw it in this run. Offline (5 ng HeLa, 5 runs): +4.2 % targets at
q 1 %, +3.8 % at matched entrapment, with entrapment-estimated calibration as good as the per-run fit.

The rows that lost the run's own fragment competition stay out. Then the best row of every competition group, and the
q-values within run and peptide length stratum. A run keeps its precursors up to KEEP_QVAL in the first pass, so that a
precursor promoted here is already quantified; one promoted from beyond KEEP_QVAL is counted but not reported.
"""

import logging
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from alphadia.fdr import fdr, joint_features, mbr_transfer
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching

logger = logging.getLogger()

# per-run precursors kept and quantified in the first pass when the joint rescoring follows
KEEP_QVAL = 0.05
MIN_RUNS = 2
MAX_LOFO_RUNS = 8
# above this many rows of the other runs, whole competition groups are sampled. The fit peaks at ~3x the float32
# training matrix (the matrix, its quantile-transformed copy and the transform's per-thread temporaries): 8 M rows of
# 48 features peaked at 6.4 GB above the identity tables, so 25 M rows of ~60 features need ~20 GB.
MAX_TRAINING_ROWS = 25_000_000
# the best-peak group of a joint fit is the precursor within its run
RUN_GROUP_STRIDE = 1 << 32
# the clip of `fdr._aligned_proba`
PROBA_CLIP_LOW = 1e-300
PROBA_CLIP_HIGH = 1 - 1e-16


def keep_qval(fdr_cutoff: float) -> float:
    return max(fdr_cutoff, KEEP_QVAL)


def run_groups(n_runs: int) -> np.ndarray:
    """The left-out group of every run: each run alone up to MAX_LOFO_RUNS runs, else MAX_LOFO_RUNS groups."""
    return np.arange(n_runs) % MAX_LOFO_RUNS


def competition_groups(ids: pd.DataFrame, group_columns: list[str]) -> np.ndarray:
    return ids.groupby(group_columns, sort=False).ngroup().to_numpy()


def training_rows(
    competitions: list[np.ndarray],
    train_runs: list[int],
    rng: np.random.Generator,
    max_rows: int = MAX_TRAINING_ROWS,
) -> dict[int, np.ndarray]:
    """The rows of every training run; above `max_rows` in total, the rows of a random share of each run's
    competition groups, so that a target and its decoy are drawn together."""
    total = sum(len(competitions[i]) for i in train_runs)
    if total <= max_rows:
        return {i: np.arange(len(competitions[i])) for i in train_runs}
    share = max_rows / total
    rows = {}
    for i in train_runs:
        competition = competitions[i]
        drawn = rng.random(competition.max() + 1 if len(competition) else 0) < share
        rows[i] = np.flatnonzero(drawn[competition])
    return rows


def ensemble_proba(run_proba: np.ndarray, joint_proba: np.ndarray) -> np.ndarray:
    """The sigmoid of the mean logit of the two probabilities."""
    logits = [
        np.log(p) - np.log1p(-p)
        for p in (
            np.clip(run_proba, PROBA_CLIP_LOW, PROBA_CLIP_HIGH),
            np.clip(joint_proba, PROBA_CLIP_LOW, PROBA_CLIP_HIGH),
        )
    ]
    return expit((logits[0] + logits[1]) / 2)


def _new_classifier(
    config_fdr: dict, random_state: int | None
) -> BinaryClassifierLegacyNewBatching:
    classifier = BinaryClassifierLegacyNewBatching(
        **mbr_transfer.CLASSIFIER_PARAMETERS,
        experimental_hyperparameter_tuning=config_fdr[
            "enable_nn_hyperparameter_tuning"
        ],
        random_state=random_state,
    )
    classifier.feature_transform = config_fdr["feature_transform"]
    return classifier


def _fit_predict_left_out(  # noqa: PLR0913 # the runs, their tables and the classifier settings
    folders: list[str],
    ids: list[pd.DataFrame],
    rows: dict[int, np.ndarray],
    left_out: list[int],
    columns: list[str],
    config_fdr: dict,
    random_state: int | None,
) -> dict[int, np.ndarray]:
    """P(decoy) of every row of the `left_out` runs from one classifier fitted on `rows` of the other runs; a fit that
    collapses on a left-out run is repeated from new start weights."""
    x = np.empty((sum(len(r) for r in rows.values()), len(columns)), np.float32)
    offset = 0
    for i, r in rows.items():
        x[offset : offset + len(r)] = joint_features.read_features(
            folders[i], columns, r
        )
        offset += len(r)
    y = np.concatenate([ids[i]["decoy"].to_numpy()[r] for i, r in rows.items()]).astype(
        float
    )
    groups = np.concatenate(
        [
            i * RUN_GROUP_STRIDE + ids[i]["precursor_idx"].to_numpy(np.int64)[r]
            for i, r in rows.items()
        ]
    )
    classifier = _new_classifier(config_fdr, random_state)
    for attempt in range(fdr._MAX_FDR_CLASSIFIER_REINITS + 1):
        if attempt > 0:
            logger.warning(
                f"Joint rescoring: the classifier collapsed, refitting ({attempt}/{fdr._MAX_FDR_CLASSIFIER_REINITS})"
            )
            classifier.reset()
        classifier.fit(x, y, groups)
        probas = {
            i: classifier.predict_proba(
                joint_features.read_features(folders[i], columns)
            )[:, 1]
            for i in left_out
        }
        if not any(fdr._is_collapsed(p) for p in probas.values()):
            break
    return probas


def _rescored_run(
    ids: pd.DataFrame,
    competition: np.ndarray,
    joint_proba: np.ndarray,
    group_columns: list[str],
) -> pd.DataFrame:
    """The best row of every competition group among the rows that kept their fragments, ranked by the ensemble of
    the run's probability and the aligned `joint_proba`, with q-values within peptide length strata."""
    decoy = ids["decoy"].to_numpy() == 1
    aligned = fdr._aligned_proba(joint_proba, decoy, competition, decoy_ecdf=False)
    rows = ids.assign(proba=ensemble_proba(ids[joint_features.PROBA_COLUMN], aligned))
    rows = rows[~rows["lost"].to_numpy()]
    best = fdr.keep_best(rows, group_columns=group_columns)
    return fdr.get_q_values_by_length(best, "proba", "decoy")


def _report(
    run_psm_df: pd.DataFrame, best: pd.DataFrame, fdr_cutoff: float
) -> pd.DataFrame:
    """The PSMs of the run whose precursor won its competition group at a joint q-value <= `fdr_cutoff`, with the
    joint probability and q-value."""
    accepted = best[best["qval"] <= fdr_cutoff]
    reported = run_psm_df.drop(columns=["proba", "qval"]).merge(
        accepted[["precursor_idx", "channel", "rank", "proba", "qval"]].rename(
            columns={"rank": "_joint_rank"}
        ),
        on=["precursor_idx", "channel"],
        how="inner",
    )
    is_target = accepted["decoy"].to_numpy() == 0
    reported_target = reported["decoy"].to_numpy() == 0
    before = int(
        ((run_psm_df["qval"] <= fdr_cutoff) & (run_psm_df["decoy"] == 0)).sum()
    )
    logger.info(
        f"Joint rescoring {run_psm_df['run'].iloc[0] if len(run_psm_df) else ''}: "
        f"{int(reported_target.sum()):,} target precursors at q <= {fdr_cutoff} (per run: {before:,}); "
        f"not quantified in the run: {int(is_target.sum() - reported_target.sum()):,} targets, "
        f"{int((~is_target).sum() - (~reported_target).sum()):,} decoys; "
        f"best candidate differs from the run's: {int((reported['rank'] != reported['_joint_rank']).sum()):,}"
    )
    return reported.drop(columns="_joint_rank")


def rescore(
    psm_df: pd.DataFrame,
    folder_list: list[str],
    config_fdr: dict,
    group_columns: list[str],
    random_state: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """`psm_df` (the PSMs of all runs) with the PSMs of the runs that wrote joint features rescored and filtered at the
    configured FDR; the PSMs of other runs are kept as they are. Returns the rescored runs' folders as well.

    With fewer than MIN_RUNS such runs, their PSMs are filtered at their per-run q-value instead.
    """
    folders = [f for f in folder_list if joint_features.has_run_features(f)]
    if not folders:
        return psm_df, []
    runs = [os.path.basename(os.path.normpath(f)) for f in folders]
    fdr_cutoff = config_fdr["fdr"]
    rescored = psm_df["run"].isin(runs).to_numpy()
    others = psm_df[~rescored]
    if len(folders) < MIN_RUNS:
        logger.info(
            f"Joint rescoring needs {MIN_RUNS} runs, keeping the per-run q-values"
        )
        kept = psm_df[rescored & (psm_df["qval"] <= fdr_cutoff).to_numpy()]
        return pd.concat([others, kept], ignore_index=True), folders

    ids = [joint_features.read_ids(f) for f in folders]
    competitions = [competition_groups(i, group_columns) for i in ids]
    # the same features in every run unless a run lacked a feature source (e.g. MS1)
    shared = set.intersection(
        *(set(joint_features.feature_columns(f)) for f in folders)
    )
    columns = [c for c in joint_features.feature_columns(folders[0]) if c in shared]
    rng = np.random.default_rng(random_state)
    groups = run_groups(len(folders))
    reported = []
    for group in np.unique(groups):
        start = time.time()
        left_out = [int(i) for i in np.flatnonzero(groups == group)]
        train_runs = [int(i) for i in np.flatnonzero(groups != group)]
        rows = training_rows(competitions, train_runs, rng, MAX_TRAINING_ROWS)
        probas = _fit_predict_left_out(
            folders,
            ids,
            rows,
            left_out,
            columns,
            config_fdr,
            None if random_state is None else random_state + int(group),
        )
        for i in left_out:
            best = _rescored_run(ids[i], competitions[i], probas[i], group_columns)
            reported.append(_report(psm_df[psm_df["run"] == runs[i]], best, fdr_cutoff))
        logger.info(
            f"Joint rescoring: model {int(group) + 1} of {len(np.unique(groups))} on "
            f"{sum(len(r) for r in rows.values()):,} rows of {len(train_runs)} runs, {len(columns)} features, "
            f"{time.time() - start:.0f} s"
        )
    return pd.concat([others, *reported], ignore_index=True), folders

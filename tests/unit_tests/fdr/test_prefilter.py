import logging

import numpy as np
import pandas as pd

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import LightGBMClassifier
from alphadia.fdr.cross_fitting import CrossFittedTrainer
from alphadia.fdr.prefilter import CascadePrefilter


def _gen_target_decoy_dfs(n_samples: int = 400, seed: int = 0):
    """Targets with a bimodal separable feature: half of them look like decoys."""
    rng = np.random.default_rng(seed)
    target_feature = np.concatenate(
        [rng.normal(0.0, 0.3, n_samples // 2), rng.normal(2.0, 0.3, n_samples // 2)]
    )
    decoy_feature = rng.normal(2.0, 0.3, n_samples)
    target_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_samples),
            "elution_group_idx": np.arange(n_samples),
            "channel": 0,
            "decoy": 0,
            "feature": target_feature,
            "noise": rng.normal(size=n_samples),
        }
    )
    decoy_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_samples, 2 * n_samples),
            "elution_group_idx": np.arange(n_samples),
            "channel": 0,
            "decoy": 1,
            "feature": decoy_feature,
            "noise": rng.normal(size=n_samples),
        }
    )
    return target_df, decoy_df


def _get_classifier() -> LightGBMClassifier:
    return LightGBMClassifier(
        n_estimators=20,
        final_n_estimators=20,
        min_child_samples=5,
        num_threads=1,
        random_state=0,
    )


def _get_prefilter(
    q_value_threshold: float, min_psms: int = 0, min_kept_psms: int = 1
) -> CascadePrefilter:
    return CascadePrefilter(
        feature_columns=["feature"],
        classifier=_get_classifier(),
        q_value_threshold=q_value_threshold,
        n_folds=2,
        min_psms=min_psms,
        min_kept_psms=min_kept_psms,
        random_state=0,
    )


def test_prefilter_keeps_the_confident_targets_and_drops_decoy_like_psms():
    # Given: targets of which half are indistinguishable from decoys
    target_df, decoy_df = _gen_target_decoy_dfs()
    psm_df = pd.concat([target_df, decoy_df]).reset_index(drop=True)
    y = psm_df["decoy"].to_numpy()

    # When: the prefilter gates the PSMs
    keep, stage1_proba, n_passed = _get_prefilter(q_value_threshold=0.2).select(
        psm_df, y, is_final=True
    )

    # Then: the good targets pass and bring their decoys along, the elution groups of the
    # decoy-like targets are mostly dropped, and every PSM has a stage-1 score
    good_targets = (psm_df["feature"] < 1.0).to_numpy()
    good_groups = psm_df["elution_group_idx"].isin(
        psm_df.loc[good_targets, "elution_group_idx"]
    )
    assert keep[good_targets].mean() > 0.95
    assert keep[good_groups & (y == 1)].mean() > 0.95
    assert keep[~good_groups].mean() < 0.3
    assert n_passed < keep.sum() < len(psm_df)
    assert stage1_proba.shape == (len(psm_df),)
    assert stage1_proba[good_targets].mean() < stage1_proba[y == 1].mean()


def test_prefilter_extends_the_cut_to_the_floor_when_the_threshold_keeps_too_few(
    caplog,
):
    # Given: a q-value threshold no PSM reaches, and a floor of 300 PSMs
    caplog.set_level(logging.INFO)
    target_df, decoy_df = _gen_target_decoy_dfs()
    psm_df = pd.concat([target_df, decoy_df]).reset_index(drop=True)
    y = psm_df["decoy"].to_numpy()

    # When: the prefilter gates the PSMs
    keep, stage1_proba, n_passed = _get_prefilter(
        q_value_threshold=0.0, min_kept_psms=300
    ).select(psm_df, y, is_final=True)

    # Then: exactly the 300 best-ranked PSMs pass the cut and are kept with their groups
    stage1_order = np.lexsort((psm_df["precursor_idx"].to_numpy(), y, stage1_proba))
    assert n_passed == 300
    assert keep[stage1_order[:300]].all()
    assert 300 <= keep.sum() <= 600
    assert "raised from" in caplog.text


def test_prefilter_passes_everything_below_min_psms():
    # Given: fewer PSMs than the prefilter minimum
    target_df, decoy_df = _gen_target_decoy_dfs()
    psm_df = pd.concat([target_df, decoy_df]).reset_index(drop=True)
    y = psm_df["decoy"].to_numpy()

    # When: the prefilter gates the PSMs
    keep, stage1_proba, n_passed = _get_prefilter(
        q_value_threshold=0.2, min_psms=len(psm_df) + 1
    ).select(psm_df, y, is_final=True)

    # Then: nothing is dropped and no stage-1 model was fitted
    assert keep.all()
    assert n_passed == len(psm_df)
    assert not stage1_proba.any()


def test_prefilter_fits_each_fold_on_at_most_max_train_psms_rows():
    # Given: a cap below the number of rows outside the fold
    prefilter = _get_prefilter(q_value_threshold=0.2)
    prefilter.max_train_psms = 100
    candidates = np.arange(400)

    # When: the training rows of one fold are drawn
    train_rows = prefilter._training_rows(candidates)

    # Then: exactly the cap is drawn, without replacement, from the candidates
    assert len(train_rows) == 100
    assert len(np.unique(train_rows)) == 100
    assert np.isin(train_rows, candidates).all()


def test_prefilter_fits_each_fold_on_every_row_when_uncapped():
    prefilter = _get_prefilter(q_value_threshold=0.2)
    candidates = np.arange(400)

    assert (prefilter._training_rows(candidates) == candidates).all()


class _SaturatingClassifier(LightGBMClassifier):
    """Scores every PSM it is not sure about at a probability of exactly 1.0, the way a
    network saturates on the false matches."""

    def predict_proba(self, x):
        proba = super().predict_proba(x)
        proba[:, 1] = np.where(proba[:, 1] > 0.5, 1.0, proba[:, 1])  # noqa: PLR2004
        proba[:, 0] = 1 - proba[:, 1]
        return proba


def test_perform_fdr_with_prefilter_never_identifies_a_dropped_psm():
    # Given: separable targets and decoys, a prefilter, and a classifier that saturates
    # the worst scored PSMs at 1.0, so a dropped PSM ranked behind them by a shifted
    # score would tie with all the others
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples=2000)
    classifier = _SaturatingClassifier(
        n_estimators=20,
        final_n_estimators=20,
        min_child_samples=5,
        num_threads=1,
        random_state=0,
    )

    # When: perform_fdr runs with the prefilter in the cross-fitted final round
    psm_df = fdr.perform_fdr(
        classifier,
        ["feature", "noise"],
        target_df,
        decoy_df,
        competitive=True,
        random_state=0,
        is_final=True,
        prefilter=_get_prefilter(q_value_threshold=0.2),
        trainer=CrossFittedTrainer(n_folds=2, random_state=0),
    )

    # Then: one PSM per elution group comes back, the good targets are identified, and
    # nothing scored at 1.0, dropped or saturated, is
    assert len(psm_df) == 2000
    good_targets = psm_df[(psm_df["_decoy"] == 0) & (psm_df["feature"] < 1.0)]
    assert (good_targets["qval"] < 0.05).mean() > 0.9
    saturated = psm_df[psm_df["proba"] == 1.0]
    assert len(saturated) > 900
    assert (saturated["qval"] > 0.05).all()


def test_perform_fdr_with_prefilter_ranks_dropped_psms_behind_scored_ones():
    # Given: separable targets and decoys and a prefilter
    target_df, decoy_df = _gen_target_decoy_dfs()

    # When: perform_fdr runs with the prefilter
    psm_df = fdr.perform_fdr(
        _get_classifier(),
        ["feature", "noise"],
        target_df.copy(),
        decoy_df.copy(),
        competitive=True,
        random_state=0,
        is_final=True,
        prefilter=_get_prefilter(q_value_threshold=0.2),
    )

    # Then: every PSM gets a probability, the good targets get the low q-values and the
    # dropped PSMs all carry a probability and q-value of one
    assert psm_df["proba"].notna().all()
    good_targets = psm_df[(psm_df["_decoy"] == 0) & (psm_df["feature"] < 1.0)]
    assert (good_targets["qval"] < 0.05).mean() > 0.9
    dropped = psm_df[psm_df["proba"] == 1.0]
    assert len(dropped) > 0
    assert (dropped["qval"] == 1.0).all()


def test_perform_fdr_with_prefilter_reports_the_recall_check(caplog):
    # Given: separable targets and decoys, a gate with room behind the confident ones and
    # the cross-fitted final round, whose false identifications do not pile up on the
    # scored rows the way a plain fit's do
    caplog.set_level(logging.INFO)
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples=2000)

    # When: the FDR is computed behind the prefilter
    psm_df = fdr.perform_fdr(
        _get_classifier(),
        ["feature", "noise"],
        target_df,
        decoy_df,
        competitive=True,
        is_final=True,
        prefilter=_get_prefilter(q_value_threshold=0.5),
        trainer=CrossFittedTrainer(n_folds=2, random_state=0),
    )

    # Then: the identifications are checked against the gate's cut, the cut holds and
    # the helper column does not leak into the result
    assert "Prefilter recall check" in caplog.text
    assert "widening the cut" not in caplog.text
    assert "_stage1_rank" not in psm_df.columns


def test_perform_fdr_with_prefilter_widens_the_cut_when_the_identifications_crowd_it(
    caplog,
):
    # Given: a gate whose floor cuts right through the confident targets
    caplog.set_level(logging.INFO)
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples=2000)

    # When: the FDR is computed in the final round behind that gate
    psm_df = fdr.perform_fdr(
        _get_classifier(),
        ["feature", "noise"],
        target_df,
        decoy_df,
        competitive=True,
        is_final=True,
        prefilter=_get_prefilter(q_value_threshold=0.0, min_kept_psms=500),
    )

    # Then: the cut is widened once and the confident targets are identified after all
    assert "widening the cut to stage-1 q-value <= 0.5" in caplog.text
    assert caplog.text.count("Prefilter passed") == 2
    good_targets = psm_df[(psm_df["_decoy"] == 0) & (psm_df["feature"] < 1.0)]
    assert (good_targets["qval"] < 0.05).mean() > 0.9


def test_perform_fdr_with_prefilter_leaves_the_cut_alone_in_optimization_rounds(
    caplog,
):
    # Given: the same truncating gate, in a round whose scores only steer the calibration
    caplog.set_level(logging.INFO)
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples=2000)

    # When: the FDR is computed in an optimization round
    fdr.perform_fdr(
        _get_classifier(),
        ["feature", "noise"],
        target_df,
        decoy_df,
        competitive=True,
        is_final=False,
        prefilter=_get_prefilter(q_value_threshold=0.0, min_kept_psms=500),
    )

    # Then: neither the check nor the widening runs
    assert "Prefilter recall check" not in caplog.text
    assert caplog.text.count("Prefilter passed") == 1

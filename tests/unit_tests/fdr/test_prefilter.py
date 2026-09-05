import numpy as np
import pandas as pd

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import LightGBMClassifier
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


def _get_prefilter(q_value_threshold: float, min_psms: int = 0) -> CascadePrefilter:
    return CascadePrefilter(
        feature_columns=["feature"],
        classifier=LightGBMClassifier(
            n_estimators=20, min_child_samples=5, num_threads=1, random_state=0
        ),
        q_value_threshold=q_value_threshold,
        n_folds=2,
        min_psms=min_psms,
        random_state=0,
    )


def test_prefilter_keeps_the_confident_targets_and_drops_decoy_like_psms():
    # Given: targets of which half are indistinguishable from decoys
    target_df, decoy_df = _gen_target_decoy_dfs()
    psm_df = pd.concat([target_df, decoy_df]).reset_index(drop=True)
    y = psm_df["decoy"].to_numpy()

    # When: the prefilter gates the PSMs
    keep, stage1_proba = _get_prefilter(q_value_threshold=0.2).select(psm_df, y)

    # Then: the good targets pass, most decoys do not, and every PSM has a stage-1 score
    good_targets = (psm_df["feature"] < 1.0).to_numpy()
    assert keep[good_targets].mean() > 0.95
    assert keep[y == 1].mean() < 0.3
    assert 0.0 < keep.mean() < 1.0
    assert stage1_proba.shape == (len(psm_df),)
    assert stage1_proba[good_targets].mean() < stage1_proba[y == 1].mean()


def test_prefilter_passes_everything_below_min_psms():
    # Given: fewer PSMs than the prefilter minimum
    target_df, decoy_df = _gen_target_decoy_dfs()
    psm_df = pd.concat([target_df, decoy_df]).reset_index(drop=True)
    y = psm_df["decoy"].to_numpy()

    # When: the prefilter gates the PSMs
    keep, stage1_proba = _get_prefilter(
        q_value_threshold=0.2, min_psms=len(psm_df) + 1
    ).select(psm_df, y)

    # Then: nothing is dropped and no stage-1 model was fitted
    assert keep.all()
    assert not stage1_proba.any()


def test_perform_fdr_with_prefilter_ranks_dropped_psms_behind_scored_ones():
    # Given: separable targets and decoys and a prefilter
    target_df, decoy_df = _gen_target_decoy_dfs()
    prefilter = _get_prefilter(q_value_threshold=0.2)
    classifier = LightGBMClassifier(
        n_estimators=20, min_child_samples=5, num_threads=1, random_state=0
    )

    # When: perform_fdr runs with the prefilter
    psm_df = fdr.perform_fdr(
        classifier,
        ["feature", "noise"],
        target_df.copy(),
        decoy_df.copy(),
        competitive=True,
        random_state=0,
        prefilter=prefilter,
    )

    # Then: every PSM gets a probability, the good targets get the low q-values and the
    # dropped PSMs all rank behind the scored ones
    assert psm_df["proba"].notna().all()
    good_targets = psm_df[(psm_df["_decoy"] == 0) & (psm_df["feature"] < 1.0)]
    assert (good_targets["qval"] < 0.05).mean() > 0.9
    decoys = psm_df[psm_df["_decoy"] == 1]
    assert good_targets["proba"].max() < decoys["proba"].quantile(0.5)


def test_perform_fdr_with_prefilter_matches_the_unfiltered_identifications():
    # Given: the same PSMs scored with and without the prefilter
    target_df, decoy_df = _gen_target_decoy_dfs()

    def _ids(prefilter):
        psm_df = fdr.perform_fdr(
            LightGBMClassifier(
                n_estimators=20, min_child_samples=5, num_threads=1, random_state=0
            ),
            ["feature", "noise"],
            target_df.copy(),
            decoy_df.copy(),
            competitive=True,
            random_state=0,
            prefilter=prefilter,
        )
        return int(((psm_df["_decoy"] == 0) & (psm_df["qval"] <= 0.05)).sum())

    # When: identifications are counted at 5% FDR
    ids_plain = _ids(None)
    ids_prefiltered = _ids(_get_prefilter(q_value_threshold=0.5))

    # Then: the prefilter does not cost identifications on separable data
    assert ids_prefiltered >= 0.9 * ids_plain

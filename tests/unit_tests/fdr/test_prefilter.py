import numpy as np
import pandas as pd

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
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


def _get_small_network() -> BinaryClassifierLegacyNewBatching:
    return BinaryClassifierLegacyNewBatching(
        test_size=0.1,
        batch_size=50,
        learning_rate=0.01,
        epochs=20,
        layers=[8],
        random_state=0,
    )


def _get_prefilter(
    q_value_threshold: float,
    min_psms: int = 0,
    far_psms: int = 0,
) -> CascadePrefilter:
    return CascadePrefilter(
        feature_columns=["feature"],
        classifier=_get_small_network(),
        q_value_threshold=q_value_threshold,
        n_folds=2,
        min_psms=min_psms,
        far_psms=far_psms,
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


def test_prefilter_adds_random_dropped_psms_to_the_kept_set():
    # Given: PSMs gated once without and once with far PSMs, on the same folds
    target_df, decoy_df = _gen_target_decoy_dfs()
    psm_df = pd.concat([target_df, decoy_df]).reset_index(drop=True)
    y = psm_df["decoy"].to_numpy()
    far_psms = 50
    gate_keep, _ = _get_prefilter(q_value_threshold=0.2).select(psm_df, y)

    # When: the prefilter gates the PSMs with far PSMs
    keep, _ = _get_prefilter(q_value_threshold=0.2, far_psms=far_psms).select(psm_df, y)

    # Then: the gated PSMs are still kept and exactly far_psms dropped ones were added
    assert keep[gate_keep].all()
    assert keep.sum() == gate_keep.sum() + far_psms


def test_perform_fdr_with_prefilter_ranks_dropped_psms_behind_scored_ones():
    # Given: separable targets and decoys and a prefilter
    target_df, decoy_df = _gen_target_decoy_dfs()
    prefilter = _get_prefilter(q_value_threshold=0.2)

    # When: perform_fdr runs with the prefilter
    psm_df = fdr.perform_fdr(
        _get_small_network(),
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


class _MemorizingClassifier:
    """Returns the label of every row it was fitted on and 0.5 for rows it has not seen."""

    def __init__(self):
        self._seen: dict[bytes, float] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._seen = {row.tobytes(): label for row, label in zip(x, y, strict=True)}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        decoy_proba = np.array([self._seen.get(row.tobytes(), 0.5) for row in x])
        return np.stack([1 - decoy_proba, decoy_proba], axis=1)

    def reset(self) -> None:
        self._seen = {}

    def to_state_dict(self) -> dict:
        return {"seen": self._seen}

    def from_state_dict(self, state_dict: dict) -> None:
        self._seen = state_dict["seen"]


def test_perform_fdr_cross_fit_scores_every_kept_psm_out_of_fold():
    # Given: a classifier that memorizes the labels of its training rows, behind a
    # prefilter that keeps every PSM
    target_df, decoy_df = _gen_target_decoy_dfs()
    prefilter = _get_prefilter(q_value_threshold=1.0)

    # When: perform_fdr cross-fits the classifier
    psm_df = fdr.perform_fdr(
        _MemorizingClassifier(),
        ["feature", "noise"],
        target_df.copy(),
        decoy_df.copy(),
        competitive=False,
        random_state=0,
        prefilter=prefilter,
        cross_fit=True,
    )

    # Then: no PSM is scored by a model that saw its label, so the memorized labels
    # never reach the probabilities
    assert (psm_df["proba"] == 0.5).all()


def test_perform_fdr_cross_fit_hands_back_a_fitted_classifier():
    # Given: separable targets and decoys and an unfitted network
    target_df, decoy_df = _gen_target_decoy_dfs()
    classifier = _get_small_network()

    # When: perform_fdr cross-fits it
    psm_df = fdr.perform_fdr(
        classifier,
        ["feature", "noise"],
        target_df.copy(),
        decoy_df.copy(),
        competitive=True,
        random_state=0,
        cross_fit=True,
    )

    # Then: the good targets are found and the passed classifier holds a fitted model
    good_targets = psm_df[(psm_df["_decoy"] == 0) & (psm_df["feature"] < 1.0)]
    assert (good_targets["qval"] < 0.05).mean() > 0.9
    assert classifier.fitted

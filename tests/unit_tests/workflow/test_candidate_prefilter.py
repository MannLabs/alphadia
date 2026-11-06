import numpy as np
import pandas as pd

from alphadia.workflow.peptidecentric.candidate_prefilter import (
    SELECTION_FEATURES,
    fit_prefilter,
)


def _candidates(rng: np.random.Generator) -> pd.DataFrame:
    """Targets and decoys whose identifications are told apart by fragment detection, not by score."""
    n = 2000
    decoy = np.r_[np.zeros(n), np.ones(n)].astype(int)
    detected = np.where(
        decoy == 0, rng.uniform(0.5, 1.0, 2 * n), rng.uniform(0.0, 0.6, 2 * n)
    )
    return pd.DataFrame(
        {
            "precursor_idx": np.arange(2 * n),
            "rank": 0,
            "decoy": decoy,
            "score": rng.normal(10.0, 1.0, 2 * n),
            "selection_det_share": detected,
            "selection_n_detected": np.round(detected * 12),
        }
    )


def test_prefilter_keeps_the_requested_share_of_identifications():
    # given
    candidates = _candidates(np.random.default_rng(0))
    identifications = candidates[candidates["decoy"] == 0].iloc[:1500]

    # when
    prefilter = fit_prefilter(candidates, identifications, kept_share=0.9)

    # then
    kept = prefilter.keep(identifications)
    assert abs(kept.mean() - 0.9) < 0.01


def test_prefilter_ranks_by_fragment_detection_when_the_score_carries_no_information():
    # given
    candidates = _candidates(np.random.default_rng(1))
    identifications = candidates[candidates["decoy"] == 0]

    # when
    prefilter = fit_prefilter(candidates, identifications, kept_share=0.95)

    # then
    kept = prefilter.keep(candidates)
    assert kept[candidates["decoy"] == 1].mean() < 0.3
    assert abs(prefilter.coef[0]) < abs(prefilter.coef[1])


def test_prefilter_is_none_without_selection_features_or_rows():
    # given
    candidates = _candidates(np.random.default_rng(2))
    identifications = candidates[candidates["decoy"] == 0]

    # when / then
    assert (
        fit_prefilter(
            candidates.drop(columns=SELECTION_FEATURES[1:]), identifications, 0.9
        )
        is None
    )
    assert fit_prefilter(candidates, identifications.iloc[:10], 0.9) is None

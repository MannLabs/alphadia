"""Unit tests for the NG mapper helpers around the context and claim features."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from conftest import mock_claim_features, mock_context_features

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_claim_feature_names,
    get_context_feature_names,
    get_feature_names,
    merge_claim_features,
    merge_context_features,
    to_features_df,
)


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"precursor_idx": [1, 1, 2], "rank": [0, 1, 0], "score": [1.0, 0.5, 2.0]}
    )


def test_merge_context_features_aligns_on_candidate():
    # given: the context rows come in a different order than the features
    features_df = _features_df()
    context_features = mock_context_features([2, 1, 1], [0, 1, 0])

    # when
    merged_df = merge_context_features(features_df, context_features)

    # then
    assert merged_df["precursor_idx"].tolist() == [1, 1, 2]
    assert merged_df["rank"].tolist() == [0, 1, 0]
    for name in get_context_feature_names():
        assert merged_df[name].tolist() == [2.0, 1.0, 0.0]


def test_merge_context_features_raises_on_missing_candidate():
    # given: candidate (2, 0) has no context features
    features_df = _features_df()
    context_features = mock_context_features([1, 1], [0, 1])

    # when / then
    with pytest.raises(ValueError, match="missing for 1 candidates"):
        merge_context_features(features_df, context_features)


def test_merge_context_features_raises_on_duplicate_candidate():
    # given: candidate (1, 0) appears twice
    features_df = _features_df()
    context_features = mock_context_features([1, 1, 2, 1], [0, 1, 0, 0])

    # when / then
    with pytest.raises(ValueError, match="duplicate candidates") as error:
        merge_context_features(features_df, context_features)
    assert "1" in str(error.value)


def test_merge_claim_features_aligns_on_candidate():
    # given: the claim rows come in a different order than the features
    features_df = _features_df()
    claim_features = mock_claim_features([2, 1, 1], [0, 1, 0], soft_claims=True)

    # when
    merged_df = merge_claim_features(features_df, claim_features)

    # then
    assert merged_df["precursor_idx"].tolist() == [1, 1, 2]
    assert merged_df["rank"].tolist() == [0, 1, 0]
    for name in get_claim_feature_names():
        assert merged_df[name].tolist() == [2.0, 1.0, 0.0]


def test_merge_claim_features_raises_on_missing_candidate():
    # given: candidate (2, 0) has no claim features
    features_df = _features_df()
    claim_features = mock_claim_features([1, 1], [0, 1])

    # when / then
    with pytest.raises(ValueError, match="Claim features are missing for 1 candidates"):
        merge_claim_features(features_df, claim_features)


def test_merge_claim_features_raises_on_duplicate_candidate():
    # given: candidate (1, 0) appears twice
    features_df = _features_df()
    claim_features = mock_claim_features([1, 1, 2, 1], [0, 1, 0, 0])

    # when / then
    with pytest.raises(ValueError, match="Claim features contain duplicate candidates"):
        merge_claim_features(features_df, claim_features)


def test_claim_feature_names_are_the_hard_and_soft_clm_columns():
    # when
    claim_names = get_claim_feature_names()

    # then: fourteen hard claims followed by four soft claims, none clashing with the scorer
    # or context features
    assert len(claim_names) == 18
    assert all(name.startswith("clm_") for name in claim_names)
    assert [name.startswith("clm_soft_") for name in claim_names] == [False] * 14 + [
        True
    ] * 4
    assert len(set(claim_names)) == len(claim_names)
    assert not set(claim_names) & set(get_feature_names() + get_context_feature_names())


def test_to_features_df_sets_the_library_mz_as_observed_mz():
    # given: scored candidates and their library precursors
    candidate_features = MagicMock()
    candidate_features.to_dict_arrays.return_value = {
        "precursor_idx": [2, 1],
        "rank": [0, 0],
        "score": [1.0, 2.0],
    }
    spectral_library = MagicMock()
    spectral_library.precursor_df = pd.DataFrame(
        {
            "precursor_idx": [1, 2],
            "decoy": [0, 1],
            "elution_group_idx": [0, 0],
            "channel": [0, 0],
            "proteins": ["P1", "P1"],
            "mz_library": [501.0, 502.0],
            "mz_calibrated": [501.1, 502.1],
        }
    )

    # when
    features_df = to_features_df(candidate_features, spectral_library)

    # then: like the quantified precursors, the observed m/z is the library m/z
    assert features_df["mz_observed"].tolist() == [502.0, 501.0]
    assert features_df["mz_library"].tolist() == [502.0, 501.0]

"""Unit tests for the NG mapper helpers around the context features."""

import numpy as np
import pandas as pd
import pytest
from conftest import mock_context_features

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    add_rank_delta_features,
    add_sibling_features,
    get_context_feature_names,
    get_rank_delta_feature_names,
    get_sibling_feature_names,
    merge_context_features,
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


def test_add_rank_delta_features():
    # given: precursor 7 has three candidates, precursor 9 one
    features_df = pd.DataFrame(
        {
            "precursor_idx": [7, 9, 7, 7],
            "rank": [0, 0, 1, 2],
            "score": [5.0, 3.0, 2.0, 4.0],
            "hyperscore_intensity_observation": [1.0, 2.0, 3.0, 3.0],
            "mean_correlation": [0.5, 0.1, 0.2, 0.2],
        }
    )

    # when
    result = add_rank_delta_features(features_df)

    # then
    assert result["rank_delta_score"].tolist() == [1.0, 3.0, -3.0, -1.0]
    assert result["rank_delta_hyperscore_intensity_observation"].tolist() == [
        -2.0,
        2.0,
        0.0,
        0.0,
    ]
    np.testing.assert_allclose(
        result["rank_delta_mean_correlation"], [0.3, 0.1, -0.3, -0.3], atol=1e-6
    )
    assert result["n_other_candidates"].tolist() == [2.0, 0.0, 2.0, 2.0]
    assert set(get_rank_delta_feature_names()) <= set(result.columns)


def test_add_sibling_features():
    # given: precursors 1 and 2 are charge states of one peptide, 3 another peptide, 4 the decoy of the first
    precursor_df = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3, 4],
            "sequence": ["PEPTIDEK", "PEPTIDEK", "OTHERK", "PEPTIDEK"],
            "mods": ["", "", "", ""],
            "mod_sites": ["", "", "", ""],
            "decoy": [0, 0, 0, 1],
            "channel": [0, 0, 0, 0],
        }
    )
    features_df = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 2, 3, 4],
            "rank": [0, 0, 1, 0, 0],
            "rt_observed": [100.0, 101.0, 150.0, 100.0, 100.0],
            "score": [5.0, 4.0, 9.0, 3.0, 2.0],
            "mean_correlation": [0.5, -0.2, 0.9, 0.1, 0.3],
        }
    )

    # when
    result = add_sibling_features(features_df, precursor_df)

    # then: only the co-eluting rank-0 candidates of 1 and 2 see each other
    assert result["n_sibling_coeluting"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert result["sibling_best_score"].tolist() == [4.0, 5.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(
        result["sibling_best_mean_correlation"], [-0.2, 0.5, 0.0, 0.0, 0.0], atol=1e-6
    )
    assert set(get_sibling_feature_names()) <= set(result.columns)

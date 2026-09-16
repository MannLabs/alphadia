"""Unit tests for the NG mapper helpers around the context features."""

import numpy as np
import pandas as pd
import pytest

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    merge_context_features,
)


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precursor_idx": [1, 1, 2],
            "rank": [0, 1, 0],
            "score": [1.0, 0.5, 2.0],
            "decoy": [0, 0, 1],
        }
    )


def _context_features(precursor_idx: list[int], rank: list[int]) -> dict:
    n = len(precursor_idx)
    context_features = {
        "precursor_idx": np.array(precursor_idx, dtype=np.uint64),
        "rank": np.array(rank, dtype=np.uint64),
    }
    for i, name in enumerate(get_context_feature_names()):
        context_features[name] = np.full(n, float(i), dtype=np.float32)
    return context_features


def test_get_context_feature_names():
    # when
    names = get_context_feature_names()

    # then
    assert len(names) == 8
    assert all(name.startswith("ctx_") for name in names)
    assert "ctx_claimant_rank" in names


def test_merge_context_features_leaves_no_nan():
    # given: the context rows come in a different order than the features
    features_df = _features_df()
    context_features = _context_features([2, 1, 1], [0, 1, 0])

    # when
    merged_df = merge_context_features(features_df, context_features)

    # then
    assert len(merged_df) == len(features_df)
    assert list(merged_df["precursor_idx"]) == [1, 1, 2]
    assert list(merged_df["rank"]) == [0, 1, 0]
    for name in get_context_feature_names():
        assert name in merged_df.columns
        assert not merged_df[name].isna().any()
    assert merged_df["ctx_n_competitors"].tolist() == [1.0, 1.0, 1.0]


def test_merge_context_features_raises_on_missing_candidate():
    # given: candidate (2, 0) has no context features
    features_df = _features_df()
    context_features = _context_features([1, 1], [0, 1])

    # when / then
    with pytest.raises(ValueError, match="missing for 1 candidates"):
        merge_context_features(features_df, context_features)


def test_merge_context_features_raises_on_duplicate_candidate():
    # given: candidate (1, 0) appears twice
    features_df = _features_df()
    context_features = _context_features([1, 1, 2, 1], [0, 1, 0, 0])

    # when / then
    with pytest.raises(ValueError, match="duplicate"):
        merge_context_features(features_df, context_features)

"""Unit tests for the NG mapper helpers around the context features."""

import pandas as pd
import pytest
from conftest import mock_context_features

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    inherit_context_features_from_targets,
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


def _target_decoy_features_df(
    precursor_idx: list[int], rank: list[int], context_value: list[float]
) -> pd.DataFrame:
    """Candidates of elution group 0: precursors 0 (target) and 1 (decoy) carry charge 2, 2 (target) and 3 (decoy)
    charge 3, 4 (decoy) charge 4 without a target."""
    df = pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "rank": rank,
            "decoy": [p % 2 if p < 4 else 1 for p in precursor_idx],
            "elution_group_idx": 0,
            "channel": 0,
        }
    )
    for name in get_context_feature_names():
        df[name] = context_value
    return df


_PRECURSOR_DF = pd.DataFrame(
    {"precursor_idx": [0, 1, 2, 3, 4], "charge": [2, 2, 3, 3, 4]}
)


def test_inherit_context_features_copies_the_target_candidate_of_the_same_rank():
    # given: targets 0 and 2 with two ranks each, their decoys 1 and 3 with their own values
    features_df = _target_decoy_features_df(
        [0, 0, 1, 1, 2, 2, 3],
        [0, 1, 0, 1, 0, 1, 1],
        [10.0, 11.0, 90.0, 91.0, 20.0, 21.0, 93.0],
    )

    # when
    result_df = inherit_context_features_from_targets(features_df, _PRECURSOR_DF)

    # then: each decoy takes the value of the target with its charge and rank, the targets keep theirs
    for name in get_context_feature_names():
        assert result_df[name].tolist() == [10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 21.0]


def test_inherit_context_features_falls_back_to_rank_zero_and_keeps_orphans():
    # given: decoy 1 has a rank 2 candidate its target lacks, decoy 4 has no target at all
    features_df = _target_decoy_features_df(
        [0, 0, 1, 4],
        [0, 1, 2, 0],
        [10.0, 11.0, 92.0, 94.0],
    )

    # when
    result_df = inherit_context_features_from_targets(features_df, _PRECURSOR_DF)

    # then
    for name in get_context_feature_names():
        assert result_df[name].tolist() == [10.0, 11.0, 10.0, 94.0]

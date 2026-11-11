"""Unit tests for the alphadia-search-rs mapper."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from alphadia_search_rs import FeatureSet

import alphadia
from alphadia.workflow.peptidecentric.rust.mapper import (
    INTERFERENCE_FEATURE_GROUPS,
    get_classifier_feature_columns,
    get_feature_names,
    get_interference_feature_groups,
    to_features_df,
)

INTERFERENCE_COLUMNS = [
    column for columns in INTERFERENCE_FEATURE_GROUPS.values() for column in columns
]


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


def test_interference_columns_are_full_features_after_the_existing_ones():
    # given / when
    full = get_feature_names(FeatureSet.Full)
    calibration = get_feature_names(FeatureSet.Calibration)

    # then the 7 columns are the last ones of the full set and absent from the calibration set
    assert full[-len(INTERFERENCE_COLUMNS) :] == INTERFERENCE_COLUMNS
    assert not set(INTERFERENCE_COLUMNS) & set(calibration)


def test_get_classifier_feature_columns_without_flags_is_the_baseline():
    # given the feature names of the branch without interference features
    baseline_full = [
        name
        for name in get_feature_names(FeatureSet.Full)
        if name not in INTERFERENCE_COLUMNS
    ]

    # when
    columns = get_classifier_feature_columns()

    # then
    assert columns == baseline_full
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.get_feature_names",
        return_value=list(baseline_full),
    ):
        # a backend without the interference block gives the same columns
        assert get_classifier_feature_columns() == baseline_full
        assert (
            get_classifier_feature_columns(
                ["fitted_spread_features", "window_signal_share_features"]
            )
            == baseline_full
        )


def test_get_classifier_feature_columns_with_both_flags_adds_the_7_columns():
    # given
    baseline = get_classifier_feature_columns()

    # when
    columns = get_classifier_feature_columns(
        ["fitted_spread_features", "window_signal_share_features"]
    )

    # then
    assert columns == get_feature_names(FeatureSet.Full)
    assert columns[: len(baseline)] == baseline
    assert columns[len(baseline) :] == INTERFERENCE_COLUMNS
    assert len(INTERFERENCE_COLUMNS) == 7


@pytest.mark.parametrize("group", list(INTERFERENCE_FEATURE_GROUPS))
def test_get_classifier_feature_columns_adds_only_the_enabled_group(group):
    # given
    baseline = get_classifier_feature_columns()

    # when
    columns = get_classifier_feature_columns([group])

    # then
    assert [c for c in columns if c not in baseline] == list(
        INTERFERENCE_FEATURE_GROUPS[group]
    )


def test_get_classifier_feature_columns_rejects_an_unknown_group():
    with pytest.raises(ValueError, match="Unknown interference feature groups"):
        get_classifier_feature_columns(["apex_spread_features"])


def test_get_interference_feature_groups_reads_the_search_flags():
    # given
    search_config = {
        "fitted_spread_features": False,
        "window_signal_share_features": True,
    }

    # when / then
    assert get_interference_feature_groups(search_config) == [
        "window_signal_share_features"
    ]
    assert get_interference_feature_groups({}) == []


def test_default_config_disables_the_interference_features():
    # given
    with open(
        os.path.join(os.path.dirname(alphadia.__file__), "constants", "default.yaml")
    ) as f:
        search_config = yaml.safe_load(f)["search"]

    # when / then
    assert search_config["fitted_spread_features"] is False
    assert search_config["window_signal_share_features"] is False
    assert get_interference_feature_groups(search_config) == []

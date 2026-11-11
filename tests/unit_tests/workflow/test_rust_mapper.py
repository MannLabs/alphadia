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


# A full feature set with the interference block at its end, so that the tests do not depend on whether the installed
# alphadia-search-rs release already scores the interference features.
CORE_COLUMNS = [
    "correlation_mean",
    "intensity_correlation",
    "mean_overlapping_mass_error",
]
FULL_WITH_INTERFERENCE = CORE_COLUMNS + INTERFERENCE_COLUMNS


@pytest.fixture
def full_feature_names():
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.get_feature_names",
        return_value=list(FULL_WITH_INTERFERENCE),
    ) as mock:
        yield mock


def test_get_classifier_feature_columns_without_flags_is_the_baseline(
    full_feature_names,
):
    # given a full feature set with the interference block

    # when
    columns = get_classifier_feature_columns()

    # then the interference columns are left out
    assert columns == CORE_COLUMNS
    full_feature_names.assert_called_once_with(FeatureSet.Full)


def test_get_classifier_feature_columns_without_the_block_in_rust():
    # given a backend release without the interference block
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.get_feature_names",
        return_value=list(CORE_COLUMNS),
    ):
        # when the flags are on
        columns = get_classifier_feature_columns(
            ["fitted_spread_features", "window_signal_share_features"]
        )

    # then the columns are unchanged
    assert columns == CORE_COLUMNS


def test_get_classifier_feature_columns_with_both_flags_adds_the_7_columns(
    full_feature_names,
):
    # given / when
    columns = get_classifier_feature_columns(
        ["fitted_spread_features", "window_signal_share_features"]
    )

    # then
    assert columns == FULL_WITH_INTERFERENCE
    assert len(INTERFERENCE_COLUMNS) == 7


@pytest.mark.parametrize("group", list(INTERFERENCE_FEATURE_GROUPS))
def test_get_classifier_feature_columns_adds_only_the_enabled_group(
    group, full_feature_names
):
    # given / when
    columns = get_classifier_feature_columns([group])

    # then
    assert columns == CORE_COLUMNS + list(INTERFERENCE_FEATURE_GROUPS[group])


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

"""Unit tests for the peptidecentric workflow utils."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import FeatureSet

import alphadia
from alphadia.outputtransform.quantification.quant_builder import (
    INTENSITY_LIBRARY_COLUMN,
)
from alphadia.workflow.peptidecentric.ng.ng_mapper import get_feature_names
from alphadia.workflow.peptidecentric.utils import (
    INTERFERENCE_FEATURE_GROUPS,
    add_library_fragment_intensity,
    feature_columns,
    get_classifier_feature_columns,
    get_interference_feature_groups,
)

INTERFERENCE_COLUMNS = [
    column for columns in INTERFERENCE_FEATURE_GROUPS.values() for column in columns
]


def test_get_classifier_feature_columns_for_python_backend():
    # when
    columns = get_classifier_feature_columns("python")

    # then
    assert columns == feature_columns


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
    columns = get_classifier_feature_columns("rust")

    # then
    assert columns == baseline_full
    with patch(
        "alphadia.workflow.peptidecentric.utils.get_feature_names",
        return_value=list(baseline_full),
    ):
        # a backend without the interference block gives the same columns
        assert get_classifier_feature_columns("rust") == baseline_full
        assert (
            get_classifier_feature_columns(
                "rust", ["fitted_spread_features", "window_signal_share_features"]
            )
            == baseline_full
        )


def test_get_classifier_feature_columns_with_both_flags_adds_the_7_columns():
    # given
    baseline = get_classifier_feature_columns("rust")

    # when
    columns = get_classifier_feature_columns(
        "rust", ["fitted_spread_features", "window_signal_share_features"]
    )

    # then
    assert columns == get_feature_names(FeatureSet.Full)
    assert columns[: len(baseline)] == baseline
    assert columns[len(baseline) :] == INTERFERENCE_COLUMNS
    assert len(INTERFERENCE_COLUMNS) == 7


@pytest.mark.parametrize("group", list(INTERFERENCE_FEATURE_GROUPS))
def test_get_classifier_feature_columns_adds_only_the_enabled_group(group):
    # given
    baseline = get_classifier_feature_columns("rust")

    # when
    columns = get_classifier_feature_columns("rust", [group])

    # then
    assert [c for c in columns if c not in baseline] == list(
        INTERFERENCE_FEATURE_GROUPS[group]
    )


def test_get_classifier_feature_columns_rejects_an_unknown_group():
    with pytest.raises(ValueError, match="Unknown interference feature groups"):
        get_classifier_feature_columns("rust", ["apex_spread_features"])


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


def _flat_library():
    library = SpecLibFlat()
    library._precursor_df = pd.DataFrame(
        {
            "precursor_idx": [5, 7],
            "flat_frag_start_idx": [0, 2],
            "flat_frag_stop_idx": [2, 5],
        }
    )
    library._fragment_df = pd.DataFrame(
        {
            "number": [1, 2, 1, 2, 3],
            "type": [121, 121, 98, 121, 121],
            "charge": [1, 1, 1, 1, 2],
            "loss_type": [0, 0, 0, 0, 0],
            "intensity": [1.0, 0.5, 0.2, 1.0, 0.7],
        }
    )
    return library


def test_add_library_fragment_intensity_matches_fragments_by_ion():
    # given quantified fragments in a different order than the library, one of them unknown to it
    fragments_df = pd.DataFrame(
        {
            "precursor_idx": np.array([7, 5, 7, 5], dtype=np.uint32),
            "number": np.array([3, 2, 1, 9], dtype=np.uint8),
            "type": np.array([121, 121, 98, 121], dtype=np.uint8),
            "charge": np.array([2, 1, 1, 1], dtype=np.uint8),
            "loss_type": np.array([0, 0, 0, 0], dtype=np.uint8),
            "intensity": [10.0, 20.0, 30.0, 40.0],
        }
    )

    # when
    result = add_library_fragment_intensity(fragments_df, _flat_library())

    # then
    np.testing.assert_allclose(result[INTENSITY_LIBRARY_COLUMN], [0.7, 0.5, 0.2, 0.0])
    np.testing.assert_allclose(result["intensity"], fragments_df["intensity"])

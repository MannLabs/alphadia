"""Unit tests for the alphadia-search-rs mapper."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from alphadia_search_rs import FeatureSet

import alphadia
from alphadia.calibration.ms1_calibration import Ms1Calibration
from alphadia.workflow.peptidecentric.rust.mapper import (
    MS1_ISOTOPE_COLUMNS,
    OPTIONAL_FEATURE_GROUPS,
    dia_data_to_rust,
    get_classifier_feature_columns,
    get_enabled_feature_groups,
    get_feature_group_columns,
    set_ms1_isotopes,
    to_features_df,
)

OPTIONAL_COLUMNS = [
    column for columns in OPTIONAL_FEATURE_GROUPS.values() for column in columns
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


# A full feature set with the optional blocks at its end, so that the tests do not depend on whether the installed
# alphadia-search-rs release already scores the optional features.
CORE_COLUMNS = [
    "correlation_mean",
    "intensity_correlation",
    "mean_overlapping_mass_error",
]
FULL_WITH_OPTIONAL = CORE_COLUMNS + OPTIONAL_COLUMNS


@pytest.fixture
def full_feature_names():
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.get_feature_names",
        return_value=list(FULL_WITH_OPTIONAL),
    ) as mock:
        yield mock


def test_get_classifier_feature_columns_without_flags_is_the_baseline(
    full_feature_names,
):
    # given a full feature set with the optional blocks

    # when
    columns = get_classifier_feature_columns()

    # then the optional columns are left out
    assert columns == CORE_COLUMNS
    full_feature_names.assert_called_once_with(FeatureSet.Full)


def test_get_classifier_feature_columns_without_the_block_in_rust():
    # given a backend release without the optional blocks
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.get_feature_names",
        return_value=list(CORE_COLUMNS),
    ):
        # when the flags are on
        columns = get_classifier_feature_columns(list(OPTIONAL_FEATURE_GROUPS))

    # then the columns are unchanged
    assert columns == CORE_COLUMNS


def test_get_classifier_feature_columns_with_all_flags_adds_the_26_columns(
    full_feature_names,
):
    # given / when
    columns = get_classifier_feature_columns(list(OPTIONAL_FEATURE_GROUPS))

    # then: 4 fitted spread, 3 window share, 10 MS1, 6 mobility profile and 3 detection columns
    assert columns == FULL_WITH_OPTIONAL
    assert [
        len(OPTIONAL_FEATURE_GROUPS[group]) for group in OPTIONAL_FEATURE_GROUPS
    ] == [
        4,
        3,
        10,
        6,
        3,
    ]


@pytest.mark.parametrize("group", list(OPTIONAL_FEATURE_GROUPS))
def test_get_classifier_feature_columns_adds_only_the_enabled_group(
    group, full_feature_names
):
    # given / when
    columns = get_classifier_feature_columns([group])

    # then
    assert columns == CORE_COLUMNS + list(OPTIONAL_FEATURE_GROUPS[group])


def test_get_classifier_feature_columns_rejects_an_unknown_group():
    with pytest.raises(ValueError, match="Unknown optional feature groups"):
        get_classifier_feature_columns(["apex_spread_features"])


def test_get_feature_group_columns_lists_the_columns_of_the_groups():
    # given / when
    columns = get_feature_group_columns(["detection_features", "ms1_features"])

    # then: in the order of OPTIONAL_FEATURE_GROUPS
    assert columns == [
        *OPTIONAL_FEATURE_GROUPS["ms1_features"],
        *OPTIONAL_FEATURE_GROUPS["detection_features"],
    ]
    assert all(
        column.startswith("ms1_") for column in OPTIONAL_FEATURE_GROUPS["ms1_features"]
    )


def test_get_enabled_feature_groups_reads_the_search_flags():
    # given
    search_config = {
        "fitted_spread_features": False,
        "window_signal_share_features": True,
    }

    # when / then
    assert get_enabled_feature_groups(search_config) == ["window_signal_share_features"]
    assert get_enabled_feature_groups({}) == []


def _default_search_config() -> dict:
    with open(
        os.path.join(os.path.dirname(alphadia.__file__), "constants", "default.yaml")
    ) as f:
        return yaml.safe_load(f)["search"]


def test_default_config_enables_the_interference_features():
    # given
    search_config = _default_search_config()

    # when / then
    assert search_config["fitted_spread_features"] is True
    assert search_config["window_signal_share_features"] is True
    assert {"fitted_spread_features", "window_signal_share_features"} <= set(
        get_enabled_feature_groups(search_config)
    )


def test_default_config_enables_the_ms1_and_detection_features():
    # given
    search_config = _default_search_config()

    # when / then
    assert search_config["ms1_features"] is True
    assert search_config["detection_features"] is True
    assert {"ms1_features", "detection_features"} <= set(
        get_enabled_feature_groups(search_config)
    )


def _precursor_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "elution_group_idx": [5, 5, 9],
            "decoy": [0, 1, 1],
            "charge": [2, 2, 3],
            "mz_library": [500.0, 507.0, 610.0],
            "rt": [10.0, 10.0, 20.0],
            "i_0": [1.0, 1.0, 1.0],
            "i_1": [0.6, 0.7, 0.8],
            "i_2": [0.2, 0.3, 0.4],
            "i_3": [0.05, 0.1, 0.1],
            "channel": 0,
            "proteins": "P",
        }
    )


def _candidate_features() -> SimpleNamespace:
    columns = {
        "precursor_idx": np.array([0, 1, 2]),
        "rank": np.array([0, 0, 0]),
        "score": np.array([1.0, 2.0, 3.0]),
    }
    for k, name in enumerate(OPTIONAL_COLUMNS):
        columns[name] = np.full(3, float(k))
    return SimpleNamespace(to_dict_arrays=lambda: columns)


@pytest.mark.parametrize(
    "unscored_feature_groups",
    [
        [],
        ["ms1_features"],
        ["detection_features"],
        ["ms1_features", "detection_features"],
    ],
)
def test_to_features_df_drops_the_unscored_groups(unscored_feature_groups):
    # given
    library = SimpleNamespace(precursor_df=_precursor_df())

    # when
    features_df = to_features_df(
        _candidate_features(),
        library,
        unscored_feature_groups=unscored_feature_groups,
    )

    # then: without a mobility dimension, the mobility profile columns are dropped as well
    for group, columns in OPTIONAL_FEATURE_GROUPS.items():
        dropped = (
            group in unscored_feature_groups or group == "mobility_profile_features"
        )
        assert all((column not in features_df.columns) == dropped for column in columns)
    assert "score" in features_df.columns


class _RecordingSpeclib:
    def set_isotopes(self, *arrays):
        self.arrays = arrays


def test_set_ms1_isotopes_scores_every_precursor_at_its_own_calibrated_mass():
    # given: a calibration of +2 ppm over the whole m/z range
    speclib = _RecordingSpeclib()
    calibration = Ms1Calibration(
        np.array([400.0, 700.0]),
        np.array([2.0, 2.0]),
        np.array([0.0]),
        np.array([0.0]),
        1.0,
        4.0,
        500,
    )

    # when
    set_ms1_isotopes(speclib, _precursor_df(), "mz_library", "rt", calibration)

    # then: decoys keep their own m/z, charge and envelope
    idx, charge, mz, intensity = speclib.arrays
    assert idx.tolist() == [0, 1, 2]
    assert charge.tolist() == [2, 2, 3]
    np.testing.assert_allclose(mz, np.array([500.0, 507.0, 610.0]) * (1 + 2e-6))
    np.testing.assert_allclose(intensity[1], [1.0, 0.7, 0.3, 0.1])
    assert intensity.shape == (3, 4)


def test_set_ms1_isotopes_uses_the_isotopes_the_library_has():
    # given: a library with three isotopes only
    speclib = _RecordingSpeclib()
    precursor_df = _precursor_df().drop(columns="i_3")

    # when
    set_ms1_isotopes(
        speclib, precursor_df, "mz_library", "rt", Ms1Calibration.identity()
    )

    # then
    assert speclib.arrays[3].shape == (3, 3)


def test_set_ms1_isotopes_rejects_a_library_without_isotopes():
    # given
    precursor_df = _precursor_df().drop(columns=MS1_ISOTOPE_COLUMNS)

    # when / then
    with pytest.raises(ValueError, match="isotope abundances"):
        set_ms1_isotopes(
            _RecordingSpeclib(),
            precursor_df,
            "mz_library",
            "rt",
            Ms1Calibration.identity(),
        )


def _dia_data(cycle_positions=None):
    return SimpleNamespace(
        spectrum_df=pd.DataFrame(
            {
                "isolation_lower_mz": [-1.0, 400.0, -1.0, 400.0],
                "isolation_upper_mz": [-1.0, 600.0, -1.0, 600.0],
                "peak_start_idx": [0, 1, 2, 3],
                "peak_stop_idx": [1, 2, 3, 4],
                "rt": [1.0, 1.1, 2.0, 2.1],
            }
        ),
        peak_df=pd.DataFrame({"mz": [1.0, 2.0, 3.0, 4.0], "intensity": [1.0] * 4}),
        cycle=np.zeros((1, 2, 1, 2)),
        cycle_positions=cycle_positions,
    )


@patch("alphadia.workflow.peptidecentric.rust.mapper.DiaDataRust")
def test_dia_data_to_rust_passes_the_cycle_positions_of_aligned_ms1_scans(
    mock_dia_data_rust,
):
    # when
    dia_data_rust = dia_data_to_rust(_dia_data(np.array([0.5, 0.0])))

    # then
    dia_data_rust.set_cycle_positions.assert_called_once_with([0.5, 0.0])
    assert dia_data_rust is mock_dia_data_rust.from_arrays.return_value


@patch("alphadia.workflow.peptidecentric.rust.mapper.DiaDataRust")
def test_dia_data_to_rust_keeps_the_default_cycle_positions(mock_dia_data_rust):
    # when
    dia_data_rust = dia_data_to_rust(_dia_data())

    # then
    dia_data_rust.set_cycle_positions.assert_not_called()

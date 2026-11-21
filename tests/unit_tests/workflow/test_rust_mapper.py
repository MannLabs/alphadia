"""Unit tests for the alphadia-search-rs mapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from alphadia.workflow.peptidecentric.rust.mapper import speclib_to_rust, to_features_df


def _mock_speclib(*, include_charge: bool = True) -> MagicMock:
    precursor_data = {
        "precursor_idx": [0],
        "mz_library": [500.0],
        "mz_calibrated": [500.1],
        "rt_library": [10.0],
        "rt_calibrated": [10.1],
        "nAA": [8],
        "flat_frag_start_idx": [0],
        "flat_frag_stop_idx": [1],
    }
    if include_charge:
        precursor_data["charge"] = [3]

    speclib = MagicMock()
    speclib.precursor_df = pd.DataFrame(precursor_data)
    speclib.fragment_df = pd.DataFrame(
        {
            "mz_library": [300.0],
            "mz_calibrated": [300.05],
            "intensity": [100.0],
            "cardinality": [1],
            "charge": [1],
            "loss_type": [0],
            "number": [3],
            "position": [3],
            "type": [121],
        }
    )
    return speclib


def test_speclib_to_rust_passes_precursor_charge_when_present():
    speclib = _mock_speclib(include_charge=True)
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.SpecLibFlatRust.from_arrays"
    ) as mock_from_arrays:
        speclib_to_rust(
            speclib,
            rt_column="rt_calibrated",
            precursor_mz_column="mz_calibrated",
            fragment_mz_column="mz_calibrated",
        )
    charge_arg = mock_from_arrays.call_args.kwargs["precursor_charge"]
    assert np.array_equal(charge_arg, np.array([3], dtype=np.uint8))


def test_speclib_to_rust_handles_missing_precursor_charge():
    speclib = _mock_speclib(include_charge=False)
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.SpecLibFlatRust.from_arrays"
    ) as mock_from_arrays:
        speclib_to_rust(
            speclib,
            rt_column="rt_calibrated",
            precursor_mz_column="mz_calibrated",
            fragment_mz_column="mz_calibrated",
        )
    assert mock_from_arrays.call_args.kwargs["precursor_charge"] is None


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

"""Unit tests for the alphadia-search-rs mapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from alphadia.workflow.peptidecentric.rust.mapper import speclib_to_rust, to_features_df


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


@patch("alphadia.workflow.peptidecentric.rust.mapper.SpecLibFlatRust")
def test_speclib_to_rust_sets_the_elution_group_of_each_precursor(mock_speclib_rust):
    # Given: a library with a target and its decoy in elution group 4, and a target in elution group 9
    precursor_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "elution_group_idx": [4, 4, 9],
            "mz_library": [500.0, 501.0, 600.0],
            "rt_library": [10.0, 10.0, 20.0],
            "nAA": [10, 10, 12],
            "flat_frag_start_idx": [0, 1, 2],
            "flat_frag_stop_idx": [1, 2, 3],
        }
    )
    fragment_df = pd.DataFrame(
        {
            column: [1, 1, 1]
            for column in [
                "mz_library",
                "intensity",
                "cardinality",
                "charge",
                "loss_type",
                "number",
                "position",
                "type",
            ]
        }
    )
    spectral_library = MagicMock(precursor_df=precursor_df, fragment_df=fragment_df)

    # When: the library is converted to the Rust format
    speclib_to_rust(
        spectral_library,
        rt_column="rt_library",
        precursor_mz_column="mz_library",
        fragment_mz_column="mz_library",
    )

    # Then: the Rust library gets the elution group of each precursor after its number of amino acids
    arrays = mock_speclib_rust.from_arrays.call_args.args
    precursor_naa, elution_group_idx = arrays[5], arrays[6]
    np.testing.assert_array_equal(precursor_naa, [10, 10, 12])
    np.testing.assert_array_equal(elution_group_idx, [4, 4, 9])
    assert elution_group_idx.dtype == np.uint64

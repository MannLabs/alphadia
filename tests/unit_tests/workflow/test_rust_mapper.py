"""Unit tests for the alphadia-search-rs mapper."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from alphadia.constants.keys import MbrCols
from alphadia.workflow.peptidecentric.rust.mapper import to_features_df


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


@pytest.mark.parametrize("library_has_column", [True, False])
def test_to_features_df_carries_the_identified_in_run_column_of_the_library(
    library_has_column,
):
    # given
    candidate_features = MagicMock()
    candidate_features.to_dict_arrays.return_value = {
        "precursor_idx": [2, 1],
        "rank": [0, 0],
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
        }
    )
    if library_has_column:
        spectral_library.precursor_df[MbrCols.IDENTIFIED_IN_RUN] = [1, 0]

    # when
    features_df = to_features_df(candidate_features, spectral_library)

    # then
    if library_has_column:
        assert features_df[MbrCols.IDENTIFIED_IN_RUN].tolist() == [0, 1]
    else:
        assert MbrCols.IDENTIFIED_IN_RUN not in features_df.columns

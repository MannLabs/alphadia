"""Unit tests for the optimization features."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alphadia.workflow.optimizers.features import (
    MeanIsotopeIntensityCorrelation,
    PrecursorProportionDetected,
)


def _optlock(total_elution_groups: int) -> MagicMock:
    optlock = MagicMock()
    optlock.total_elution_groups = total_elution_groups
    return optlock


def test_precursor_proportion_detected_returns_ratio():
    """Test that the feature divides the number of precursors by the elution groups."""
    # given
    feature = PrecursorProportionDetected()
    precursors_df = pd.DataFrame({"precursor_idx": np.arange(500)})

    # when
    value = feature.measure(precursors_df, pd.DataFrame(), _optlock(2000))

    # then
    assert value == 500 / 2000


def test_precursor_proportion_detected_returns_zero_for_empty_precursors():
    """Test that a batch with elution groups and no precursor measures zero.

    This value is a correct measurement. It is not a missing measurement.
    """
    # given
    feature = PrecursorProportionDetected()

    # when
    value = feature.measure(pd.DataFrame(), pd.DataFrame(), _optlock(2000))

    # then
    assert value == 0.0


def test_precursor_proportion_detected_returns_none_for_empty_batch():
    """Test that a batch with no elution group gives no measurement."""
    # given
    feature = PrecursorProportionDetected()

    # when
    value = feature.measure(pd.DataFrame(), pd.DataFrame(), _optlock(0))

    # then
    assert value is None


def test_mean_isotope_intensity_correlation_returns_mean():
    """Test that the feature calculates the mean over the precursors."""
    # given
    feature = MeanIsotopeIntensityCorrelation()
    precursors_df = pd.DataFrame({"isotope_intensity_correlation": [0.2, 0.4, 0.6]})

    # when
    value = feature.measure(precursors_df, pd.DataFrame(), _optlock(2000))

    # then
    assert value == pytest.approx(0.4)


def test_mean_isotope_intensity_correlation_returns_none_for_empty_precursors():
    """Test that an empty precursor frame gives None and not NaN."""
    # given
    feature = MeanIsotopeIntensityCorrelation()
    precursors_df = pd.DataFrame({"isotope_intensity_correlation": []})

    # when
    value = feature.measure(precursors_df, pd.DataFrame(), _optlock(2000))

    # then
    assert value is None


def test_mean_isotope_intensity_correlation_returns_none_for_all_nan_column():
    """Test that a column with only NaN gives None.

    A NaN moves through the subsequent calculations without a message.
    """
    # given
    feature = MeanIsotopeIntensityCorrelation()
    precursors_df = pd.DataFrame(
        {"isotope_intensity_correlation": [np.nan, np.nan, np.nan]}
    )

    # when
    value = feature.measure(precursors_df, pd.DataFrame(), _optlock(2000))

    # then
    assert value is None


def test_feature_names():
    """Test that the features keep the history column names of the previous code."""
    # given / when / then
    assert PrecursorProportionDetected.name == "precursor_proportion_detected"
    assert MeanIsotopeIntensityCorrelation.name == "mean_isotope_intensity_correlation"

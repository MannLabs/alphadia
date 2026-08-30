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
    """Tests that the precursor proportion is the number of precursors over the elution groups."""
    # given
    feature = PrecursorProportionDetected()
    precursors_df = pd.DataFrame({"precursor_idx": np.arange(500)})

    # when
    value = feature.measure(precursors_df, pd.DataFrame(), _optlock(2000))

    # then
    assert value == 500 / 2000


def test_precursor_proportion_detected_returns_zero_for_empty_precursors():
    """Tests that no precursors in a non-empty batch is a measurement of zero, not a missing measurement."""
    # given
    feature = PrecursorProportionDetected()

    # when
    value = feature.measure(pd.DataFrame(), pd.DataFrame(), _optlock(2000))

    # then
    assert value == 0.0


def test_precursor_proportion_detected_returns_none_for_empty_batch():
    """Tests that a batch without elution groups admits no measurement."""
    # given
    feature = PrecursorProportionDetected()

    # when
    value = feature.measure(pd.DataFrame(), pd.DataFrame(), _optlock(0))

    # then
    assert value is None


def test_mean_isotope_intensity_correlation_returns_mean():
    """Tests that the mean isotope intensity correlation is averaged over the precursors."""
    # given
    feature = MeanIsotopeIntensityCorrelation()
    precursors_df = pd.DataFrame({"isotope_intensity_correlation": [0.2, 0.4, 0.6]})

    # when
    value = feature.measure(precursors_df, pd.DataFrame(), _optlock(2000))

    # then
    assert value == pytest.approx(0.4)


def test_mean_isotope_intensity_correlation_returns_none_for_empty_precursors():
    """Tests that an empty precursor frame yields None rather than NaN."""
    # given
    feature = MeanIsotopeIntensityCorrelation()
    precursors_df = pd.DataFrame({"isotope_intensity_correlation": []})

    # when
    value = feature.measure(precursors_df, pd.DataFrame(), _optlock(2000))

    # then
    assert value is None


def test_feature_names():
    """Tests that the features keep the history column names used before the refactor."""
    # given / when / then
    assert PrecursorProportionDetected.name == "precursor_proportion_detected"
    assert MeanIsotopeIntensityCorrelation.name == "mean_isotope_intensity_correlation"

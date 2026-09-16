"""Unit tests for the optimization features."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphadia.workflow.optimizers.features import (
    MeanIsotopeIntensityCorrelation,
    PrecursorProportionDetected,
)


def _optlock(total_elution_groups: int) -> SimpleNamespace:
    return SimpleNamespace(total_elution_groups=total_elution_groups)


def test_precursor_proportion_detected_returns_ratio():
    """Test that the feature divides the number of precursors by the elution groups."""
    # given
    precursors_df = pd.DataFrame({"precursor_idx": np.arange(500)})

    # when
    value = PrecursorProportionDetected.measure(
        precursors_df, pd.DataFrame(), _optlock(2000)
    )

    # then
    assert value == 500 / 2000


def test_precursor_proportion_detected_returns_zero_for_empty_precursors():
    """Test that a batch with elution groups and no precursor measures zero."""
    # given / when
    value = PrecursorProportionDetected.measure(
        pd.DataFrame(), pd.DataFrame(), _optlock(2000)
    )

    # then
    assert value == 0.0


def test_mean_isotope_intensity_correlation_returns_mean():
    """Test that the feature calculates the mean over the precursors."""
    # given
    precursors_df = pd.DataFrame({"isotope_intensity_correlation": [0.2, 0.4, 0.6]})

    # when
    value = MeanIsotopeIntensityCorrelation.measure(
        precursors_df, pd.DataFrame(), _optlock(2000)
    )

    # then
    assert value == pytest.approx(0.4)


def test_mean_isotope_intensity_correlation_is_nan_without_precursors():
    """Test that a round without precursors measures NaN, which no maximum search picks."""
    # given
    precursors_df = pd.DataFrame(
        {"isotope_intensity_correlation": pd.Series([], dtype=float)}
    )

    # when
    value = MeanIsotopeIntensityCorrelation.measure(
        precursors_df, pd.DataFrame(), _optlock(2000)
    )

    # then
    assert np.isnan(value)

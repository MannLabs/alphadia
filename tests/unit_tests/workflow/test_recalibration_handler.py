"""Unit tests for the recalibration handler."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphadia.constants.settings import PeakGroupScoreCutoff
from alphadia.workflow.peptidecentric.recalibration_handler import (
    decoy_probit_score_cutoff,
    peak_parameters,
)


@pytest.mark.parametrize(
    ("memory_saving", "factor", "percentile"),
    [
        (False, PeakGroupScoreCutoff.FACTOR, PeakGroupScoreCutoff.PERCENTILE),
        (
            True,
            PeakGroupScoreCutoff.MEMORY_SAVING_FACTOR,
            PeakGroupScoreCutoff.MEMORY_SAVING_PERCENTILE,
        ),
    ],
)
def test_peak_parameters_take_the_median_widths_and_a_low_score_percentile(
    memory_saving, factor, percentile
):
    """Test that the peak widths are the medians of the identifications and the score cutoff a factor times a low
    percentile of their scores, higher when saving memory."""
    # Given: 101 identifications with scores 0 to 100
    identifications = pd.DataFrame(
        {
            "fwhm_rt": [4.0] * 50 + [6.0] * 51,
            "fwhm_mobility": [0.01] * 101,
            "score": [float(score) for score in range(101)],
        }
    )

    # When: the peak parameters are derived
    parameters = peak_parameters(identifications, memory_saving)

    # Then: the widths are the medians and the cutoff the factor times the percentile score
    assert parameters.fwhm_rt == 6.0
    assert parameters.fwhm_mobility == 0.01
    assert parameters.score_cutoff == pytest.approx(factor * percentile)


def test_decoy_probit_score_cutoff_extrapolates_the_identifications_on_the_decoy_scale():
    """Test that the cutoff is the decoy score at the accepted-loss quantile of a normal fitted to the probit of the
    identification scores on the decoy score distribution."""
    # Given: decoy scores from a standard normal, and identification scores whose decoy-CDF probits follow N(3, 0.5)
    rng = np.random.default_rng(0)
    decoy_scores = rng.normal(0, 1, 200_000)
    identification_scores = rng.normal(3, 0.5, 5_000)

    # When: the cutoff for an accepted loss of 0.3% is estimated
    cutoff = decoy_probit_score_cutoff(identification_scores, decoy_scores, 0.003)

    # Then: it is the decoy score at Phi(3 + 0.5 * Phi^-1(0.003)), about 1.63
    assert cutoff == pytest.approx(3 + 0.5 * stats.norm.ppf(0.003), abs=0.05)


def test_decoy_probit_score_cutoff_needs_decoys_and_identifications():
    """Test that no cutoff is estimated without decoys or with fewer than two identifications."""
    assert decoy_probit_score_cutoff(np.array([5.0, 6.0]), np.array([]), 0.003) is None
    assert decoy_probit_score_cutoff(np.array([5.0]), np.arange(100.0), 0.003) is None


def test_peak_parameters_take_the_decoy_probit_cutoff_with_decoy_scores():
    """Test that the score cutoff is the decoy-probit estimate when the decoy scores of the search are given."""
    # Given: identifications and the decoy scores of their search
    rng = np.random.default_rng(1)
    identifications = pd.DataFrame(
        {
            "fwhm_rt": [4.0] * 2000,
            "fwhm_mobility": [0.01] * 2000,
            "score": rng.normal(3, 0.5, 2000),
        }
    )
    decoy_scores = rng.normal(0, 1, 50_000)

    # When: the peak parameters are derived with and without the decoy scores
    with_decoys = peak_parameters(identifications, False, decoy_scores)
    memory_saving = peak_parameters(identifications, True, decoy_scores)

    # Then: the cutoff is the decoy-probit estimate, except when saving memory, which keeps its percentile rule
    assert with_decoys.score_cutoff == pytest.approx(
        decoy_probit_score_cutoff(
            identifications["score"].to_numpy(),
            decoy_scores,
            PeakGroupScoreCutoff.ACCEPTED_LOSS,
        )
    )
    assert memory_saving.score_cutoff == pytest.approx(
        PeakGroupScoreCutoff.MEMORY_SAVING_FACTOR
        * np.percentile(
            identifications["score"], PeakGroupScoreCutoff.MEMORY_SAVING_PERCENTILE
        )
    )

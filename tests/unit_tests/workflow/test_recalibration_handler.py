"""Unit tests for the recalibration handler."""

import pandas as pd
import pytest

from alphadia.constants.settings import PeakGroupScoreCutoff
from alphadia.workflow.peptidecentric.recalibration_handler import peak_parameters


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

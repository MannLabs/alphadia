"""Unit tests for the score cutoff of the recalibration handler."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alphadia.workflow.peptidecentric.recalibration_handler import (
    RecalibrationHandler,
)

N_CONFIDENT = 1000
N_BORDERLINE = 30
CONFIDENT_SCORE = 20.0
BORDERLINE_SCORE = 2.0


def _recalibrated_cutoff(precursor_df: pd.DataFrame) -> float:
    optimization_manager = MagicMock(name="optimization_manager")
    handler = RecalibrationHandler(
        config={
            "search": {
                "target_num_candidates": 3,
                "optimized_peak_group_score": False,
            }
        },
        optimization_manager=optimization_manager,
        calibration_manager=MagicMock(name="calibration_manager"),
        reporter=MagicMock(name="reporter"),
        figure_path=None,
        dia_data_has_ms1=True,
    )
    handler.recalibrate(precursor_df, pd.DataFrame())
    return optimization_manager.update.call_args_list[-1].kwargs["score_cutoff"]


def _precursors(n_confident: int, n_borderline: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": np.r_[
                np.full(n_confident, CONFIDENT_SCORE),
                np.full(n_borderline, BORDERLINE_SCORE),
            ],
            "qval": np.r_[np.zeros(n_confident), np.full(n_borderline, 0.009)],
            "fwhm_rt": 3.0,
            "fwhm_mobility": 0.01,
        }
    )


def test_score_cutoff_ignores_low_scoring_identifications_near_the_fdr_threshold():
    # given: 3 % of the identifications are low-scoring and pass only just below 1 % FDR
    precursor_df = _precursors(N_CONFIDENT, N_BORDERLINE)

    # when
    cutoff = _recalibrated_cutoff(precursor_df)

    # then: the cutoff comes from the confident identifications, not from the borderline ones
    assert cutoff == pytest.approx(RecalibrationHandler.OPTIMIZED_FAC * CONFIDENT_SCORE)


def test_score_cutoff_uses_all_identifications_when_few_are_confident():
    # given: fewer confident identifications than needed for a percentile
    precursor_df = _precursors(
        RecalibrationHandler.SCORE_CUTOFF_MIN_CONFIDENT - 1, N_CONFIDENT
    )

    # when
    cutoff = _recalibrated_cutoff(precursor_df)

    # then: the borderline identifications take part
    assert cutoff == pytest.approx(
        RecalibrationHandler.OPTIMIZED_FAC * BORDERLINE_SCORE
    )

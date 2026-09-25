"""Unit tests for the score cutoff of the recalibration handler."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alphadia.workflow.peptidecentric.recalibration_handler import (
    RecalibrationHandler,
)

N_TRUE = 980
N_FALSE = 20
TRUE_SCORE = 20.0
FALSE_SCORE = 2.0


def test_score_cutoff_is_not_set_by_the_false_identifications_in_the_low_tail():
    # given: 2 % of the calibration identifications are false and score low
    precursor_df = pd.DataFrame(
        {
            "score": np.r_[np.full(N_TRUE, TRUE_SCORE), np.full(N_FALSE, FALSE_SCORE)],
            "fwhm_rt": 3.0,
            "fwhm_mobility": 0.01,
        }
    )
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

    # when
    handler.recalibrate(precursor_df, pd.DataFrame())

    # then: the cutoff comes from the true identifications
    cutoff = optimization_manager.update.call_args_list[-1].kwargs["score_cutoff"]
    assert cutoff == pytest.approx(RecalibrationHandler.OPTIMIZED_FAC * TRUE_SCORE)

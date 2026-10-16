"""Unit test for the peptidecentric module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphadia.exceptions import NoOptimizationLockTargetError
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow


@pytest.fixture
def mock_config():
    return {
        "general": {"save_figures": True, "reuse_calibration": False},
        "output_directory": "",
        "calibration": {"min_correlation": 0.55, "max_fragments": 3, "fdr": 0.01},
    }


def test_filters_precursors_and_fragments_correctly(mock_config):
    """Test that the filter_dfs method filters precursors and fragments correctly."""
    precursor_df = pd.DataFrame(
        {"qval": [0.005, 0.005, 0.011], "decoy": [0, 1, 0], "precursor_idx": [1, 2, 3]}
    )
    fragments_df = pd.DataFrame(
        {
            "precursor_idx": [1, 1, 1, 2, 1, 1],
            "mass_error": [1, 3, -2, 5, -201, 1],
            "correlation": [0.7, 0.5, 0.8, 0.6, 0.9, 0.95],
        }
    )
    instance = PeptideCentricWorkflow("test_instance", mock_config)
    instance.reporter = MagicMock()
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        instance._optimization_handler = OptimizationHandler(
            mock_config,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    # when
    filtered_precursors, filtered_fragments = (
        instance._optimization_handler._filter_dfs(precursor_df, fragments_df)
    )

    pd.testing.assert_frame_equal(
        filtered_precursors,
        pd.DataFrame(
            {
                "qval": [0.005],
                "decoy": [0],
                "precursor_idx": [1],
            }
        ),
    )

    pd.testing.assert_frame_equal(
        filtered_fragments.reset_index(drop=True),
        pd.DataFrame(
            {
                "precursor_idx": [1, 1, 1],
                "mass_error": [1, -2, 1],
                "correlation": [0.95, 0.8, 0.7],
            }
        ),
        check_like=True,
    )


def _create_handler_that_exhausts_the_batch_plan(total_elution_groups: int):
    """Make a handler whose optimization lock runs out of batches before the target."""
    optimizer = MagicMock(has_converged=False, parameter_name="ms2_error")
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        handler = OptimizationHandler(
            {"calibration": {"max_steps": 20, "fdr": 0.01}},
            SimpleNamespace(ms2_error=100.0),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = False
    handler._optlock.total_elution_groups = total_elution_groups
    handler._get_ordered_optimizers = MagicMock(return_value=[[optimizer]])
    handler._process_batch = MagicMock(return_value=pd.DataFrame())
    handler._filter_dfs = MagicMock(return_value=(pd.DataFrame(), pd.DataFrame()))
    return handler, optimizer


def test_search_parameter_optimization_raises_without_candidates():
    """Test that a raw file without a single candidate fails instead of running unoptimized."""
    # given
    handler, _ = _create_handler_that_exhausts_the_batch_plan(total_elution_groups=0)

    # when / then
    with (
        patch(
            "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
        ),
        pytest.raises(NoOptimizationLockTargetError),
    ):
        handler.search_parameter_optimization()


def test_search_parameter_optimization_recovers_with_insufficient_precursors():
    """Test that a raw file with candidates but too few precursors takes the recovery path."""
    # given
    handler, optimizer = _create_handler_that_exhausts_the_batch_plan(
        total_elution_groups=2000
    )

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # then
    optimizer.proceed_with_insufficient_precursors.assert_called_once()


@pytest.mark.parametrize(
    ("n_found", "expect_linear"),
    [(150, False), (250, True)],
)
def test_linear_calibration_falls_back_to_the_mlp_below_the_lock_target(
    n_found, expect_linear
):
    """The logistic regression scores a step only if it finds as many precursors as the lock needs."""
    # given: a logistic regression that finds n_found targets at q < linear_fdr
    config = {
        "calibration": {
            "fdr": 0.01,
            "linear_fdr": 0.05,
            "optimization_lock_target": 200,
        }
    }
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        handler = OptimizationHandler(
            config,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            linear_fdr_manager=MagicMock(),
        )
    precursor_df = pd.DataFrame({"qval": [0.03] * n_found, "decoy": [0] * n_found})
    extraction_handler = MagicMock()
    extraction_handler.perform_fdr_and_filter_candidates.return_value = (
        None,
        precursor_df,
    )

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.ExtractionHandler.create_handler",
        return_value=extraction_handler,
    ):
        result = handler._perform_linear_fdr(pd.DataFrame())

    # then: enough precursors keep the logistic regression and its q-value cutoff, too few fall back for good
    if expect_linear:
        assert result is precursor_df
        assert handler._step_fdr == 0.05
        assert handler._linear_fdr_manager is not None
    else:
        assert result is None
        assert handler._step_fdr == 0.01
        assert handler._linear_fdr_manager is None

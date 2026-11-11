"""Unit test for the peptidecentric module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphadia.exceptions import NoOptimizationLockTargetError
from alphadia.workflow.peptidecentric.python.optimization_handler import (
    PythonOptimizationHandler,
)
from alphadia.workflow.peptidecentric.rust.optimization_handler import (
    RustOptimizationHandler,
)
from alphadia.workflow.peptidecentric.rust.workflow import RustPeptideCentricWorkflow


@pytest.fixture
def mock_config():
    return {
        "general": {"save_figures": True, "reuse_calibration": False},
        "output_directory": "",
        "calibration": {"min_correlation": 0.55, "max_fragments": 3},
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
    instance = RustPeptideCentricWorkflow("test_instance", mock_config)
    instance.reporter = MagicMock()
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        instance._optimization_handler = RustOptimizationHandler(
            mock_config,
            MagicMock(),
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
        handler = RustOptimizationHandler(
            {"calibration": {"max_steps": 20}},
            SimpleNamespace(ms2_error=100.0),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            create_extraction_handler=MagicMock(),
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


def _create_converged_handler(*, final_refit: bool):
    """Make a handler whose optimizers have converged, with a lock step that finds enough precursors."""
    optimizer = MagicMock(has_converged=True, parameter_name="ms2_error")
    optimization_manager = MagicMock(ms2_error=7.0)
    fdr_manager = MagicMock(current_version=7)
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        handler = RustOptimizationHandler(
            {"calibration": {"max_steps": 20, "final_refit": final_refit}},
            optimization_manager,
            MagicMock(),
            fdr_manager,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            create_extraction_handler=MagicMock(),
        )
    handler._get_ordered_optimizers = MagicMock(return_value=[[optimizer]])
    handler._process_batch = MagicMock(return_value=pd.DataFrame())
    filtered = pd.DataFrame({"precursor_idx": range(10)})
    handler._filter_dfs = MagicMock(return_value=(filtered, pd.DataFrame()))
    return handler, optimization_manager


def test_search_parameter_optimization_refits_with_final_tolerances():
    """Test that final_refit searches the lock once more and takes its calibration and classifier forward."""
    # given
    handler, optimization_manager = _create_converged_handler(final_refit=True)

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler_class:
        handler.search_parameter_optimization()

    # then
    handler._process_batch.assert_called_once()
    recalibration_handler_class.return_value.recalibrate.assert_called_once()
    optimization_manager.update.assert_called_once_with(classifier_version=7)


def test_search_parameter_optimization_without_final_refit_keeps_the_optimum():
    """Test that without final_refit the lock is not searched again after the optimizers converged."""
    # given
    handler, optimization_manager = _create_converged_handler(final_refit=False)

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler_class:
        handler.search_parameter_optimization()

    # then
    handler._process_batch.assert_not_called()
    recalibration_handler_class.return_value.recalibrate.assert_not_called()
    optimization_manager.update.assert_not_called()


def _create_fdr_handler(handler_class, *, prune_lock: bool):
    """Make an optimization handler whose FDR steps are mocks."""
    config = {
        "calibration": {
            "prune_lock": prune_lock,
            "prune_lock_qval": 0.2,
            "classifier_max_train_samples": 1000,
        },
        "fdr": {"competitive_scoring": True, "channel_wise_fdr": False},
    }
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        return handler_class(
            config,
            MagicMock(classifier_version=3),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            create_extraction_handler=MagicMock(),
        )


@pytest.mark.parametrize(
    ("prune_lock", "expected_max_qval"), [(False, None), (True, 0.2)]
)
def test_rust_optimization_fdr_keeps_the_qvals_to_prune_the_lock(
    prune_lock, expected_max_qval
):
    # Given: a Rust optimization handler with or without pruning of the lock
    handler = _create_fdr_handler(RustOptimizationHandler, prune_lock=prune_lock)
    extraction_handler = MagicMock()
    extraction_handler.perform_fdr_and_filter_candidates.return_value = (
        None,
        pd.DataFrame(),
    )

    # When: the handler performs FDR on a batch
    handler._perform_fdr(extraction_handler, MagicMock())

    # Then: the FDR step keeps the q-values up to the prune q-value and trains on at most the given rows
    kwargs = extraction_handler.perform_fdr_and_filter_candidates.call_args.kwargs
    assert kwargs["max_qval"] == expected_max_qval
    assert kwargs["max_train_samples"] == 1000


@pytest.mark.parametrize(
    ("prune_lock", "expected_fdr_heuristic"), [(False, 0.1), (True, 0.2)]
)
def test_python_optimization_fdr_keeps_the_qvals_to_prune_the_lock(
    prune_lock, expected_fdr_heuristic
):
    # Given: a Python optimization handler with or without pruning of the lock
    handler = _create_fdr_handler(PythonOptimizationHandler, prune_lock=prune_lock)

    # When: the handler performs FDR on a batch
    handler._perform_fdr(MagicMock(), MagicMock())

    # Then: the q-values up to the prune q-value enter fragment competition and the classifier trains on at most the given rows
    kwargs = handler._fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["fdr_heuristic"] == expected_fdr_heuristic
    assert kwargs["max_train_samples"] == 1000

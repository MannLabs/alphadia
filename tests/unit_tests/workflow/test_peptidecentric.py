"""Unit test for the peptidecentric module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphadia.constants.keys import SearchTolerance
from alphadia.constants.settings import CalibrationLoop
from alphadia.exceptions import NoOptimizationLockTargetError
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
    optimization_manager = SimpleNamespace(ms2_error=100.0, rt_error=800.0)
    optimization_manager.update = lambda **kwargs: optimization_manager.__dict__.update(
        kwargs
    )
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        handler = RustOptimizationHandler(
            {
                "search": {
                    "target_ms2_tolerance": 0.0,
                    "target_rt_tolerance": 0.0,
                },
                "calibration": {"max_steps": 20},
            },
            optimization_manager,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(has_ms1=False, has_mobility=False),
            create_extraction_handler=MagicMock(),
        )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = False
    handler._optlock.total_elution_groups = total_elution_groups
    handler._optlock.batch_plan = [(0, 1000)]
    handler._process_batch = MagicMock(return_value=pd.DataFrame())
    handler._filter_dfs = MagicMock(return_value=(pd.DataFrame(), pd.DataFrame()))
    return handler, optimization_manager


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


def test_search_parameter_optimization_keeps_the_initial_tolerances_with_too_few_precursors():
    """Test that a raw file with candidates but too few identifications to calibrate keeps the initial tolerances."""
    # Given: a raw file with candidates and an optimization lock without batches left before the target
    handler, optimization_manager = _create_handler_that_exhausts_the_batch_plan(
        total_elution_groups=2000
    )

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # Then: the handler does not calibrate and keeps the initial tolerances
    recalibration_handler.return_value.recalibrate.assert_not_called()
    assert optimization_manager.ms2_error == 100.0
    assert optimization_manager.rt_error == 800.0


def _identifications(n_identified: int) -> pd.DataFrame:
    return pd.DataFrame({"precursor_idx": range(n_identified)})


def _create_calibration_loop_handler(
    identifications_per_round,
    rt_residual_intervals,
    *,
    target_ms2=10.0,
    identifications_per_refit=(),
):
    """Make a handler whose searches find `identifications_per_round` identifications on a lock of 1000 elution
    groups. Its RT estimator gives one value of `rt_residual_intervals` per calibration. Refits of the classifier
    on the last search find `identifications_per_refit` identifications, then none."""
    config = {
        "search": {
            "target_ms2_tolerance": target_ms2,
            "target_rt_tolerance": 0.0,
            "target_ms1_tolerance": 5.0,
            "target_mobility_tolerance": 0.0,
        },
        "calibration": {
            "max_steps": 10,
        },
    }
    optimization_manager = SimpleNamespace(ms2_error=30.0, rt_error=800.0)
    optimization_manager.update = lambda **kwargs: optimization_manager.__dict__.update(
        kwargs
    )
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        handler = RustOptimizationHandler(
            config,
            optimization_manager,
            MagicMock(),
            MagicMock(current_version=0),
            MagicMock(),
            MagicMock(),
            MagicMock(has_ms1=False, has_mobility=False),
            create_extraction_handler=MagicMock(),
        )
    handler._optlock.has_target_num_precursors = True
    handler._optlock.total_elution_groups = 1000
    handler._optlock.start_idx, handler._optlock.stop_idx = 0, 1000
    remaining_identifications = iter(identifications_per_round)
    handler._process_batch = MagicMock(
        side_effect=lambda: _identifications(next(remaining_identifications))
    )
    remaining_refits = iter(identifications_per_refit)
    handler._refit_classifier = MagicMock(
        side_effect=lambda: _identifications(next(remaining_refits, 0))
    )
    handler._true_match_tolerance = MagicMock(return_value=None)
    handler._dia_data.rt_values = np.arange(0, 600, 0.5)
    handler._dia_data.cycle = np.zeros((1, 4, 1, 2))
    handler._filter_dfs = lambda precursor_df, _: (precursor_df, pd.DataFrame())
    remaining_rt_intervals = iter(rt_residual_intervals)
    handler._calibration_manager.get_estimator.return_value.ci.side_effect = (
        lambda df, quantile: next(remaining_rt_intervals)
    )
    return handler, optimization_manager


RT_TOLERANCE_FACTOR = CalibrationLoop.TOLERANCE_FACTOR[SearchTolerance.RT_ERROR]


def test_calibration_stops_when_tolerances_settle():
    """Test that the loop stops when no tolerance changes by more than the convergence threshold."""
    # Given: RT residual intervals of 100, 70 and 68 s
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 41],
        rt_residual_intervals=[100.0, 70.0, 68.0],
    )

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the third proposal changes by less than 10% and the handler keeps the second
    assert handler._process_batch.call_count == 3
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 70.0)
    assert optimization_manager.ms2_error == 10.0


def test_calibration_refits_the_classifier_until_identifications_stop_increasing():
    """Test that after the tolerances converge, the classifier is refit on the last search while it identifies more
    precursors, and the main search is calibrated on the identifications of the best refit."""
    # Given: tolerances that settle on the third round and refits that find 45, 50 and then 48 identifications
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 41],
        rt_residual_intervals=[100.0, 70.0, 68.0],
        identifications_per_refit=[45, 50, 48],
    )

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # Then: the third refit stops the refits and the last calibration uses the 50 identifications of the second
    assert handler._process_batch.call_count == 3
    assert handler._refit_classifier.call_count == 3
    last_call = recalibration_handler.return_value.recalibrate.call_args_list[-1]
    assert len(last_call.args[0]) == 50
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 70.0)


def test_calibration_keeps_its_calibration_when_a_refit_finds_no_more_identifications():
    """Test that a refit that does not identify more precursors leaves the calibration of the converged round."""
    # Given: tolerances that settle on the third round and a refit that finds fewer identifications
    handler, _ = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 41],
        rt_residual_intervals=[100.0, 70.0, 68.0],
        identifications_per_refit=[39],
    )

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # Then: the handler refits once and the last calibration uses the 41 identifications of the converged round
    assert handler._refit_classifier.call_count == 1
    last_call = recalibration_handler.return_value.recalibrate.call_args_list[-1]
    assert len(last_call.args[0]) == 41


def test_main_search_takes_the_true_match_interval_of_an_automatic_rt_tolerance():
    """Test that an automatic RT tolerance is set from the true matches of the last calibration round."""
    # Given: tolerances that settle on the third round, and true matches whose 99% RT interval is 55 s
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 41],
        rt_residual_intervals=[100.0, 70.0, 68.0],
    )
    handler._true_match_tolerance.return_value = 55.0

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the main search uses the true match interval for RT and the target for the targeted ms2 tolerance
    handler._true_match_tolerance.assert_called_once_with(SearchTolerance.RT_ERROR)
    assert optimization_manager.rt_error == 55.0
    assert optimization_manager.ms2_error == 10.0


def test_main_search_rt_tolerance_is_at_least_ten_cycles():
    """Test that the RT tolerance of the main search does not fall below the minimum number of DIA cycles."""
    # Given: true matches with a 99% RT interval of 5 s, and the handler's DIA cycle of 4 spectra 0.5 s apart (2 s)
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 41],
        rt_residual_intervals=[100.0, 70.0, 68.0],
    )
    handler._true_match_tolerance.return_value = 5.0

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the main search uses the minimum of 10 cycles
    assert optimization_manager.rt_error == pytest.approx(
        CalibrationLoop.MIN_RT_TOLERANCE_CYCLES * 2.0
    )


def test_main_search_keeps_the_rt_tolerance_without_a_true_match_interval():
    """Test that the RT tolerance of the loop is kept when the true matches give no interval."""
    # Given: tolerances that settle on the third round and too few true matches for an interval
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 41],
        rt_residual_intervals=[100.0, 70.0, 68.0],
    )

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the RT tolerance stays at the converged value of the loop
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 70.0)


def test_main_search_keeps_a_targeted_rt_tolerance():
    """Test that a targeted RT tolerance is not replaced by the true match interval."""
    # Given: an RT target of 150 s
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40],
        rt_residual_intervals=[100.0, 70.0],
    )
    handler._config["search"]["target_rt_tolerance"] = 150.0
    handler._true_match_tolerance.return_value = 55.0

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the main search uses the target
    handler._true_match_tolerance.assert_not_called()
    assert optimization_manager.rt_error == 150.0


def test_calibration_restores_the_best_checkpoint_on_identification_loss():
    """Test that a round that loses identifications ends the loop with the tolerances of the best checkpoint."""
    # Given: the third round loses 25% of the identifications of the second round
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 30],
        rt_residual_intervals=[100.0, 40.0, 40.0],
    )

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # Then: the handler restores the tolerance of the second round and calibrates on its 40 identifications
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 100.0)
    last_call = recalibration_handler.return_value.recalibrate.call_args_list[-1]
    assert len(last_call.args[0]) == 40


def test_calibration_does_not_compare_rounds_on_different_locks():
    """Test that fewer identifications on another lock do not restore an earlier checkpoint."""
    # Given: the lock shrinks to 500 elution groups after the second round and finds fewer identifications
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[], rt_residual_intervals=[100.0, 40.0, 40.0]
    )
    rounds = iter([(20, 1000), (40, 1000), (30, 500)])

    def next_round():
        n_identified, stop_idx = next(rounds)
        handler._optlock.stop_idx = stop_idx
        return _identifications(n_identified)

    handler._process_batch = MagicMock(side_effect=next_round)

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the handler restores no checkpoint and the loop converges at the smaller tolerance
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 40.0)


def test_calibration_extends_the_lock_until_the_target():
    """Test that the lock extends at the current tolerances until it holds the target number of identifications."""
    # Given: a lock below its target that reaches the target with the second batch
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[1, 20, 20], rt_residual_intervals=[40.0, 40.0]
    )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = True

    def reach_target():
        handler._optlock.has_target_num_precursors = True

    handler._optlock.update.side_effect = reach_target

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # Then: the first round only extends the lock and the calibration starts on the second round
    first_recalibration = recalibration_handler.return_value.recalibrate.call_args_list[
        0
    ]
    assert len(first_recalibration.args[0]) == 20
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 40.0)

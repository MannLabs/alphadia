"""Unit test for the peptidecentric module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphadia.constants.keys import SearchTolerance
from alphadia.constants.settings import ConvergeCalibration
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


def test_search_parameter_optimization_recovers_with_insufficient_precursors():
    """Test that a raw file with candidates but too few precursors to calibrate keeps the initial tolerances."""
    # given
    handler, optimization_manager = _create_handler_that_exhausts_the_batch_plan(
        total_elution_groups=2000
    )

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # then
    recalibration_handler.return_value.recalibrate.assert_not_called()
    assert optimization_manager.ms2_error == 100.0
    assert optimization_manager.rt_error == 800.0


def _create_converging_handler(rates, rt_intervals, *, target_ms2=10.0):
    """Make a handler whose rounds find `rates` identifications per elution group and whose RT estimator gives
    `rt_intervals` residual intervals, one per round."""
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
    round_rates = iter(rates)
    handler._process_batch = MagicMock(side_effect=lambda: next(round_rates))
    handler._filter_dfs = lambda rate, _: (
        pd.DataFrame({"precursor_idx": range(int(rate * 1000))}),
        pd.DataFrame(),
    )
    intervals = iter(rt_intervals)
    handler._calibration_manager.get_estimator.return_value.ci.side_effect = (
        lambda df, quantile: next(intervals)
    )
    return handler, optimization_manager


RT_FACTOR = ConvergeCalibration.TOLERANCE_FACTOR[SearchTolerance.RT_ERROR]


@pytest.fixture(autouse=True)
def min_calibration_precursors(monkeypatch):
    # keeps the identification counts of the converge tests small
    monkeypatch.setattr(ConvergeCalibration, "MIN_CALIBRATION_PRECURSORS", 10)


def test_converge_stops_when_tolerances_settle():
    """Test that the loop stops once no tolerance changes by more than the convergence threshold."""
    # given: RT residual intervals of 100, 70 and 68 s
    handler, optimization_manager = _create_converging_handler(
        rates=[0.02, 0.04, 0.041], rt_intervals=[100.0, 70.0, 68.0]
    )

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # then: the third proposal changes by less than 10 % and keeps the second
    assert handler._process_batch.call_count == 3
    assert optimization_manager.rt_error == pytest.approx(RT_FACTOR * 70.0)
    assert optimization_manager.ms2_error == 10.0


def test_converge_restores_best_round_on_id_loss():
    """Test that a round losing identifications ends the loop with the tolerances of the best round."""
    # given: the third round loses 25 % of the identifications of the second
    handler, optimization_manager = _create_converging_handler(
        rates=[0.02, 0.04, 0.03], rt_intervals=[100.0, 40.0, 40.0]
    )

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # then: the tolerance of the second round, recalibrated on its 40 identifications
    assert optimization_manager.rt_error == pytest.approx(RT_FACTOR * 100.0)
    last_call = recalibration_handler.return_value.recalibrate.call_args_list[-1]
    assert len(last_call.args[0]) == 40


def test_converge_does_not_compare_rounds_on_different_locks():
    """Test that fewer identifications on another lock do not restore an earlier round."""
    # given: the lock shrinks to 500 elution groups after the second round and finds fewer identifications
    handler, optimization_manager = _create_converging_handler(
        rates=[], rt_intervals=[100.0, 40.0, 40.0]
    )
    rounds = iter([(0.02, 1000), (0.04, 1000), (0.03, 500)])

    def next_round():
        rate, stop_idx = next(rounds)
        handler._optlock.stop_idx = stop_idx
        return rate

    handler._process_batch = MagicMock(side_effect=next_round)

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # then: no restore, the loop converges at the tightened tolerance
    assert optimization_manager.rt_error == pytest.approx(RT_FACTOR * 40.0)


def test_converge_grows_the_lock_until_it_can_calibrate():
    """Test that a lock with too few identifications to calibrate grows at the current tolerances."""
    # given: 1 identification in the first batch, 20 after the lock has grown
    handler, optimization_manager = _create_converging_handler(
        rates=[0.001, 0.02, 0.02], rt_intervals=[40.0, 40.0]
    )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = True

    def reach_target():
        handler._optlock.has_target_num_precursors = True

    handler._optlock.update.side_effect = reach_target

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ) as recalibration_handler:
        handler.search_parameter_optimization()

    # then: the first round only grows the lock, calibration starts on the second
    first_recalibration = recalibration_handler.return_value.recalibrate.call_args_list[
        0
    ]
    assert len(first_recalibration.args[0]) == 20
    assert optimization_manager.rt_error == pytest.approx(RT_FACTOR * 40.0)


def test_converge_calibrates_early_and_grows_the_lock():
    """Test that a lock below its target is recalibrated as soon as it holds enough identifications, and then grows
    with the calibrated tolerances."""
    # given: 15 identifications (enough to calibrate, below the target) and a lock that can grow
    handler, optimization_manager = _create_converging_handler(
        rates=[0.015, 0.03, 0.03], rt_intervals=[10.0, 60.0, 60.0]
    )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = True

    def reach_target(_):
        handler._optlock.has_target_num_precursors = True

    handler._optlock.grow.side_effect = reach_target

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # then: the early round shrinks RT by half at most (800 -> 400), the lock grows, and RT converges
    handler._optlock.grow.assert_called_once()
    assert optimization_manager.rt_error == pytest.approx(RT_FACTOR * 60.0)


def test_converge_calibrates_new_batches_after_an_early_calibration():
    """Test that a lock which falls below the calibration threshold after an early calibration grows with the
    calibrated columns, so that the next extraction finds them (regression: KeyError 'rt_calibrated')."""
    # given: 15 identifications (calibrate), then 5 (below the threshold of 10), then the target
    handler, _ = _create_converging_handler(
        rates=[0.015, 0.005, 0.03, 0.03], rt_intervals=[100.0] * 4
    )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = True
    calls = []

    def update():
        calls.append("update")
        handler._optlock.has_target_num_precursors = True

    handler._optlock.update.side_effect = update
    handler._optlock.update_with_calibration.side_effect = lambda _: calls.append(
        "calibrate"
    )
    handler._optlock.grow.side_effect = lambda _: calls.append("grow")

    # when
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # then: the batch added below the threshold is calibrated before it is extracted
    assert calls[:3] == ["grow", "update", "calibrate"]

"""Unit test for the peptidecentric module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphadia.constants.keys import SearchTolerance
from alphadia.constants.settings import CalibrationLoop
from alphadia.exceptions import NoOptimizationLockTargetError
from alphadia.workflow.peptidecentric.recalibration_handler import (
    RecalibrationHandler,
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


def test_search_parameter_optimization_keeps_the_initial_tolerances_without_a_checkpoint():
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
    identifications_per_round, rt_residual_intervals, *, target_ms2=10.0
):
    """Make a handler whose searches find `identifications_per_round` identifications on a lock of 1000 elution
    groups. Its RT estimator gives one value of `rt_residual_intervals` per calibration."""
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
    handler._filter_dfs = lambda precursor_df, _: (precursor_df, pd.DataFrame())
    remaining_rt_intervals = iter(rt_residual_intervals)
    handler._calibration_manager.get_estimator.return_value.ci.side_effect = (
        lambda df, quantile: next(remaining_rt_intervals)
    )
    return handler, optimization_manager


RT_TOLERANCE_FACTOR = CalibrationLoop.TOLERANCE_FACTOR[SearchTolerance.RT_ERROR]


@pytest.fixture(autouse=True)
def warm_up_min_precursors(monkeypatch):
    # Keeps the identification counts of the calibration loop tests small.
    monkeypatch.setattr(CalibrationLoop, "WARM_UP_MIN_PRECURSORS", 10)


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


def test_calibration_restores_the_best_checkpoint_on_identification_loss():
    """Test that a round that loses identifications ends the loop with the tolerances of the best checkpoint and the
    latest calibration."""
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

    # Then: the handler restores the tolerance of the second round and the last calibration used its 40 identifications
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 100.0)
    last_call = recalibration_handler.return_value.recalibrate.call_args_list[-1]
    assert len(last_call.args[0]) == 40


def test_identification_loss_is_measured_per_elution_group():
    """Test that fewer identifications on a smaller lock do not count as a loss when the identification rate holds."""
    # Given: the lock shrinks to 500 elution groups after the second round and finds 30 instead of 40 identifications
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[], rt_residual_intervals=[100.0, 40.0, 40.0]
    )
    rounds = iter([(20, 1000), (40, 1000), (30, 500)])

    def next_round():
        n_identified, n_elution_groups = next(rounds)
        handler._optlock.total_elution_groups = n_elution_groups
        return _identifications(n_identified)

    handler._process_batch = MagicMock(side_effect=next_round)

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the handler restores no checkpoint and the loop converges at the smaller tolerance
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 40.0)


def test_last_round_is_checked_before_the_loop_ends():
    """Test that the loop checks the round that searched with the last calibration for an identification loss."""
    # Given: two calibrations allowed, and the round that searched with the second calibration loses half the
    # identifications
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 40, 20], rt_residual_intervals=[100.0, 40.0]
    )
    handler._config["calibration"]["max_steps"] = 2

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the handler takes the tolerance of the best checkpoint and not the tolerance of the last calibration
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 100.0)


def test_unfitted_estimator_keeps_the_current_tolerance():
    """Test that a tolerance whose estimator could not be fitted keeps its value and does not become 0."""
    # Given: the RT estimator fails to fit in every round
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[20, 20], rt_residual_intervals=[None, None]
    )

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: RT keeps its initial tolerance and the handler still sets the targeted MS2 tolerance
    assert optimization_manager.rt_error == 800.0
    assert optimization_manager.ms2_error == 10.0


def test_expand_extends_the_lock_until_it_can_calibrate():
    """Test that the handler extends a lock with too few identifications to calibrate at the current tolerances."""
    # Given: 1 identification in the first batch and 20 after the lock extends
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[1, 20, 20], rt_residual_intervals=[40.0, 40.0]
    )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = True

    def reach_target(_):
        handler._optlock.has_target_num_precursors = True

    handler._optlock.extend.side_effect = reach_target

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


def test_warm_up_calibrates_and_grows_the_lock():
    """Test that the handler calibrates a lock below its target as soon as it holds enough identifications. The lock
    then grows with the calibrated tolerances."""
    # Given: 15 identifications (enough to calibrate, below the target) and a lock that can grow
    handler, optimization_manager = _create_calibration_loop_handler(
        identifications_per_round=[15, 30, 30], rt_residual_intervals=[10.0, 60.0, 60.0]
    )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = True

    def reach_target(_):
        handler._optlock.has_target_num_precursors = True

    handler._optlock.grow.side_effect = reach_target

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the warm-up round shrinks RT by at most half (800 -> 400), the lock grows and RT converges
    handler._optlock.grow.assert_called_once()
    assert optimization_manager.rt_error == pytest.approx(RT_TOLERANCE_FACTOR * 60.0)


def test_expand_after_warm_up_extends_with_the_calibration():
    """Test that a lock that goes back to expand after a warm-up round extends with the calibrated columns. The next
    extraction needs these columns (regression: KeyError 'rt_calibrated')."""
    # Given: 15 identifications (warm-up), then 5 (expand, below the threshold of 10), then the target
    handler, _ = _create_calibration_loop_handler(
        identifications_per_round=[15, 5, 30, 30], rt_residual_intervals=[100.0] * 4
    )
    handler._optlock.has_target_num_precursors = False
    handler._optlock.batches_remaining.return_value = True
    calls = []

    def extend(calibration_manager):
        calls.append(("extend", calibration_manager))
        handler._optlock.has_target_num_precursors = True

    handler._optlock.extend.side_effect = extend
    handler._optlock.grow.side_effect = lambda _: calls.append(("grow",))

    # When: the handler runs the calibration loop
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.RecalibrationHandler"
    ):
        handler.search_parameter_optimization()

    # Then: the batch that expand adds gets the calibration of the warm-up round
    assert calls[:2] == [("grow",), ("extend", handler._calibration_manager)]


def test_recalibrate_keeps_the_best_peak_group_only():
    """Test that a calibration keeps one peak group per precursor."""
    # given
    config = {
        "search": {"target_num_candidates": 3, "optimized_peak_group_score": False},
    }
    optimization_manager = SimpleNamespace(num_candidates=1)
    optimization_manager.update = lambda **kwargs: optimization_manager.__dict__.update(
        kwargs
    )
    handler = RecalibrationHandler(
        config, optimization_manager, MagicMock(), MagicMock(), None, False
    )
    precursor_df = pd.DataFrame(
        {"score": [1.0, 2.0], "fwhm_rt": [3.0, 3.0], "fwhm_mobility": [0.0, 0.0]}
    )

    # when
    handler.recalibrate(precursor_df, pd.DataFrame())

    # then
    assert optimization_manager.num_candidates == 1


def test_search_parameter_optimization_sets_the_number_of_candidates_for_the_main_search(
    mock_config,
):
    """Test that the main search scores target_num_candidates peak groups after a calibration with one."""
    # given
    config = {**mock_config, "search": {"target_num_candidates": 3}}
    workflow = RustPeptideCentricWorkflow("test_instance", config)
    workflow._optimization_manager = MagicMock()
    workflow._calibration_manager = MagicMock()
    workflow._fdr_manager = MagicMock()
    workflow._spectral_library = MagicMock()
    workflow._dia_data = MagicMock()
    workflow.reporter = MagicMock()

    # when
    with (
        patch.object(RustPeptideCentricWorkflow, "OPTIMIZATION_HANDLER_CLASS"),
        patch.object(RustPeptideCentricWorkflow, "_save_managers"),
    ):
        workflow.search_parameter_optimization()

    # then
    workflow.optimization_manager.update.assert_called_once_with(num_candidates=3)

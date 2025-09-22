"""Unit tests for the OptimizationLock class."""

from alphadia.workflow.optimizers.optimization_lock import OptimizationLock


def test_get_exponential_batch_plan_correctly():
    """Tests that the exponential batch plan is constructed correctly."""

    batch_plan = OptimizationLock._get_batch_plan(1000, 100)

    expected_plan = [(0, 100), (100, 300), (300, 700), (700, 1000)]
    assert batch_plan == expected_plan


def test_get_exponential_batch_plan_fixed_start_idx_correctly():
    """Tests that the exponential batch plan is set constructed with fixed start idx."""

    batch_plan = OptimizationLock._get_batch_plan(1000, 100, fixed_start_idx=True)

    expected_plan = [(0, 100), (0, 300), (0, 700), (0, 1000)]
    assert batch_plan == expected_plan

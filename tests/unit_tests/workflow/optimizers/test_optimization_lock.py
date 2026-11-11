"""Unit tests for the OptimizationLock class."""

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.workflow.optimizers.optimization_lock import OptimizationLock


def test_get_exponential_batch_plan_correctly():
    """Tests that the exponential batch plan is constructed correctly."""

    batch_plan = OptimizationLock._get_batch_plan(1000, 100)

    expected_plan = [(0, 100), (100, 300), (300, 700), (700, 1000)]
    assert batch_plan == expected_plan


def test_get_exponential_batch_plan_fixed_start_idx_correctly():
    """Tests that the exponential batch plan is constructed with fixed start idx."""

    batch_plan = OptimizationLock._get_batch_plan(1000, 100, fixed_start_idx=True)

    expected_plan = [(0, 100), (0, 300), (0, 700), (0, 1000)]
    assert batch_plan == expected_plan


NO_PRUNING = {
    "prune_lock": False,
    "prune_lock_qval": 0.2,
    "prune_lock_background_factor": 10,
    "prune_lock_after_steps": 0,
}


def _create_test_library(count=100000):
    lib = SpecLibFlat()
    precursor_idx = np.arange(count)
    elution_group_idx = np.concatenate(
        [np.full(2, i, dtype=int) for i in np.arange(len(precursor_idx) / 2)]
    )
    flat_frag_start_idx = precursor_idx * 10
    flat_frag_stop_idx = (precursor_idx + 1) * 10
    lib._precursor_df = pd.DataFrame(
        {
            "elution_group_idx": elution_group_idx,
            "precursor_idx": precursor_idx,
            "flat_frag_start_idx": flat_frag_start_idx,
            "flat_frag_stop_idx": flat_frag_stop_idx,
        }
    )

    lib._fragment_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(0, flat_frag_stop_idx[-1]),
        }
    )

    return lib


def _create_test_library_for_indexing():
    lib = SpecLibFlat()
    precursor_idx = np.arange(1000)
    elution_group_idx = np.concatenate(
        [np.full(2, i, dtype=int) for i in np.arange(len(precursor_idx) / 2)]
    )
    flat_frag_start_idx = precursor_idx**2
    flat_frag_stop_idx = (precursor_idx + 1) ** 2
    lib._precursor_df = pd.DataFrame(
        {
            "elution_group_idx": elution_group_idx,
            "precursor_idx": precursor_idx,
            "flat_frag_start_idx": flat_frag_start_idx,
            "flat_frag_stop_idx": flat_frag_stop_idx,
        }
    )

    lib._fragment_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(0, flat_frag_stop_idx[-1]),
        }
    )

    return lib


def test_optlock_spot_on_target():
    TEST_OPTLOCK_CONFIG = {
        "calibration": {
            "batch_size": 2000,
            "optimization_lock_target": 200,
            **NO_PRUNING,
        }
    }

    # edge case where the number of precursors is exactly the target
    library = _create_test_library(2000)
    optlock = OptimizationLock(library, TEST_OPTLOCK_CONFIG)

    assert optlock.start_idx == optlock.batch_plan[0][0]

    feature_df = pd.DataFrame({"elution_group_idx": np.arange(0, 1000)})
    fragment_df = pd.DataFrame({"elution_group_idx": np.arange(0, 10000)})

    optlock.update_with_extraction(feature_df, fragment_df)

    assert optlock.total_elution_groups == 1000
    precursor_df = pd.DataFrame(
        {
            "qval": np.concatenate([np.full(200, 0.005), np.full(800, 0.05)]),
            "decoy": np.zeros(1000),
        }
    )
    optlock.update_with_fdr(precursor_df)
    optlock.update()

    assert optlock.start_idx == 0
    assert optlock.stop_idx == optlock.batch_plan[0][1]
    assert optlock.has_target_num_precursors


TEST_OPTLOCK_CONFIG = {
    "calibration": {
        "batch_size": 8000,
        "optimization_lock_target": 200,
        **NO_PRUNING,
    }
}


def test_optlock():
    library = _create_test_library()
    optlock = OptimizationLock(library, TEST_OPTLOCK_CONFIG)

    assert optlock.start_idx == optlock.batch_plan[0][0]

    feature_df = pd.DataFrame({"elution_group_idx": np.arange(0, 1000)})
    fragment_df = pd.DataFrame({"elution_group_idx": np.arange(0, 10000)})

    optlock.update_with_extraction(feature_df, fragment_df)

    assert optlock.total_elution_groups == 1000
    precursor_df = pd.DataFrame(
        {
            "qval": np.concatenate([np.full(100, 0.005), np.full(1000, 0.05)]),
            "decoy": np.zeros(1100),
        }
    )
    optlock.update_with_fdr(precursor_df)

    assert not optlock.has_target_num_precursors
    assert not optlock.previously_calibrated
    optlock.update()

    assert optlock.start_idx == optlock.batch_plan[1][0]

    feature_df = pd.DataFrame({"elution_group_idx": np.arange(1000, 2000)})
    fragment_df = pd.DataFrame({"elution_group_idx": np.arange(10000, 20000)})

    optlock.update_with_extraction(feature_df, fragment_df)

    assert optlock.total_elution_groups == 2000

    precursor_df = pd.DataFrame(
        {
            "qval": np.concatenate([np.full(200, 0.005), np.full(1000, 0.05)]),
            "decoy": np.zeros(1200),
        }
    )

    optlock.update_with_fdr(precursor_df)

    assert optlock.has_target_num_precursors
    assert not optlock.previously_calibrated

    optlock.update()

    assert optlock.start_idx == 0

    assert optlock.total_elution_groups == 2000


def test_optlock_batch_idx():
    library = _create_test_library()
    optlock = OptimizationLock(library, TEST_OPTLOCK_CONFIG)

    optlock.batch_plan = [[0, 100], [100, 2000], [2000, 8000]]

    assert optlock.start_idx == 0

    optlock.update()
    assert optlock.start_idx == 100

    optlock.update()
    assert optlock.start_idx == 2000

    precursor_df = pd.DataFrame({"qval": np.full(4500, 0.005), "decoy": np.zeros(4500)})

    optlock.update_with_fdr(precursor_df)

    optlock.has_target_num_precursors = True
    optlock.update()

    assert optlock.start_idx == 0
    assert optlock.stop_idx == 2000


def test_optlock_reindex():
    library = _create_test_library_for_indexing()
    optlock = OptimizationLock(library, TEST_OPTLOCK_CONFIG)
    optlock.batch_plan = [[0, 100], [100, 200]]
    optlock.set_batch_dfs(
        optlock._elution_group_order[optlock.start_idx : optlock.stop_idx]
    )

    assert (
        (
            optlock.batch_library._precursor_df["flat_frag_stop_idx"].iloc[100]
            - optlock.batch_library._precursor_df["flat_frag_start_idx"].iloc[100]
        )
        == (
            (optlock.batch_library._precursor_df["precursor_idx"].iloc[100] + 1) ** 2
            - optlock.batch_library._precursor_df["precursor_idx"].iloc[100] ** 2
        )
    )  # Since each precursor was set (based on its original ID) to have a number of fragments equal to its original ID squared, the difference between the start and stop index should be equal to the original ID squared (even if the start and stop index have been changed to different values)
    assert (
        optlock.batch_library._fragment_df.iloc[
            optlock.batch_library._precursor_df.iloc[50]["flat_frag_start_idx"]
        ]["precursor_idx"]
        == optlock.batch_library._precursor_df.iloc[50]["precursor_idx"] ** 2
    )  # The original start index of any precursor should be equal to the square of the its original ID


def _pruning_optlock(
    background_factor: float, after_steps: int = 0
) -> OptimizationLock:
    """Optimization lock over 1000 elution groups whose first batch already reaches the target."""
    config = {
        "calibration": {
            "batch_size": 1000,
            "optimization_lock_target": 100,
            "prune_lock": True,
            "prune_lock_qval": 0.2,
            "prune_lock_background_factor": background_factor,
            "prune_lock_after_steps": after_steps,
        }
    }
    optlock = OptimizationLock(_create_test_library(2000), config)
    lock_elution_groups = optlock._elution_group_order[: optlock.stop_idx]
    optlock.update_with_extraction(
        pd.DataFrame({"elution_group_idx": lock_elution_groups}), pd.DataFrame()
    )
    return optlock


def test_optlock_prune_keeps_the_elution_groups_below_the_qval():
    """Test that pruning keeps the elution groups with a q-value below prune_lock_qval and removes the others."""
    # given
    optlock = _pruning_optlock(background_factor=0)
    lock_elution_groups = optlock._elution_group_order[: optlock.stop_idx]
    passing = lock_elution_groups[:300]
    precursor_df = pd.DataFrame(
        {
            "elution_group_idx": lock_elution_groups[:400],
            "qval": np.concatenate(
                [np.full(150, 0.005), np.full(150, 0.05), np.full(100, 0.5)]
            ),
            "decoy": np.zeros(400),
        }
    )

    # when
    optlock.update_with_fdr(precursor_df)
    optlock.update()

    # then
    assert optlock.has_target_num_precursors
    assert optlock.num_pruned_elution_groups == 700
    kept = optlock.batch_library._precursor_df["elution_group_idx"].unique()
    assert set(kept) == set(passing)
    # targets and decoys share the elution group, so both precursors of each kept elution group stay
    assert len(optlock.batch_library._precursor_df) == 2 * len(passing)


def test_optlock_prune_keeps_a_background_of_a_multiple_of_the_passing_elution_groups():
    """Test that pruning keeps background_factor times as many other elution groups as pass the q-value."""
    # given
    optlock = _pruning_optlock(background_factor=2)
    lock_elution_groups = optlock._elution_group_order[: optlock.stop_idx]
    passing = lock_elution_groups[500:600]
    precursor_df = pd.DataFrame(
        {
            "elution_group_idx": passing,
            "qval": np.full(100, 0.005),
            "decoy": np.zeros(100),
        }
    )

    # when
    optlock.update_with_fdr(precursor_df)
    optlock.update()

    # then
    assert optlock.num_pruned_elution_groups == 1000 - 300
    kept = set(optlock.batch_library._precursor_df["elution_group_idx"])
    assert kept == set(passing) | set(lock_elution_groups[:200])


def test_optlock_prune_counts_the_pruned_elution_groups_in_the_lock():
    """Test that the pruned elution groups count as lock elution groups without a precursor."""
    # given
    optlock = _pruning_optlock(background_factor=0)
    lock_elution_groups = optlock._elution_group_order[: optlock.stop_idx]
    optlock.update_with_fdr(
        pd.DataFrame(
            {
                "elution_group_idx": lock_elution_groups[:200],
                "qval": np.full(200, 0.005),
                "decoy": np.zeros(200),
            }
        )
    )
    optlock.update()

    # when
    optlock.update_with_extraction(
        pd.DataFrame({"elution_group_idx": lock_elution_groups[:200]}), pd.DataFrame()
    )

    # then
    assert optlock.total_elution_groups == 1000


def test_optlock_prunes_only_the_first_time_the_target_is_reached():
    """Test that the lock is not pruned again once the optimization started."""
    # given
    optlock = _pruning_optlock(background_factor=0)
    lock_elution_groups = optlock._elution_group_order[: optlock.stop_idx]
    precursor_df = pd.DataFrame(
        {
            "elution_group_idx": lock_elution_groups[:200],
            "qval": np.full(200, 0.005),
            "decoy": np.zeros(200),
        }
    )
    optlock.update_with_fdr(precursor_df)
    optlock.update()
    optlock.previously_calibrated = True

    # when
    optlock.update_with_fdr(precursor_df.iloc[:150])
    optlock.update()

    # then
    assert optlock.num_pruned_elution_groups == 800


def test_optlock_does_not_prune_if_disabled():
    """Test that reaching the target keeps all elution groups without prune_lock."""
    # given
    optlock = OptimizationLock(
        _create_test_library(2000),
        {
            "calibration": {
                "batch_size": 1000,
                "optimization_lock_target": 100,
                **NO_PRUNING,
            }
        },
    )
    precursor_df = pd.DataFrame({"qval": np.full(200, 0.005), "decoy": np.zeros(200)})

    # when
    optlock.update_with_fdr(precursor_df)
    optlock.update()

    # then
    assert optlock.num_pruned_elution_groups == 0
    assert (
        optlock.batch_library._precursor_df["elution_group_idx"].nunique()
        == optlock.stop_idx
    )


def test_optlock_prunes_after_the_given_number_of_optimization_steps():
    """Test that with prune_lock_after_steps the lock is pruned after that many steps on the full lock."""
    # given
    optlock = _pruning_optlock(background_factor=0, after_steps=2)
    lock_elution_groups = optlock._elution_group_order[: optlock.stop_idx]

    def fdr_round(num_passing):
        optlock.update_with_fdr(
            pd.DataFrame(
                {
                    "elution_group_idx": lock_elution_groups[:num_passing],
                    "qval": np.full(num_passing, 0.005),
                    "decoy": np.zeros(num_passing),
                }
            )
        )
        optlock.update()

    # when / then: the first time the target is reached (calibration only) and step 1 keep the full lock
    fdr_round(300)
    optlock.previously_calibrated = True
    assert optlock.num_pruned_elution_groups == 0
    fdr_round(300)
    assert optlock.num_pruned_elution_groups == 0
    assert not optlock.pruned_in_last_update

    # step 2 prunes with its own q-values
    fdr_round(250)
    assert optlock.pruned_in_last_update
    assert optlock.num_pruned_elution_groups == 750

    # later steps keep the pruned lock
    fdr_round(200)
    assert not optlock.pruned_in_last_update
    assert optlock.num_pruned_elution_groups == 750

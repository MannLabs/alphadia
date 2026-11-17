"""Unit tests for the OptimizationLock class."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.workflow.optimizers.optimization_lock import OptimizationLock


def test_get_exponential_batch_plan_correctly():
    """Tests that the exponential batch plan is constructed correctly."""

    batch_plan = OptimizationLock._get_batch_plan(1000, 100)

    expected_plan = [(0, 100), (100, 300), (300, 700), (700, 1000)]
    assert batch_plan == expected_plan


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
    optlock.resize(MagicMock())

    assert optlock.start_idx == 0
    assert optlock.stop_idx == optlock.batch_plan[0][1]
    assert optlock.has_target_num_precursors


TEST_OPTLOCK_CONFIG = {
    "calibration": {
        "batch_size": 8000,
        "optimization_lock_target": 200,
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
    optlock.extend()

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

    optlock.resize(MagicMock())

    assert optlock.start_idx == 0

    assert optlock.total_elution_groups == 2000


def test_optlock_batch_idx():
    library = _create_test_library()
    optlock = OptimizationLock(library, TEST_OPTLOCK_CONFIG)

    optlock.batch_plan = [[0, 100], [100, 2000], [2000, 8000]]

    assert optlock.start_idx == 0

    optlock.extend()
    assert optlock.start_idx == 100

    optlock.extend()
    assert optlock.start_idx == 2000

    precursor_df = pd.DataFrame({"qval": np.full(4500, 0.005), "decoy": np.zeros(4500)})

    optlock.update_with_fdr(precursor_df)

    optlock.has_target_num_precursors = True
    optlock.resize(MagicMock())

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


def test_optlock_grow_searches_the_whole_larger_lock_again():
    """Test that grow drops the extracted features and covers all elution groups from the first elution group to the
    end of the next batch, with the current calibration."""
    # Given: a lock with extracted features and fragments
    library = _create_test_library()
    optlock = OptimizationLock(library, TEST_OPTLOCK_CONFIG)
    optlock.update_with_extraction(
        pd.DataFrame({"elution_group_idx": np.arange(0, 1000)}),
        pd.DataFrame({"elution_group_idx": np.arange(0, 10000)}),
    )
    calibration_manager = MagicMock()

    # When: the lock grows
    optlock.grow(calibration_manager)

    # Then: the lock covers the next batch from its first elution group, without features, and calibrated
    assert optlock.batch_idx == 1
    expected_elution_groups = set(
        optlock._elution_group_order[: optlock.batch_plan[1][1]]
    )
    assert (
        set(optlock.batch_library._precursor_df["elution_group_idx"])
        == expected_elution_groups
    )
    assert optlock._feature_dfs == []
    assert optlock._fragment_dfs == []
    assert calibration_manager.predict.call_count == 2


PRUNE_OPTLOCK_CONFIG = {
    "calibration": {
        "batch_size": 1000,
        "optimization_lock_target": 200,
        "prune_lock": True,
        "prune_lock_qval": 0.1,
    }
}


def _reach_target(optlock: OptimizationLock, kept: np.ndarray) -> None:
    """Extract the first two batches and give the elution groups `kept` a q-value below 0.1, the others 0.5."""
    lock = optlock._elution_group_order[: optlock.batch_plan[1][1]]
    optlock.batch_idx = 1
    optlock.update_with_extraction(
        pd.DataFrame({"elution_group_idx": lock}), pd.DataFrame()
    )
    precursor_df = pd.DataFrame(
        {
            "elution_group_idx": lock,
            "qval": np.where(np.isin(lock, kept), 0.0, 0.5),
            "decoy": 0,
        }
    )
    optlock.update_with_fdr(precursor_df)
    optlock.record_prune_candidates(precursor_df)


def test_resize_prunes_the_elution_groups_above_the_prune_qval():
    """Test that the first resize keeps only the elution groups with a candidate below prune_lock_qval."""
    # Given: a lock that reached its target with 300 elution groups below the prune q-value
    optlock = OptimizationLock(_create_test_library(), PRUNE_OPTLOCK_CONFIG)
    kept = optlock._elution_group_order[:3000:10]
    _reach_target(optlock, kept)

    # When: the converge phase resizes the lock
    optlock.resize(MagicMock())

    # Then: the lock holds only the kept elution groups up to the resized stop, and pruning is done
    stop_idx = optlock.stop_idx
    expected = set(kept) & set(optlock._elution_group_order[:stop_idx])
    assert set(optlock.batch_library._precursor_df["elution_group_idx"]) == expected
    assert optlock.num_pruned_elution_groups == stop_idx - len(expected)
    assert optlock.prune_qval is None


def test_pruned_elution_groups_count_in_the_lock_size():
    """Test that the pruned elution groups count as extracted, so the identification rate compares with earlier rounds."""
    # Given: a pruned lock
    optlock = OptimizationLock(_create_test_library(), PRUNE_OPTLOCK_CONFIG)
    kept = optlock._elution_group_order[:3000:10]
    _reach_target(optlock, kept)
    optlock.resize(MagicMock())

    # When: the pruned lock is extracted
    optlock.update_with_extraction(
        pd.DataFrame(
            {
                "elution_group_idx": optlock.batch_library._precursor_df[
                    "elution_group_idx"
                ]
            }
        ),
        pd.DataFrame(),
    )

    # Then: the lock size is the whole resized lock
    assert optlock.total_elution_groups == optlock.stop_idx


def test_lock_is_pruned_only_once():
    """Test that a later resize or grow does not prune again and keeps the pruned elution groups out."""
    # Given: a pruned lock
    optlock = OptimizationLock(_create_test_library(), PRUNE_OPTLOCK_CONFIG)
    kept = optlock._elution_group_order[:3000:10]
    _reach_target(optlock, kept)
    optlock.resize(MagicMock())
    num_pruned = optlock.num_pruned_elution_groups

    # When: a later search records other q-values and the lock grows
    optlock.record_prune_candidates(pd.DataFrame({"elution_group_idx": [], "qval": []}))
    optlock.has_target_num_precursors = False
    optlock.grow(MagicMock())

    # Then: the earlier pruned elution groups stay out, the new batch is complete
    lock = optlock._elution_group_order[: optlock.stop_idx]
    pruned = set(lock[optlock._is_pruned[: optlock.stop_idx]])
    assert optlock.num_pruned_elution_groups == num_pruned
    assert not pruned & set(optlock.batch_library._precursor_df["elution_group_idx"])
    assert len(pruned) + optlock.batch_library._precursor_df[
        "elution_group_idx"
    ].nunique() == len(lock)


def test_resize_without_prune_lock_keeps_the_whole_lock():
    """Test that without prune_lock the resize keeps every elution group."""
    # Given: a lock without pruning that reached its target
    config = {"calibration": {"batch_size": 1000, "optimization_lock_target": 200}}
    optlock = OptimizationLock(_create_test_library(), config)
    _reach_target(optlock, optlock._elution_group_order[:3000:10])

    # When: the lock is resized
    optlock.resize(MagicMock())

    # Then: nothing is pruned
    assert optlock.num_pruned_elution_groups == 0
    assert set(optlock.batch_library._precursor_df["elution_group_idx"]) == set(
        optlock._elution_group_order[: optlock.stop_idx]
    )

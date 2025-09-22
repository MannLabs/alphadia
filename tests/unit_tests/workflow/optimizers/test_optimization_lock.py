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
    """Tests that the exponential batch plan is set constructed with fixed start idx."""

    batch_plan = OptimizationLock._get_batch_plan(1000, 100, fixed_start_idx=True)

    expected_plan = [(0, 100), (0, 300), (0, 700), (0, 1000)]
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
    optlock.update()

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

#!python -m unittest tests.test_utils
"""This module provides unit tests for alphadia.cli."""

# builtin

# local
# global
import numpy as np
import pandas as pd
import pytest

from alphadia.utils import (
    amean0,
    amean1,
    calculate_score_groups,
    merge_missing_columns,
    windows_to_wsl,
    wsl_to_windows,
)


def test_amean0():
    test_array = np.random.random((10, 10))

    numba_mean = amean0(test_array)
    np_mean = np.mean(test_array, axis=0)

    assert np.allclose(numba_mean, np_mean)


def test_amean1():
    test_array = np.random.random((10, 10))

    numba_mean = amean1(test_array)
    np_mean = np.mean(test_array, axis=1)

    assert np.allclose(numba_mean, np_mean)


def test_score_groups():
    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "channel": np.array([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]),
            "decoy": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        }
    )

    sample_df = calculate_score_groups(sample_df)

    assert np.allclose(sample_df["score_group_idx"].values, np.arange(10))

    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "channel": np.array([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]),
            "decoy": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        }
    )

    sample_df = calculate_score_groups(sample_df, group_channels=True)

    assert np.allclose(
        sample_df["score_group_idx"].values, np.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 3])
    )

    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "channel": np.array([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]),
            "decoy": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
            "rank": np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
        }
    )

    sample_df = calculate_score_groups(sample_df, group_channels=True)
    assert np.allclose(sample_df["score_group_idx"].values, np.arange(10))

    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            "channel": np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
            "decoy": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            "rank": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        }
    )

    sample_df = calculate_score_groups(sample_df, group_channels=True)

    assert np.allclose(
        sample_df["score_group_idx"].values, np.array([0, 0, 1, 1, 2, 3, 4, 4, 5, 5])
    )


def test_wsl_conversion():
    test_path = "/mnt/c/Users/username/Documents/test.txt"
    expected_path = "C:\\Users\\username\\Documents\\test.txt"

    assert wsl_to_windows(test_path) == expected_path
    assert windows_to_wsl(expected_path) == test_path

    test_path = "/mnt/d/Users/us__.sdername/D ocuments/test.txt"
    expected_path = "D:\\Users\\us__.sdername\\D ocuments\\test.txt"

    assert wsl_to_windows(test_path) == expected_path
    assert windows_to_wsl(expected_path) == test_path


@pytest.fixture()
def left_and_right_df():
    left_df = pd.DataFrame([{"idx": 1, "col_1": 0, "col_2": 0}])

    right_df = pd.DataFrame([{"idx": 1, "col_3": 0, "col_4": 0}])
    return left_df, right_df


def test_merge_missing_fail_on(left_and_right_df):
    # given:
    left_df, right_df = left_and_right_df

    # when, then
    with pytest.raises(ValueError):
        merge_missing_columns(left_df, right_df, ["col_3"], on="idx_doesnt_exist")


def test_merge_missing_fail_right(left_and_right_df):
    # given:
    left_df, right_df = left_and_right_df

    # when, then
    with pytest.raises(ValueError):
        merge_missing_columns(left_df, right_df, ["col_5"], on="idx")


def test_merge_missing(left_and_right_df):
    # given
    left_df, right_df = left_and_right_df
    # when
    df = merge_missing_columns(left_df, right_df, ["col_3"], on="idx")
    # then
    assert np.all(df.columns == ["idx", "col_1", "col_2", "col_3"])

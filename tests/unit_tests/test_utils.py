#!python -m unittest tests.test_utils
"""This module provides unit tests for alphadia.cli."""

# builtin
import unittest

# local
from alphadia.utils import (
    amean0,
    amean1,
    calculate_score_groups,
    wsl_to_windows,
    windows_to_wsl,
)


# global
import numpy as np
import pandas as pd


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

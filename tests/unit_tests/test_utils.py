#!python -m unittest tests.test_utils
"""This module provides unit tests for alphadia.cli."""

# builtin
import unittest

# local
from alphadia.extraction.utils import (
        join_left, 
        amean0, 
        amean1, 
        calc_isotopes_center, 
        calculate_score_groups
    )



# global
import numpy as np
import pandas as pd

def test_join_left():

    # right array in order
    left = np.random.randint(0,10,20)
    right = np.arange(0,10)
    joined = join_left(left, right)

    assert all(left==joined)

    # right array unordered
    left = np.random.randint(0,10,20)
    right = np.arange(10,-1,-1)
    joined = join_left(left, right)

    assert all(left==(10-joined))

    # no elements found in right array
    left = np.random.randint(0,10,20)
    right = np.arange(10,20)
    joined = join_left(left, right)
    assert all(joined == -1)

    # left array empty
    left = np.array([])
    right = np.arange(10,20)
    joined = join_left(left, right)
    assert len(joined)==0

    # same element appears multiple times in right array
    left = np.random.randint(0,10,20)
    right = np.ones(10)
    joined = join_left(left, right)
    assert all(joined[joined > -1] == 9)

def test_amean0():
    test_array = np.random.random((10,10))

    numba_mean = amean0(test_array)
    np_mean = np.mean(test_array, axis=0)

    assert np.allclose(numba_mean, np_mean)

def test_amean1():
    test_array = np.random.random((10,10))

    numba_mean = amean1(test_array)
    np_mean = np.mean(test_array, axis=1)

    assert np.allclose(numba_mean, np_mean)

def test_calc_isotopes_center():

    mz = 500
    charge = 2
    num_isotopes = 3

    isotopes = calc_isotopes_center(mz, charge, num_isotopes)
    assert np.allclose(isotopes, np.array([500., 500.50165, 501.0033 ]))

    isotopes = calc_isotopes_center(mz, charge, 0)
    assert np.allclose(isotopes, np.empty(0))

    # charge 0 should result in charge 1
    isotopes = calc_isotopes_center(mz, 0, 1)
    assert np.allclose(isotopes, np.array([500.]))

def test_score_groups():

    sample_df = pd.DataFrame({
        'precursor_idx': np.arange(10),
        'elution_group_idx': np.array([0,0,0,0,0,1,1,1,1,1]),
        'channel': np.array([0,1,2,3,0,0,1,2,3,0]),
        'decoy' : np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    })

    sample_df = calculate_score_groups(sample_df)

    assert np.allclose(sample_df['score_group_idx'].values, np.arange(10))

    sample_df = pd.DataFrame({
        'precursor_idx': np.arange(10),
        'elution_group_idx': np.array([0,0,0,0,0,1,1,1,1,1]),
        'channel': np.array([0,1,2,3,0,0,1,2,3,0]),
        'decoy' : np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    })

    sample_df = calculate_score_groups(sample_df, group_channels=True)

    assert np.allclose(sample_df['score_group_idx'].values, np.array([0,0,0,0,1,2,2,2,2,3]))

    sample_df = pd.DataFrame({
        'precursor_idx': np.arange(10),
        'elution_group_idx': np.array([0,0,0,0,0,1,1,1,1,1]),
        'channel': np.array([0,1,2,3,0,0,1,2,3,0]),
        'decoy' : np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        'rank' : np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    })

    sample_df = calculate_score_groups(sample_df, group_channels=True)
    assert np.allclose(sample_df['score_group_idx'].values, np.arange(10))

    sample_df = pd.DataFrame({
        'precursor_idx': np.arange(10),
        'elution_group_idx': np.array([0,0,0,0,1,1,1,1,0,0]),
        'channel': np.array([0,0,1,1,0,0,1,1,0,0]),
        'decoy' : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
        'rank' : np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    })

    sample_df = calculate_score_groups(sample_df, group_channels=True)

    assert np.allclose(sample_df['score_group_idx'].values, np.array([0,0,1,1,2,3,4,4,5,5]))

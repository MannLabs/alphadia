#!python -m unittest tests.test_utils
"""This module provides unit tests for alphadia.cli."""

# builtin
import unittest

# local
from alphadia.extraction.utils import (
        join_left, amean0, amean1, calculate_correlations, calc_isotopes_center
    )

# global
import numpy as np

class TestUtilMethods(unittest.TestCase):

    def test_join_left(self):

        # right array in order
        left = np.random.randint(0,10,20)
        right = np.arange(0,10)
        joined = join_left(left, right)

        self.assertTrue(all(left==joined))

        # right array unordered
        left = np.random.randint(0,10,20)
        right = np.arange(10,-1,-1)
        joined = join_left(left, right)

        self.assertTrue(all(left==(10-joined)))

        # no elements found in right array
        left = np.random.randint(0,10,20)
        right = np.arange(10,20)
        joined = join_left(left, right)
        self.assertTrue(all(joined == -1))

        # left array empty
        left = np.array([])
        right = np.arange(10,20)
        joined = join_left(left, right)
        self.assertTrue(len(joined)==0)

        # same element appears multiple times in right array
        left = np.random.randint(0,10,20)
        right = np.ones(10)
        joined = join_left(left, right)
        self.assertTrue(all(joined[joined > -1] == 9))

    def test_amean0(self):
        test_array = np.random.random((10,10))

        numba_mean = amean0(test_array)
        np_mean = np.mean(test_array, axis=0)

        self.assertTrue(np.allclose(numba_mean, np_mean))

    def test_amean1(self):
        test_array = np.random.random((10,10))

        numba_mean = amean1(test_array)
        np_mean = np.mean(test_array, axis=1)

        self.assertTrue(np.allclose(numba_mean, np_mean))

    # with version 0.3 smoothin is performed inside the correlation function
    # this should be changed in the future and the correlation function should use the smoothed or raw profiles as provided
    
    #def test_calculate_correlations(self):

        #test_precursors = np.zeros((10, 10))
        #test_precursors[4:6, 4:6] = 1

        #test_fragments = np.zeros((5, 10, 10))
        #test_fragments[0, 4:6, 4:6] = 1
        #test_fragments[1, 4:6, 4:6] = -1
        #test_fragments[2, 4:6, 4:6] = 1
        #test_fragments[3, 4:6, 4:6] = -1
        #test_fragments[4] = np.random.random((10, 10))

        #correlation = calculate_correlations(test_precursors, test_fragments)

        """
        [[-0.14488292 -0.25511708 -0.14488292 -0.25511708 0.        ]
        [-0.21473459 -0.18526541 -0.21473459 -0.18526541  0.        ]
        [ 1.         -1.          1.         -1.          0.27558542]
        [ 1.         -1.          1.         -1.         -0.07367296]]
        
        """

        #self.assertTrue(np.allclose(correlation[2,:-1], np.array([1, -1, 1, -1])))
        #self.assertTrue(np.allclose(correlation[3,:-1], np.array([1, -1, 1, -1])))

    def test_calc_isotopes_center(self):

        mz = 500
        charge = 2
        num_isotopes = 3

        isotopes = calc_isotopes_center(mz, charge, num_isotopes)
        self.assertTrue(np.allclose(isotopes, np.array([500., 500.50165, 501.0033 ])))

        isotopes = calc_isotopes_center(mz, charge, 0)
        self.assertTrue(np.allclose(isotopes, np.empty(0)))

        # charge 0 should result in charge 1
        isotopes = calc_isotopes_center(mz, 0, 1)
        self.assertTrue(np.allclose(isotopes, np.array([500.])))


if __name__ == "__main__":
    unittest.main()
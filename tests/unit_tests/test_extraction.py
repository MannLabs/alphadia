#!python -m unittest tests.test_utils
"""This module provides unit tests for alphadia.cli."""

# builtin
import unittest

# local
from alphadia.extraction.utils import (
        join_left, amean0, amean1, calculate_correlations, calc_isotopes_center
    )

from alphadia.extraction.quadrupole import SimpleQuadrupole, logistic_rectangle, quadrupole_transfer_function

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

class TestQuadrupoleCalibration(unittest.TestCase):

    def setUp(self):
        fake_cycle = np.array([[25.,55.],[15,40]])
        fake_cycle = np.repeat(fake_cycle[:,np.newaxis, :], 10, axis=1)[np.newaxis, :, :, :]

        self.q = SimpleQuadrupole(fake_cycle)

    def a_fit_quadrupole(self):
        mz_train1 = np.concatenate([np.random.uniform(20,30,20), np.random.uniform(45,65,20)])
        precursor_train1 = np.zeros(40, dtype=np.int64)
        int_train1 = logistic_rectangle(25,60, 0.5, 0.5, mz_train1) + np.random.normal(0, 0.05, 40)
        scan_train1 = np.random.randint(0,4, 40)

        mz_train2 = np.concatenate([np.random.uniform(10,20,20), np.random.uniform(35,45,20)])
        precursor_train2 = np.ones(40, dtype=np.int64)
        int_train2 = logistic_rectangle(15,45, 0.5, 0.5, mz_train2)+ np.random.normal(0, 0.05, 40)
        scan_train2 = np.random.randint(0,4, 40)

        mz_train = np.concatenate([mz_train1, mz_train2])
        precursor_train = np.concatenate([precursor_train1, precursor_train2])
        int_train = np.concatenate([int_train1, int_train2])
        scan_train = np.concatenate([scan_train1, scan_train2])

        self.q.fit(precursor_train, scan_train, mz_train, int_train)


    def b_predict(self):

        test_mz = np.linspace(0,70,1000)

        for precursor in [0,1]:
            test_scan = np.zeros(1000, dtype=np.int64)
            test_precursor = np.ones(1000, dtype=np.int64) * precursor

            intensity = self.q.jit.predict(test_precursor, test_scan, test_mz)

    def c_qtf(self):

        fake_cycle = np.array([[780.,801],[801,820]])
        fake_cycle = np.repeat(fake_cycle[:,np.newaxis, :], 10, axis=1)[np.newaxis, :, :, :]
                
        quad = SimpleQuadrupole(fake_cycle)

        isotope_mz = np.array([[800., 800.1, 800.2],[802.42944, 802.9311, 803.1]])
        observation_indices = np.array([0,1])
        scan_indices = np.array(np.arange(2,9))


        qtf = quadrupole_transfer_function(
            quad.jit,
            observation_indices,
            scan_indices,
            isotope_mz
        )

        self.assertTrue(qtf.shape == (2, 3, 2, 7))
        self.assertTrue(np.all(qtf[0,:,0,:] > 0.9))
        self.assertTrue(np.all(qtf[0,:,1,:] < 0.1))
        self.assertTrue(np.all(qtf[1,:,0,:] < 0.1))
        self.assertTrue(np.all(qtf[1,:,1,:] > 0.9))
    

if __name__ == "__main__":
    test_classes_to_run = [TestUtilMethods, TestQuadrupoleCalibration]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
from alphadia.extraction.data import (
    transpose
)
import numpy as np

def test_transpose():
    values = np.array([1., 2., 3., 4., 5., 6., 7.])
    tof_indices = np.array([0, 3, 2, 4 ,1, 2, 4])
    push_ptr = np.array([0, 2, 4, 5, 7])

    push_indices, tof_indptr, intensity_values = transpose(tof_indices, push_ptr, values)

    _push_indices = np.array([0, 2, 1, 3, 0, 1, 3])
    _tof_indptr = np.array([0, 1, 2, 4, 5, 7])
    _intensity_values = np.array([1., 5., 3., 6., 2., 4., 7.])

    assert np.allclose(push_indices, _push_indices)
    assert np.allclose(tof_indptr, _tof_indptr)
    assert np.allclose(intensity_values, _intensity_values)
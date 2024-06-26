import numpy as np

from alphadia.numba.numeric import search_sorted_left


def test_search_sorted_left():
    test_array = np.arange(100)
    value = 50
    assert search_sorted_left(test_array, value) == 50

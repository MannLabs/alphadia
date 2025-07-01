import numpy as np

from alphadia.data.alpharaw_wrapper import _search_sorted_left


def test_search_sorted_left():
    test_array = np.arange(100)
    value = 50
    assert _search_sorted_left(test_array, value) == 50

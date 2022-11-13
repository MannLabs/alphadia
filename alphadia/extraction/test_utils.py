from .utils import *
import numpy as np
def test_join_left():

    # right array in order
    left = np.random.randint(0,10,20)
    right = np.arange(0,10)
    joined = alphadia.extraction.utils.join_left(left, right)

    assert all(left==joined)

    # right array unordered
    left = np.random.randint(0,10,20)
    right = np.arange(10,-1,-1)
    joined = alphadia.extraction.utils.join_left(left, right)

    assert all(left==(10-joined))

    # no elements found in right array
    left = np.random.randint(0,10,20)
    right = np.arange(10,20)
    joined = alphadia.extraction.utils.join_left(left, right)
    assert(all(joined == -1))

    # left array empty
    left = np.array([])
    right = np.arange(10,20)
    joined = alphadia.extraction.utils.join_left(left, right)
    assert(len(joined)==0)

    # same element appears multiple times in right array
    left = np.random.randint(0,10,20)
    right = np.ones(10)
    joined = alphadia.extraction.utils.join_left(left, right)
    assert(all(joined[joined > -1] == 9))
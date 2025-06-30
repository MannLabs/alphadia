import numpy as np

# local
from alphadia.numba.numeric import (
    fragment_correlation,
    fragment_correlation_different,
    save_corrcoeff,
)


def test_save_corrcoeff():
    p = save_corrcoeff(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
        np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32),
    )
    assert np.isclose(p, -1.0)

    p = save_corrcoeff(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
    )
    assert np.isclose(p, 1.0)

    p = save_corrcoeff(
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
    )
    assert np.isclose(p, 0.0)


def test_fragment_correlation():
    a = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[3, 2, 1], [1, 2, 3]], [[0, 0, 0], [0, 0, 0]]]
    )
    corr = fragment_correlation(a)
    assert corr.shape == (2, 3, 3)

    test_a = np.array(
        [
            [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert np.allclose(corr, test_a)

    b = np.zeros((10, 10, 10))
    corr = fragment_correlation(b)
    assert corr.shape == (10, 10, 10)
    assert np.allclose(corr, b)


def test_fragment_correlation_different():
    a = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[3, 2, 1], [1, 2, 3]], [[0, 0, 0], [0, 0, 0]]]
    )
    corr = fragment_correlation_different(a, a)
    assert corr.shape == (2, 3, 3)

    test_a = np.array(
        [
            [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert np.allclose(corr, test_a)

    b = np.zeros((10, 10, 10))
    corr = fragment_correlation_different(b, b)
    assert corr.shape == (10, 10, 10)
    assert np.allclose(corr, b)

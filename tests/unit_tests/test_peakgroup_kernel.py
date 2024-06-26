import numpy as np
import pytest

from alphadia.peakgroup.kernel import GaussianKernel, multivariate_normal


def test_multivariate_normal():
    x1_values = np.random.uniform(0, 100, size=100).astype(np.float32)
    x2_values = np.random.uniform(0, 100, size=100).astype(np.float32)

    x = np.stack([x1_values, x2_values], axis=1)

    mu = np.mean(x, axis=0, keepdims=True)
    cov = np.array([[1000, 0], [0, 1000]], dtype=np.float32)

    y = multivariate_normal(x, mu, cov)

    assert y.ndim == 1
    assert y.shape[0] == 100

    assert np.all(y >= 0)

    max_idx = np.argmin(np.sum((x - mu) ** 2, axis=1))
    assert max_idx == np.argmax(y)


class FakeDiaData:
    def __init__(self, has_mobility, rt_values, mobility_values, cycle):
        self.has_mobility = has_mobility
        self.rt_values = rt_values
        self.mobility_values = mobility_values
        self.cycle = cycle


@pytest.mark.parametrize(
    "has_mobility, kernel_height, kernel_width, cycle_shape",
    [
        (False, 11, 11, (1, 4, 1, 2)),
        (True, 11, 11, (1, 4, 1, 2)),
        (False, 10, 10, (1, 4, 1, 2)),
        (True, 10, 10, (1, 4, 1, 2)),
        (False, 11, 11, (1, 4, 10, 2)),
        (True, 11, 11, (1, 4, 10, 2)),
        (False, 10, 10, (1, 4, 10, 2)),
        (True, 10, 10, (1, 4, 10, 2)),
    ],
)
def test_gaussian_kernel(has_mobility, kernel_height, kernel_width, cycle_shape):
    kernel = GaussianKernel(
        FakeDiaData(
            has_mobility,
            np.arange(0, 1000, 1).astype(np.float32),
            np.linspace(0.1, 0, 10).astype(np.float32),
            cycle=np.ones(cycle_shape, dtype=np.float32),
        ),
        kernel_height=kernel_height,
        kernel_width=kernel_width,
    )

    mat = kernel.get_dense_matrix()

    # assert all dims even
    assert all([i % 2 == 0 for i in mat.shape])
    # assert all values non nan
    assert np.all(~np.isnan(mat))

    assert mat.dtype == np.float32

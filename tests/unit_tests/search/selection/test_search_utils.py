from unittest import skip

import numpy as np

from alphadia.search.selection.fft import convolve_fourier
from alphadia.search.selection.kernel import multivariate_normal
from alphadia.search.selection.utils import _symetric_limits_1d, symetric_limits_2d


def test_symetric_limits_1d():
    # test both the numba and the python version
    for f in [_symetric_limits_1d, _symetric_limits_1d.py_func]:
        for _ in range(1000):
            x = np.random.random(int(np.random.random() * 20))
            center = int(np.random.random() * 20)
            f = np.random.random() * 2
            center_fraction = np.random.random()
            min_size = np.random.randint(0, 20)
            max_size = np.random.randint(0, 20)

            limits = _symetric_limits_1d(
                x,
                center,
                f=f,
                center_fraction=center_fraction,
                min_size=min_size,
                max_size=max_size,
            )

            assert limits[0] <= limits[1]
            assert limits[0] >= 0

            assert limits[0] <= center
            assert limits[1] >= center


@skip("raises NumbaContextOnly")
def test_symetric_limits_2d():
    for f in [symetric_limits_2d, symetric_limits_2d.py_func]:
        for _ in range(1000):
            size = np.random.randint(10, 20)
            sigma_x, sigma_y = np.random.random(2) * 10

            x, y = np.meshgrid(
                np.arange(-size // 2, size // 2), np.arange(-size // 2, size // 2)
            )
            xy = np.column_stack((x.flatten(), y.flatten())).astype("float32")

            # mean is always zero
            mu = np.array([[0.0, 0.0]])

            # sigma is set with no covariance
            sigma_mat = np.array([[sigma_x, 0.0], [0.0, sigma_y]])

            kernel = multivariate_normal(xy, mu, sigma_mat).reshape(size, size)

            dense = np.random.random((100, 100))
            peak_scan = np.random.randint(0, 100)
            peak_dia_cycle = np.random.randint(0, 100)

            dense[peak_scan, peak_dia_cycle] = 5

            min_size = np.random.randint(0, 20)

            score = convolve_fourier(dense, kernel)

            scan_limit, cycle_limit = f(
                score,
                peak_scan,
                peak_dia_cycle,
                f_mobility=0.99,
                f_rt=0.99,
                # center_fraction = 0.0005,
                min_size_mobility=min_size,
                min_size_rt=min_size,
                max_size_mobility=500,
                max_size_rt=500,
            )

            assert scan_limit[0] <= scan_limit[1]
            assert cycle_limit[0] <= cycle_limit[1]
            assert scan_limit[0] >= 0
            assert cycle_limit[0] >= 0
            assert scan_limit[1] <= score.shape[0]
            assert cycle_limit[1] <= score.shape[1]
            assert scan_limit[0] <= peak_scan
            assert scan_limit[1] >= peak_scan
            assert cycle_limit[0] <= peak_dia_cycle
            assert cycle_limit[1] >= peak_dia_cycle

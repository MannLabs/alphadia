import numpy as np

# local
from alphadia import utils
from alphadia.numba.fragments import get_ion_group_mapping

from alphadia.numba.fft import convolve_fourier

from alphadia.numba.numeric import (
    symetric_limits_1d,
    symetric_limits_2d,
    save_corrcoeff,
    fragment_correlation,
    fragment_correlation_different,
)


def test_get_ion_group_mapping():
    """Test the get_ion_group_mapping function."""

    ion_mz = np.array(
        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    )
    ion_count = np.ones(len(ion_mz), dtype=np.uint8)
    ion_precursor = np.tile(np.arange(2, dtype=np.uint8), 5)
    ion_intensity = np.random.rand(len(ion_mz))
    ion_cardinality = np.ones(len(ion_mz))

    precursor_abundance = np.array([1, 1])

    print(ion_mz, len(ion_mz))
    print(ion_count, len(ion_count))
    print(ion_precursor, len(ion_precursor))
    print(precursor_abundance)

    mz, intensity = get_ion_group_mapping(
        ion_precursor, ion_mz, ion_intensity, ion_cardinality, precursor_abundance
    )

    print(mz, mz.shape)
    print(intensity, intensity.shape)

    assert np.allclose(mz, ion_mz)
    assert np.allclose(np.ceil(intensity), np.ones((1, 10)))
    assert np.all(intensity.shape == (10,))

    ion_mz = np.repeat(np.array([100, 200.0]), 5)

    mz, intensity = get_ion_group_mapping(
        ion_precursor, ion_mz, ion_intensity, ion_cardinality, precursor_abundance
    )

    assert np.all(intensity.shape == (2,))
    assert np.all(mz.shape == (2,))


def fuzz_symetric_limits_1d():
    # test both the numba and the python version
    for f in [symetric_limits_1d, symetric_limits_1d.py_func]:
        for _ in range(1000):
            x = np.random.random(int(np.random.random() * 20))
            center = int(np.random.random() * 20)
            f = np.random.random() * 2
            center_fraction = np.random.random()
            min_size = np.random.randint(0, 20)
            max_size = np.random.randint(0, 20)

            limits = symetric_limits_1d(
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


def fuzz_symetric_limits_2d():
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

            kernel = utils.multivariate_normal(xy, mu, sigma_mat).reshape(size, size)

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

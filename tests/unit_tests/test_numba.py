
import numpy as np

# local
from alphadia.extraction import utils
from alphadia.extraction.numba.fragments import (
        get_ion_group_mapping
    )

from alphadia.extraction.numba.numeric import (
        transpose, symetric_limits_1d, symetric_limits_2d, convolve_fourier
    )

def test_get_ion_group_mapping():
    """Test the get_ion_group_mapping function."""

    ion_mz = np.array([100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.])
    ion_count = np.ones(len(ion_mz), dtype=np.uint8)
    ion_precursor = np.tile(np.arange(2, dtype=np.uint8),5)
    ion_intensity = np.random.rand(len(ion_mz))
    ion_cardinality = np.ones(len(ion_mz))

    precursor_abundance = np.array([1, 1])

    print(ion_mz, len(ion_mz))
    print(ion_count, len(ion_count))
    print(ion_precursor, len(ion_precursor))
    print(precursor_abundance)

    mz, intensity = get_ion_group_mapping(
        ion_precursor,
        ion_mz,
        ion_intensity,
        ion_cardinality,
        precursor_abundance
    )

    print(mz, mz.shape)
    print(intensity, intensity.shape)

    assert np.allclose(mz, ion_mz)
    assert np.allclose(np.ceil(intensity), np.ones((1, 10)))
    assert np.all(intensity.shape == (10, ))

    ion_mz = np.repeat(np.array([100, 200.]), 5)

    mz, intensity = get_ion_group_mapping(
        ion_precursor,
        ion_mz,
        ion_intensity,
        ion_cardinality,
        precursor_abundance
    )

    assert np.all(intensity.shape == (2, ))
    assert np.all(mz.shape == (2, ))


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

def fuzz_symetric_limits_1d():
    
    # test both the numba and the python version
    for f in [symetric_limits_1d, symetric_limits_1d.py_func]:
        for i in range(1000):
            x = np.random.random((int(np.random.random()*20)))
            center = int(np.random.random()*20)
            f = np.random.random()*2
            center_fraction = np.random.random()
            min_size = np.random.randint(0, 20)
            max_size = np.random.randint(0, 20)

            limits = symetric_limits_1d(x, center, f=f, center_fraction=center_fraction, min_size=min_size, max_size=max_size)

            assert(limits[0] <= limits[1])
            assert(limits[0] >= 0)

            assert(limits[0] <= center)
            assert(limits[1] >= center)

def fuzz_symetric_limits_2d():
    for f in [symetric_limits_2d, symetric_limits_2d.py_func]:
        for i in range(1000):
            size = np.random.randint(10, 20)
            sigma_x, sigma_y = np.random.random(2)*10
            
            x, y = np.meshgrid(np.arange(-size//2,size//2),np.arange(-size//2,size//2))
            xy = np.column_stack((x.flatten(), y.flatten())).astype('float32')

            # mean is always zero
            mu = np.array([[0., 0.]])

            # sigma is set with no covariance
            sigma_mat = np.array([[sigma_x,0.],[0.,sigma_y]])

            kernel = utils.multivariate_normal(xy, mu, sigma_mat).reshape(size,size)

            dense = np.random.random((100,100))
            peak_scan = np.random.randint(0, 100)
            peak_dia_cycle = np.random.randint(0, 100)

            dense[peak_scan, peak_dia_cycle] = 5

            min_size = np.random.randint(0, 20)

            score = convolve_fourier(dense, kernel)

            scan_limit, cycle_limit = f(
                score, 
                peak_scan, 
                peak_dia_cycle,
                f_mobility = 0.99,
                f_rt = 0.99,
                #center_fraction = 0.0005,
                min_size_mobility = min_size,
                min_size_rt = min_size,
                max_size_mobility = 500,
                max_size_rt = 500)

            assert(scan_limit[0] <= scan_limit[1])
            assert(cycle_limit[0] <= cycle_limit[1])
            assert(scan_limit[0] >= 0)
            assert(cycle_limit[0] >= 0)
            assert(scan_limit[1] <= score.shape[0])
            assert(cycle_limit[1] <= score.shape[1])
            assert(scan_limit[0] <= peak_scan)
            assert(scan_limit[1] >= peak_scan)
            assert(cycle_limit[0] <= peak_dia_cycle)
            assert(cycle_limit[1] >= peak_dia_cycle)
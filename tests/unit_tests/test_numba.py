
import numpy as np

# local
from alphadia.extraction.numba.fragments import (
        get_ion_group_mapping
    )

from alphadia.extraction.numba.numeric import (
        transpose
    )

def test_get_ion_group_mapping():
    """Test the get_ion_group_mapping function."""

    ion_mz = np.array([100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.])
    ion_count = np.ones(len(ion_mz), dtype=np.uint8)
    ion_precursor = np.repeat(np.arange(2), 5)
    ion_intensity = np.random.rand(len(ion_mz))

    precursor_group = np.array([0, 0])
    precursor_abundance = np.array([1, 1])

    print(ion_mz)
    print(ion_count)
    print(ion_precursor)

    print(precursor_group)
    print(precursor_abundance)

    mz, intensity = get_ion_group_mapping(
        ion_precursor,
        ion_mz,
        ion_intensity,
        precursor_abundance,
        precursor_group,
        exclude_shared=False
    )

    assert np.allclose(mz, ion_mz)
    assert np.allclose(np.ceil(intensity), np.ones((1, 10)))
    assert np.all(intensity.shape == (1, 10))

    mz, intensity = get_ion_group_mapping(
        ion_precursor,
        ion_mz,
        ion_intensity,
        precursor_abundance,
        precursor_group,
        exclude_shared=True
    )

    assert np.allclose(mz, ion_mz)
    assert np.allclose(np.ceil(intensity), np.ones((1, 10)))
    assert np.all(intensity.shape == (1, 10))

    precursor_group = np.array([0, 1])

    mz, intensity = get_ion_group_mapping(
        ion_precursor,
        ion_mz,
        ion_intensity,
        precursor_abundance,
        precursor_group,
        exclude_shared=False
    )
    print(intensity)
    assert np.allclose(mz, ion_mz)
    expected = np.ones((2, 10))
    expected[0, 5:] = 0
    expected[1, :5] = 0

    assert np.allclose(np.ceil(intensity), expected)
    assert np.all(intensity.shape == (2, 10))

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
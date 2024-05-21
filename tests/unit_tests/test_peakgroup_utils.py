import numba as nb
import numpy as np
import pytest

from alphadia.peakgroup.utils import assemble_isotope_mz


@nb.njit(cache=True)
def wrap_assemble_isotope_mz(mz, charge, intensities):
    return assemble_isotope_mz(mz, charge, intensities)


@pytest.mark.parametrize(
    "should_fail, mz, charge, intensities",
    [
        (False, 100.0, 1, np.array([1, 2, 3, 4], dtype=np.float32)),
        (False, 100.0, 1, np.array([1, 2, 3, 4], dtype=np.float64)),
        (True, 100.0, 1, np.array([1, 2, 3, 4], dtype=np.float32).reshape(2, 2)),
    ],
)
def test_assemble_isotope_mz(should_fail, mz, charge, intensities):
    if should_fail:
        with pytest.raises(Exception):  # noqa: B017
            wrap_assemble_isotope_mz(mz, charge, intensities)
    else:
        wrap_assemble_isotope_mz(mz, charge, intensities)

import numba as nb
import numpy as np
from utils import USE_NUMBA_CACHING


@nb.njit(cache=USE_NUMBA_CACHING)
def mass_range(mz_list, ppm_tolerance):
    out_mz = np.zeros((len(mz_list), 2), dtype=mz_list.dtype)
    out_mz[:, 0] = mz_list - ppm_tolerance * mz_list / (10**6)
    out_mz[:, 1] = mz_list + ppm_tolerance * mz_list / (10**6)
    return out_mz

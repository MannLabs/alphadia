"""JIT-compiled utility functions for search operations.

This module provides general-purpose numba-compiled utility functions used across
the search pipeline, including mass calculations, array operations, and frame indexing.
"""

import math

import numba as nb
import numpy as np

from alphadia.utils import USE_NUMBA_CACHING


@nb.njit(cache=USE_NUMBA_CACHING)
def mass_range(mz_list, ppm_tolerance):
    out_mz = np.zeros((len(mz_list), 2), dtype=mz_list.dtype)
    out_mz[:, 0] = mz_list - ppm_tolerance * mz_list / (10**6)
    out_mz[:, 1] = mz_list + ppm_tolerance * mz_list / (10**6)
    return out_mz


@nb.njit(inline="always", cache=USE_NUMBA_CACHING)
def get_frame_indices(
    rt_values: np.ndarray,
    rt_values_array: np.ndarray,
    zeroth_frame: int,
    cycle_len: int,
    precursor_cycle_max_index: int,
    optimize_size: int = 16,
    min_size: int = 32,
) -> np.ndarray:
    """Convert an interval of two rt values to a frame index interval.
    The length of the interval is rounded up so that a multiple of `optimize_size` cycles are included.

    Parameters
    ----------
    rt_values : np.ndarray, shape = (2,), dtype = float32
        Array of rt values.
    rt_values_array : np.ndarray
        Array containing all rt values for searching.
    zeroth_frame : int
        Indicator if the first frame is zero.
    cycle_len : int
        The size of the cycle dimension.
    precursor_cycle_max_index : int
        Maximum index for precursor cycles.
    optimize_size : int, default = 16
        Optimize for FFT efficiency by using multiples of this size.
    min_size : int, default = 32
        Minimum number of DIA cycles to include.

    Returns
    -------
    np.ndarray, shape = (1, 3), dtype = int64
        Array of frame indices (start, stop, 1)

    """
    if rt_values.shape != (2,):
        raise ValueError("rt_values must be a numpy array of shape (2,)")

    frame_index = np.searchsorted(rt_values_array, rt_values, "left")

    precursor_cycle_limits = (frame_index + zeroth_frame) // cycle_len
    precursor_cycle_len = precursor_cycle_limits[1] - precursor_cycle_limits[0]

    # Apply minimum size
    optimal_len = max(precursor_cycle_len, min_size)
    # Round up to the next multiple of `optimize_size`
    optimal_len = int(optimize_size * math.ceil(optimal_len / optimize_size))

    # By default, extend the precursor cycle to the right
    optimal_cycle_limits = np.array(
        [precursor_cycle_limits[0], precursor_cycle_limits[0] + optimal_len],
        dtype=np.int64,
    )

    # If the cycle is too long, extend it to the left
    if optimal_cycle_limits[1] > precursor_cycle_max_index:
        optimal_cycle_limits[1] = precursor_cycle_max_index
        optimal_cycle_limits[0] = precursor_cycle_max_index - optimal_len

        if optimal_cycle_limits[0] < 0:
            optimal_cycle_limits[0] = 0 if precursor_cycle_max_index % 2 == 0 else 1

    # Convert back to frame indices
    frame_limits = optimal_cycle_limits * cycle_len + zeroth_frame
    return make_slice_1d(frame_limits)


@nb.njit(cache=USE_NUMBA_CACHING)
def make_slice_1d(start_stop):
    """Numba helper function to create a 1D slice object from a start and stop value.

        e.g. make_slice_1d([0, 10]) -> np.array([[0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (2,) containing the start and stop value.

    Returns
    -------
    np.ndarray
        Array of shape (1,3) containing the start, stop and step value.

    """
    return np.array([[start_stop[0], start_stop[1], 1]], dtype=start_stop.dtype)


@nb.njit(cache=USE_NUMBA_CACHING)
def make_slice_2d(start_stop):
    """Numba helper function to create a 2D slice object from multiple start and stop value.

        e.g. make_slice_2d([[0, 10], [0, 10]]) -> np.array([[0, 10, 1], [0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (N, 2) containing the start and stop value for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing the start, stop and step value for each dimension.

    """
    out = np.ones((start_stop.shape[0], 3), dtype=start_stop.dtype)
    out[:, 0] = start_stop[:, 0]
    out[:, 1] = start_stop[:, 1]
    return out

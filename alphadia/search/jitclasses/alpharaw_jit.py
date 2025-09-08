"""JIT-compiled utilities for AlphaRaw data processing.

This module provides numba-compiled functions for efficient processing of
mass spectrometry data from AlphaRaw format, including spectrum extraction
and mass range calculations.
"""

import logging

import numba as nb
import numpy as np

from alphadia.search.jitclasses.utils import get_frame_indices, mass_range
from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger()


@nb.njit(cache=USE_NUMBA_CACHING)
def _calculate_valid_scans(quad_slices: np.ndarray, cycle: np.ndarray):
    """Calculate the DIA cycle quadrupole mask for each score group.

    Parameters
    ----------
    quad_slices : np.ndarray
        The quadrupole slices for each score group. (1, 2)

    cycle : np.ndarray
        The DIA cycle quadrupole mask. (1, n_precursor, 1, 2)

    Returns
    -------
    np.ndarray
        The precursor index of all scans within the quad slices. (n_precursor_indices)

    """
    if not quad_slices.ndim == 2:
        raise ValueError("quad_slices must be of shape (1, 2)")

    if not cycle.ndim == 4:
        raise ValueError("cycle must be of shape (1, n_precursor, 1, 2)")

    flat_cycle = cycle.reshape(-1, 2)
    precursor_idx_list = []

    for i, (mz_start, mz_stop) in enumerate(flat_cycle):
        if (quad_slices[0, 0] <= mz_stop) and (quad_slices[0, 1] >= mz_start):
            precursor_idx_list.append(i)

    return np.array(precursor_idx_list)


@nb.njit(parallel=False, fastmath=True, cache=USE_NUMBA_CACHING)
def _search_sorted_left(slice, value):
    left = 0
    right = len(slice)

    while left < right:
        mid = (left + right) >> 1
        if slice[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@nb.njit(inline="always", fastmath=True, cache=USE_NUMBA_CACHING)
def _search_sorted_reference_left(array, left, right, value):
    while left < right:
        mid = (left + right) >> 1
        if array[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@nb.experimental.jitclass(
    [
        ("has_mobility", nb.core.types.boolean),
        ("cycle", nb.core.types.float64[:, :, :, ::1]),
        ("rt_values", nb.core.types.float32[::1]),
        ("mobility_values", nb.core.types.float32[::1]),
        ("zeroth_frame", nb.core.types.boolean),
        ("max_mz_value", nb.core.types.float32),
        ("min_mz_value", nb.core.types.float32),
        ("quad_max_mz_value", nb.core.types.float32),
        ("quad_min_mz_value", nb.core.types.float32),
        ("precursor_cycle_max_index", nb.core.types.int64),
        ("peak_start_idx_list", nb.core.types.int64[::1]),
        ("peak_stop_idx_list", nb.core.types.int64[::1]),
        ("mz_values", nb.core.types.float32[::1]),
        ("intensity_values", nb.core.types.float32[::1]),
        ("scan_max_index", nb.core.types.int64),
        ("frame_max_index", nb.core.types.int64),
    ]
)
class AlphaRawJIT:
    """Numba compatible AlphaRaw data structure."""

    def __init__(
        self,
        cycle: nb.core.types.float64[:, :, :, ::1],
        rt_values: nb.core.types.float32[::1],
        mobility_values: nb.core.types.float32[::1],
        zeroth_frame: nb.core.types.boolean,
        max_mz_value: nb.core.types.float32,
        min_mz_value: nb.core.types.float32,
        quad_max_mz_value: nb.core.types.float32,
        quad_min_mz_value: nb.core.types.float32,
        precursor_cycle_max_index: nb.core.types.int64,
        peak_start_idx_list: nb.core.types.int64[::1],
        peak_stop_idx_list: nb.core.types.int64[::1],
        mz_values: nb.core.types.float32[::1],
        intensity_values: nb.core.types.float32[::1],
        scan_max_index: nb.core.types.int64,
        frame_max_index: nb.core.types.int64,
    ):
        """Numba compatible AlphaRaw data structure."""
        self.has_mobility = False

        self.cycle = cycle
        self.rt_values = rt_values
        self.mobility_values = mobility_values
        self.zeroth_frame = zeroth_frame
        self.max_mz_value = max_mz_value
        self.min_mz_value = min_mz_value
        self.quad_max_mz_value = quad_max_mz_value
        self.quad_min_mz_value = quad_min_mz_value
        self.precursor_cycle_max_index = precursor_cycle_max_index

        self.peak_start_idx_list = peak_start_idx_list
        self.peak_stop_idx_list = peak_stop_idx_list
        self.mz_values = mz_values
        self.intensity_values = intensity_values

        self.scan_max_index = scan_max_index
        self.frame_max_index = frame_max_index

    def _get_frame_indices(
        self, rt_values: np.array, optimize_size: int = 16, min_size: int = 32
    ):
        """Convert an interval of two rt values to a frame index interval.
        The length of the interval is rounded up so that a multiple of 16 cycles are included.

        Parameters
        ----------
        rt_values : np.ndarray, shape = (2,), dtype = float32
            array of rt values

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the precursor cycle to a multiple of 16

        min_size : int, default = 32
            The minimum number of dia cycles to include

        Returns
        -------
        np.ndarray, shape = (2,), dtype = int64
            array of frame indices

        """
        return get_frame_indices(
            rt_values=rt_values,
            rt_values_array=self.rt_values,
            zeroth_frame=self.zeroth_frame,
            cycle_len=self.cycle.shape[1],
            precursor_cycle_max_index=self.precursor_cycle_max_index,
            optimize_size=optimize_size,
            min_size=min_size,
        )

    def get_frame_indices_tolerance(
        self, rt: float, tolerance: float, optimize_size: int = 16, min_size: int = 32
    ):
        """Determine the frame indices for a given retention time and tolerance.
        The frame indices will make sure to include full precursor cycles and will be optimized for fft.

        Parameters
        ----------
        rt : float
            retention time in seconds

        tolerance : float
            tolerance in seconds

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the precursor cycle to a multiple of 16

        min_size : int, default = 32
            The minimum number of dia cycles to include

        Returns
        -------
        np.ndarray, shape = (1, 3,), dtype = int64
            array which contains a slice object for the frame indices [[start, stop step]]

        """
        rt_limits = np.array([rt - tolerance, rt + tolerance], dtype=np.float32)

        return self._get_frame_indices(
            rt_limits, optimize_size=optimize_size, min_size=min_size
        )

    def get_scan_indices_tolerance(self, mobility, tolerance, optimize_size=16):
        return np.array([[0, 2, 1]], dtype=np.int64)

    def get_dense(
        self,
        frame_limits,
        scan_limits,
        mz_query_list,
        mass_tolerance,
        quadrupole_mz,
        absolute_masses=False,
        custom_cycle=None,
    ):
        """Get a dense representation of the data for a given set of parameters.

        Parameters
        ----------
        frame_limits : np.ndarray, shape = (1,2,)
            array of frame indices

        scan_limits : np.ndarray, shape = (1,2,)
            array of scan indices

        mz_query_list : np.ndarray, shape = (n_tof_slices,)
            array of query m/z values

        mass_tolerance : float
            mass tolerance in ppm

        quadrupole_mz : np.ndarray, shape = (1,2,)
            array of quadrupole m/z values

        absolute_masses : bool, default = False
            if True, the first slice of the dense output will contain the absolute m/z values instead of the mass error

        custom_cycle : np.ndarray, shape = (1, n_precursor, 1, 2), default = None
            custom cycle quadrupole mask, for example after calibration

        Returns
        -------
        np.ndarray, shape = (2, n_tof_slices, n_precursor_indices, 2, n_precursor_cycles)

        """
        # intensities below HIGH_EPSILON will be set to zero
        HIGH_EPSILON = 1e-26

        # LOW_EPSILON will be used to avoid division errors
        # as LOW_EPSILON will be added to the numerator and denominator
        # intensity values approaching LOW_EPSILON would result in updated dim1 values with 1
        # therefore, LOW_EPSILON should be orderes of magnitude smaller than HIGH_EPSILON
        # TODO: refactor the calculation of dim1 for performance and numerical stability
        LOW_EPSILON = 1e-36

        # (n_tof_slices, 2) array of start, stop mz for each slice
        mz_query_slices = mass_range(mz_query_list, mass_tolerance)
        n_tof_slices = len(mz_query_slices)

        cycle_length = self.cycle.shape[1]

        # (n_precursors) array of precursor indices, the precursor index refers to each scan within the cycle
        precursor_idx_list = _calculate_valid_scans(quadrupole_mz, self.cycle)
        n_precursor_indices = len(precursor_idx_list)

        precursor_cycle_start = frame_limits[0, 0] // cycle_length
        precursor_cycle_stop = frame_limits[0, 1] // cycle_length
        precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

        dense_output = np.zeros(
            (2, n_tof_slices, n_precursor_indices, 2, precursor_cycle_len),
            dtype=np.float32,
        )
        if absolute_masses:
            pass
        else:
            dense_output[1, :, :, :, :] = mass_tolerance

        for i, cycle_idx in enumerate(
            range(precursor_cycle_start, precursor_cycle_stop)
        ):
            for j, precursor_idx in enumerate(precursor_idx_list):
                scan_idx = precursor_idx + cycle_idx * cycle_length

                peak_start_idx = self.peak_start_idx_list[scan_idx]
                peak_stop_idx = self.peak_stop_idx_list[scan_idx]

                idx = peak_start_idx

                for k, (mz_query_start, mz_query_stop) in enumerate(mz_query_slices):
                    rel_idx = _search_sorted_left(
                        self.mz_values[idx:peak_stop_idx], mz_query_start
                    )

                    idx += rel_idx

                    while idx < peak_stop_idx and self.mz_values[idx] <= mz_query_stop:
                        accumulated_intensity = dense_output[0, k, j, 0, i]
                        accumulated_dim1 = dense_output[1, k, j, 0, i]

                        new_intensity = self.intensity_values[idx]
                        new_intensity = new_intensity * (new_intensity > HIGH_EPSILON)
                        new_mz_value = self.mz_values[idx]

                        if absolute_masses:
                            new_dim1 = (
                                accumulated_dim1 * accumulated_intensity
                                + new_intensity * new_mz_value
                                + LOW_EPSILON
                            ) / (accumulated_intensity + new_intensity + LOW_EPSILON)

                        else:
                            new_error = (
                                (new_mz_value - mz_query_list[k])
                                / mz_query_list[k]
                                * 10**6
                            )
                            new_dim1 = (
                                accumulated_dim1 * accumulated_intensity
                                + new_intensity * new_error
                                + LOW_EPSILON
                            ) / (accumulated_intensity + new_intensity + LOW_EPSILON)

                        dense_output[0, k, j, 0, i] = (
                            accumulated_intensity + new_intensity
                        )
                        dense_output[0, k, j, 1, i] = (
                            accumulated_intensity + new_intensity
                        )
                        dense_output[1, k, j, 0, i] = new_dim1
                        dense_output[1, k, j, 1, i] = new_dim1

                        idx += 1

        return dense_output, precursor_idx_list

    def get_dense_intensity(
        self,
        frame_limits,
        scan_limits,
        mz_query_list,
        mass_tolerance,
        quadrupole_mz,
        absolute_masses=False,
        custom_cycle=None,
    ):
        """Get a dense representation of the data for a given set of parameters.

        Parameters
        ----------
        frame_limits : np.ndarray, shape = (1,2,)
            array of frame indices

        scan_limits : np.ndarray, shape = (1,2,)
            array of scan indices

        mz_query_list : np.ndarray, shape = (n_tof_slices,)
            array of query m/z values

        mass_tolerance : float
            mass tolerance in ppm

        quadrupole_mz : np.ndarray, shape = (1,2,)
            array of quadrupole m/z values

        absolute_masses : bool, default = False
            if True, the first slice of the dense output will contain the absolute m/z values instead of the mass error

        custom_cycle : np.ndarray, shape = (1, n_precursor, 1, 2), default = None
            custom cycle quadrupole mask, for example after calibration

        Returns
        -------
        np.ndarray, shape = (1, n_tof_slices, n_precursor_indices, 2, n_precursor_cycles)

        """
        # (n_tof_slices, 2) array of start, stop mz for each slice
        mz_query_slices = mass_range(mz_query_list, mass_tolerance)
        n_tof_slices = len(mz_query_slices)

        cycle_length = self.cycle.shape[1]

        # (n_precursors) array of precursor indices, the precursor index refers to each scan within the cycle
        precursor_idx_list = _calculate_valid_scans(quadrupole_mz, self.cycle)
        # n_precursor_indices = len(precursor_idx_list)

        precursor_cycle_start = frame_limits[0, 0] // cycle_length
        precursor_cycle_stop = frame_limits[0, 1] // cycle_length
        precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

        dense_output = np.zeros(
            (1, n_tof_slices, 2, precursor_cycle_len),
            dtype=np.float32,
        )

        for i, cycle_idx in enumerate(
            range(precursor_cycle_start, precursor_cycle_stop)
        ):
            for precursor_idx in precursor_idx_list:
                scan_idx = precursor_idx + cycle_idx * cycle_length

                peak_start_idx = self.peak_start_idx_list[scan_idx]
                peak_stop_idx = self.peak_stop_idx_list[scan_idx]

                idx = peak_start_idx

                for k, (mz_query_start, mz_query_stop) in enumerate(mz_query_slices):
                    idx = _search_sorted_reference_left(
                        self.mz_values, idx, peak_stop_idx, mz_query_start
                    )

                    while idx < peak_stop_idx and self.mz_values[idx] <= mz_query_stop:
                        accumulated_intensity = dense_output[0, k, 0, i]
                        # accumulated_dim1 = dense_output[1, k, 0, i]

                        new_intensity = self.intensity_values[idx]

                        dense_output[0, k, 0, i] = accumulated_intensity + new_intensity
                        dense_output[0, k, 1, i] = accumulated_intensity + new_intensity

                        idx += 1

        return dense_output, precursor_idx_list

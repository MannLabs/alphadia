"""JIT-compiled data structures for Bruker timsTOF data processing.

This module provides numba-compiled classes and functions specifically optimized
for processing Bruker timsTOF data, including ion mobility and frame-based
data structures.
"""

import math

import numpy as np
from numba.core import types
from numba.experimental import jitclass

from alphadia.search.jitclasses.utils import (
    get_frame_indices,
    make_slice_1d,
    make_slice_2d,
    mass_range,
)


@jitclass(
    [
        ("accumulation_times", types.float64[:]),
        ("cycle", types.float64[:, :, :, ::1]),
        ("dia_mz_cycle", types.float64[:, ::1]),
        ("dia_precursor_cycle", types.int64[::1]),
        ("frame_max_index", types.int64),
        ("intensity_corrections", types.float64[::1]),
        ("intensity_max_value", types.int64),
        ("intensity_min_value", types.int64),
        ("max_accumulation_time", types.float64),
        ("mobility_max_value", types.float64),
        ("mobility_min_value", types.float64),
        ("mobility_values", types.float64[::1]),
        ("mz_values", types.float64[::1]),
        ("precursor_indices", types.int64[::1]),
        ("precursor_max_index", types.int64),
        ("quad_indptr", types.int64[::1]),
        ("quad_max_mz_value", types.float64),
        ("quad_min_mz_value", types.float64),
        ("quad_mz_values", types.float64[::1, :]),
        ("raw_quad_indptr", types.int64[::1]),
        ("rt_values", types.float64[::1]),
        ("scan_max_index", types.int64),
        ("tof_max_index", types.int64),
        ("use_calibrated_mz_values_as_default", types.int64),
        ("zeroth_frame", types.boolean),
        ("precursor_cycle_max_index", types.int64),
        ("push_indices", types.uint32[::1]),
        ("tof_indptr", types.int64[::1]),
        ("intensity_values", types.uint16[::1]),
        ("has_mobility", types.boolean),
    ]
)
class TimsTOFTransposeJIT:
    """Numba compatible transposed TimsTOF data structure."""

    def __init__(
        self,
        accumulation_times: types.float64[::1],
        cycle: types.float64[:, :, :, ::1],
        dia_mz_cycle: types.float64[:, ::1],
        dia_precursor_cycle: types.int64[::1],
        frame_max_index: types.int64,
        intensity_corrections: types.float64[::1],
        intensity_max_value: types.int64,
        intensity_min_value: types.int64,
        intensity_values: types.uint16[::1],
        max_accumulation_time: types.float64,
        mobility_max_value: types.float64,
        mobility_min_value: types.float64,
        mobility_values: types.float64[::1],
        mz_values: types.float64[::1],
        precursor_indices: types.int64[::1],
        precursor_max_index: types.int64,
        # push_indptr: types.int64[::1],
        quad_indptr: types.int64[::1],
        quad_max_mz_value: types.float64,
        quad_min_mz_value: types.float64,
        quad_mz_values: types.float64[::1, :],
        raw_quad_indptr: types.int64[::1],
        rt_values: types.float64[::1],
        scan_max_index: types.int64,
        # tof_indices: types.uint32[::1],
        tof_max_index: types.int64,
        use_calibrated_mz_values_as_default: types.int64,
        zeroth_frame: types.boolean,
        push_indices: types.uint32[::1],
        tof_indptr: types.int64[::1],
    ):
        """Numba compatible transposed TimsTOF data structure.

        Parameters
        ----------
        accumulation_times : np.ndarray, shape = (n_frames,), dtype = float64
            array of accumulation times

        """
        self.accumulation_times = accumulation_times
        self.cycle = cycle
        self.dia_mz_cycle = dia_mz_cycle

        self.dia_precursor_cycle = dia_precursor_cycle
        self.frame_max_index = frame_max_index
        self.intensity_corrections = intensity_corrections
        self.intensity_max_value = intensity_max_value
        self.intensity_min_value = intensity_min_value
        self.max_accumulation_time = max_accumulation_time
        self.mobility_max_value = mobility_max_value
        self.mobility_min_value = mobility_min_value
        self.mobility_values = mobility_values
        self.mz_values = mz_values

        self.precursor_indices = precursor_indices
        self.precursor_max_index = precursor_max_index
        self.quad_indptr = quad_indptr
        self.quad_max_mz_value = quad_max_mz_value
        self.quad_min_mz_value = quad_min_mz_value
        self.quad_mz_values = quad_mz_values
        self.raw_quad_indptr = raw_quad_indptr
        self.rt_values = rt_values

        self.scan_max_index = scan_max_index

        self.tof_max_index = tof_max_index

        self.use_calibrated_mz_values_as_default = use_calibrated_mz_values_as_default
        self.zeroth_frame = zeroth_frame

        self.precursor_cycle_max_index = frame_max_index // self.cycle.shape[1]

        self.push_indices = push_indices
        self.tof_indptr = tof_indptr
        self.intensity_values = intensity_values

        self.has_mobility = True

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

    def _get_scan_indices(self, mobility_values: np.array, optimize_size: int = 16):
        """Convert array of mobility values into scan indices, njit compatible.

        Parameters
        ----------
        mobility_values : np.ndarray, shape = (2,), dtype = float32
            array of mobility values

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the scan to a multiple of 16

        Returns
        -------
        np.ndarray, shape = (2,), dtype = int64
            array of scan indices

        """
        scan_index = self.scan_max_index - np.searchsorted(
            self.mobility_values[::-1], mobility_values, "right"
        )

        # scan_index[1] += 1

        scan_len = scan_index[0] - scan_index[1]
        # round up to the next multiple of 16
        optimal_len = int(optimize_size * math.ceil(scan_len / optimize_size))

        # by default, we extend the scans cycle to the bottom
        optimal_scan_limits = np.array(
            [scan_index[0], scan_index[0] - optimal_len], dtype=np.int64
        )

        # if the scan is too short, we extend it to the top

        if optimal_scan_limits[1] < 0:
            optimal_scan_limits[1] = 0
            optimal_scan_limits[0] = optimal_len

            # if the scan is too long, we truncate it
            optimal_scan_limits[0] = min(optimal_scan_limits[0], self.scan_max_index)

        return make_slice_1d(optimal_scan_limits)

    def get_scan_indices_tolerance(self, mobility, tolerance, optimize_size=16):
        """Determine the scan limits for the elution group based on the mobility and mobility tolerance.

        Parameters
        ----------
        mobility : float
            mobility value

        tolerance : float
            mobility tolerance

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the scan to a multiple of 16

        Returns
        -------
        np.ndarray, shape = (1, 3,), dtype = int64
            array which contains a slice object for the scan indices [[start, stop step]]

        """
        mobility_limits = np.array(
            [mobility + tolerance, mobility - tolerance], dtype=np.float32
        )

        return self._get_scan_indices(mobility_limits, optimize_size=optimize_size)

    def _get_tof_indices(
        self,
        mz_values: np.ndarray,
    ):
        """Convert array of mobility values into scan indices, njit compatible"""
        return np.searchsorted(self.mz_values, mz_values, "left")

    def _cycle_mask(self, quad_slices: np.ndarray, custom_cycle: np.ndarray = None):
        """Calculate the DIA cycle quadrupole mask for each score group.

        Parameters
        ----------
        cycle : np.ndarray
            The DIA mz cycle as part of the bruker.TimsTOF object. (n_frames * n_scans)

        quad_slices : np.ndarray
            The quadrupole slices for each score group. (n_score_groups, 2)

        Returns
        -------
        np.ndarray
            The DIA cycle quadrupole mask for each score group. (n_score_groups, n_frames * n_scans)

        """
        n_score_groups = quad_slices.shape[0]

        if custom_cycle is None:
            dia_mz_cycle = self.cycle.reshape(-1, 2)

        else:
            if not custom_cycle.ndim == 4:
                raise ValueError("custom_cycle must be a 4d array")
            dia_mz_cycle = custom_cycle.reshape(-1, 2)

        mz_mask = np.zeros((n_score_groups, len(dia_mz_cycle)), dtype=np.bool_)
        for i, (mz_start, mz_stop) in enumerate(dia_mz_cycle):
            for j, (quad_mz_start, quad_mz_stop) in enumerate(quad_slices):
                if (quad_mz_start <= mz_stop) and (quad_mz_stop >= mz_start):
                    mz_mask[j, i] = True

        return mz_mask

    def _get_push_indices(
        self,
        frame_limits,
        scan_limits,
        cycle_mask,
    ):
        n_score_groups = cycle_mask.shape[0]

        push_indices = []
        absolute_precursor_cycle = []
        len_dia_mz_cycle = len(self.dia_mz_cycle)

        frame_start, frame_stop, frame_step = frame_limits[0]
        scan_start, scan_stop, scan_step = scan_limits[0]
        for frame_index in range(frame_start, frame_stop, frame_step):
            for scan_index in range(scan_start, scan_stop, scan_step):
                push_index = frame_index * self.scan_max_index + scan_index
                # subtract a whole frame if the first frame is zero
                if self.zeroth_frame:
                    cyclic_push_index = push_index - self.scan_max_index
                else:
                    cyclic_push_index = push_index

                # gives the scan index in the dia mz cycle
                scan_in_dia_mz_cycle = cyclic_push_index % len_dia_mz_cycle

                # check fragment push indices
                for i in range(n_score_groups):
                    if cycle_mask[i, scan_in_dia_mz_cycle]:
                        precursor_cycle = self.dia_precursor_cycle[scan_in_dia_mz_cycle]
                        absolute_precursor_cycle.append(precursor_cycle)
                        push_indices.append(push_index)

        return np.array(push_indices, dtype=np.uint32), np.array(
            absolute_precursor_cycle, dtype=np.int64
        )

    def _assemble_push(
        self,
        tof_limits,
        mz_values,
        push_query,
        precursor_index,
        frame_limits,
        scan_limits,
        ppm_background,
        absolute_masses=False,
    ):
        if len(precursor_index) == 0:
            return np.empty((0, 0, 0, 0, 0), dtype=np.float32), np.empty(
                (0), dtype=np.int64
            )

        unique_precursor_index = np.unique(precursor_index)
        precursor_index_reverse = np.zeros(
            np.max(unique_precursor_index) + 1, dtype=np.uint8
        )
        precursor_index_reverse[unique_precursor_index] = np.arange(
            len(unique_precursor_index)
        )

        relative_precursor_index = precursor_index_reverse[precursor_index]

        n_precursor_indices = len(unique_precursor_index)
        n_tof_slices = len(tof_limits)

        # scan values
        mobility_start = int(scan_limits[0, 0])
        mobility_stop = int(scan_limits[0, 1])
        mobility_len = mobility_stop - mobility_start

        # cycle values
        precursor_cycle_start = (
            int(frame_limits[0, 0] - self.zeroth_frame) // self.cycle.shape[1]
        )
        precursor_cycle_stop = (
            int(frame_limits[0, 1] - self.zeroth_frame) // self.cycle.shape[1]
        )
        precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

        dense_output = np.zeros(
            (2, n_tof_slices, n_precursor_indices, mobility_len, precursor_cycle_len),
            dtype=np.float32,
        )

        # intensities below HIGH_EPSILON will be set to zero
        HIGH_EPSILON = 1e-26

        # LOW_EPSILON will be used to avoid division errors
        # as LOW_EPSILON will be added to the numerator and denominator
        # intensity values approaching LOW_EPSILON would result in updated dim1 values with 1
        # therefore, LOW_EPSILON should be orderes of magnitude smaller than HIGH_EPSILON
        # TODO: refactor the calculation of dim1 for performance and numerical stability
        LOW_EPSILON = 1e-36

        if absolute_masses:
            pass
        else:
            dense_output[1, :, :, :, :] = ppm_background

        for j, (tof_start, tof_stop, tof_step) in enumerate(tof_limits):
            library_mz_value = mz_values[j]

            for tof_index in range(tof_start, tof_stop, tof_step):
                measured_mz_value = self.mz_values[tof_index]

                start = self.tof_indptr[tof_index]
                stop = self.tof_indptr[tof_index + 1]

                i = 0
                idx = int(start)

                while (idx < stop) and (i < len(push_query)):
                    if push_query[i] < self.push_indices[idx]:
                        i += 1

                    else:
                        if push_query[i] == self.push_indices[idx]:
                            frame_index = self.push_indices[idx] // self.scan_max_index
                            scan_index = self.push_indices[idx] % self.scan_max_index
                            precursor_cycle_index = (
                                frame_index - self.zeroth_frame
                            ) // self.cycle.shape[1]

                            relative_scan = scan_index - mobility_start
                            relative_precursor = (
                                precursor_cycle_index - precursor_cycle_start
                            )

                            accumulated_intensity = dense_output[
                                0,
                                j,
                                relative_precursor_index[i],
                                relative_scan,
                                relative_precursor,
                            ]
                            accumulated_dim1 = dense_output[
                                1,
                                j,
                                relative_precursor_index[i],
                                relative_scan,
                                relative_precursor,
                            ]

                            new_intensity = self.intensity_values[idx]
                            new_intensity = new_intensity * (
                                new_intensity > HIGH_EPSILON
                            )

                            if absolute_masses:
                                new_dim1 = (
                                    accumulated_dim1 * accumulated_intensity
                                    + new_intensity * measured_mz_value
                                    + LOW_EPSILON
                                ) / (
                                    accumulated_intensity + new_intensity + LOW_EPSILON
                                )

                            else:
                                new_error = (
                                    (measured_mz_value - library_mz_value)
                                    / library_mz_value
                                    * 10**6
                                )
                                new_dim1 = (
                                    accumulated_dim1 * accumulated_intensity
                                    + new_intensity * new_error
                                    + LOW_EPSILON
                                ) / (
                                    accumulated_intensity + new_intensity + LOW_EPSILON
                                )

                            dense_output[
                                0,
                                j,
                                relative_precursor_index[i],
                                relative_scan,
                                relative_precursor,
                            ] = accumulated_intensity + new_intensity
                            dense_output[
                                1,
                                j,
                                relative_precursor_index[i],
                                relative_scan,
                                relative_precursor,
                            ] = new_dim1

                        idx = idx + 1

        return dense_output, unique_precursor_index

    def _assemble_push_intensity(
        self,
        tof_limits,
        mz_values,
        push_query,
        precursor_index,
        frame_limits,
        scan_limits,
        ppm_background,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(precursor_index) == 0:
            return np.empty((0, 0, 0, 0), dtype=np.float32), np.empty(
                (0), dtype=np.int64
            )

        unique_precursor_index = np.unique(precursor_index)
        precursor_index_reverse = np.zeros(
            np.max(unique_precursor_index) + 1, dtype=np.uint8
        )
        precursor_index_reverse[unique_precursor_index] = np.arange(
            len(unique_precursor_index)
        )

        n_tof_slices = len(tof_limits)

        # scan valuesa
        mobility_start = int(scan_limits[0, 0])
        mobility_stop = int(scan_limits[0, 1])
        mobility_len = mobility_stop - mobility_start

        # cycle values
        precursor_cycle_start = (
            int(frame_limits[0, 0] - self.zeroth_frame) // self.cycle.shape[1]
        )
        precursor_cycle_stop = (
            int(frame_limits[0, 1] - self.zeroth_frame) // self.cycle.shape[1]
        )
        precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

        dense_output = np.zeros(
            (1, n_tof_slices, mobility_len, precursor_cycle_len),
            dtype=np.float32,
        )

        for j, (tof_start, tof_stop, tof_step) in enumerate(tof_limits):
            for tof_index in range(tof_start, tof_stop, tof_step):
                start = self.tof_indptr[tof_index]
                stop = self.tof_indptr[tof_index + 1]

                i = 0
                idx = int(start)

                while (idx < stop) and (i < len(push_query)):
                    if push_query[i] < self.push_indices[idx]:
                        i += 1

                    else:
                        if push_query[i] == self.push_indices[idx]:
                            frame_index = self.push_indices[idx] // self.scan_max_index
                            scan_index = self.push_indices[idx] % self.scan_max_index
                            precursor_cycle_index = (
                                frame_index - self.zeroth_frame
                            ) // self.cycle.shape[1]

                            relative_scan = scan_index - mobility_start
                            relative_precursor = (
                                precursor_cycle_index - precursor_cycle_start
                            )

                            dense_output[
                                0,
                                j,
                                relative_scan,
                                relative_precursor,
                            ] += self.intensity_values[idx]

                        idx = idx + 1

        return dense_output, unique_precursor_index

    def get_dense(
        self,
        frame_limits,
        scan_limits,
        mz_values,
        mass_tolerance,
        quadrupole_mz,
        absolute_masses=False,
        custom_cycle=None,
    ):
        tof_limits = make_slice_2d(
            self._get_tof_indices(mass_range(mz_values, mass_tolerance))
        )

        mz_mask = self._cycle_mask(quadrupole_mz, custom_cycle)

        push_query, _absolute_precursor_index = self._get_push_indices(
            frame_limits, scan_limits, mz_mask
        )

        return self._assemble_push(
            tof_limits,
            mz_values,
            push_query,
            _absolute_precursor_index,
            frame_limits,
            scan_limits,
            mass_tolerance,
            absolute_masses=absolute_masses,
        )

    def get_dense_intensity(
        self,
        frame_limits,
        scan_limits,
        mz_values,
        mass_tolerance,
        quadrupole_mz,
        absolute_masses=False,
        custom_cycle=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tof_limits = make_slice_2d(
            self._get_tof_indices(mass_range(mz_values, mass_tolerance))
        )

        mz_mask = self._cycle_mask(quadrupole_mz, custom_cycle)

        push_query, _absolute_precursor_index = self._get_push_indices(
            frame_limits, scan_limits, mz_mask
        )

        return self._assemble_push_intensity(
            tof_limits,
            mz_values,
            push_query,
            _absolute_precursor_index,
            frame_limits,
            scan_limits,
            mass_tolerance,
        )

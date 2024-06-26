# native imports
import logging
import math
import os

import alphatims.bruker
import alphatims.tempmmap as tm

# alpha family imports
import alphatims.utils
import numba as nb

# third party imports
import numpy as np
from numba.core import types
from numba.experimental import jitclass

# alphadia imports
from alphadia import utils
from alphadia.data.stats import log_stats

logger = logging.getLogger()


class TimsTOFTranspose(alphatims.bruker.TimsTOF):
    """Transposed TimsTOF data structure."""

    def __init__(
        self,
        bruker_d_folder_name: str,
        *,
        mz_estimation_from_frame: int = 1,
        mobility_estimation_from_frame: int = 1,
        slice_as_dataframe: bool = True,
        use_calibrated_mz_values_as_default: int = 0,
        use_hdf_if_available: bool = True,
        mmap_detector_events: bool = True,
        drop_polarity: bool = True,
        convert_polarity_to_int: bool = True,
    ):
        self.has_mobility = True
        self.has_ms1 = True
        self.mmap_detector_events = mmap_detector_events

        if bruker_d_folder_name.endswith("/"):
            bruker_d_folder_name = bruker_d_folder_name[:-1]
        logger.info(f"Importing data from {bruker_d_folder_name}")
        if bruker_d_folder_name.endswith(".d"):
            bruker_hdf_file_name = f"{bruker_d_folder_name[:-2]}.hdf"
            hdf_file_exists = os.path.exists(bruker_hdf_file_name)
            if use_hdf_if_available and hdf_file_exists:
                self._import_data_from_hdf_file(
                    bruker_hdf_file_name,
                    mmap_detector_events,
                )
                self.bruker_hdf_file_name = bruker_hdf_file_name
            else:
                self.bruker_d_folder_name = os.path.abspath(bruker_d_folder_name)
                self._import_data_from_d_folder(
                    bruker_d_folder_name,
                    mz_estimation_from_frame,
                    mobility_estimation_from_frame,
                    drop_polarity,
                    convert_polarity_to_int,
                    mmap_detector_events,
                )

                if self._cycle.shape[0] != 1:
                    logger.error(
                        "Unexpected cycle shape. Will only retain first frame group"
                    )
                    raise ValueError(
                        "Unexpected cycle shape. Will only retain first frame group"
                    )

                self.transpose()
        elif bruker_d_folder_name.endswith(".hdf"):
            self._import_data_from_hdf_file(
                bruker_d_folder_name,
                mmap_detector_events,
            )
            self.bruker_hdf_file_name = bruker_d_folder_name
        else:
            raise NotImplementedError("WARNING: file extension not understood")
        if not hasattr(self, "version"):
            self._version = "N.A."
        if self.version != alphatims.__version__:
            logger.info(
                "WARNING: "
                f"AlphaTims version {self.version} was used to initialize "
                f"{bruker_d_folder_name}, while the current version of "
                f"AlphaTims is {alphatims.__version__}."
            )
        self.slice_as_dataframe = slice_as_dataframe
        self.use_calibrated_mz_values_as_default(use_calibrated_mz_values_as_default)

        # Precompile
        logger.info(f"Successfully imported data from {bruker_d_folder_name}")
        log_stats(self.rt_values, self.cycle)

    def transpose(self):
        # abort if transposed data is already present
        if hasattr(self, "_push_indices") and hasattr(self, "_tof_indptr"):
            logger.info("Transposed data already present, aborting")
            return

        logger.info("Transposing detector events")
        push_indices, tof_indptr, intensity_values = transpose(
            self._tof_indices, self._push_indptr, self._intensity_values
        )
        logger.info("Finished transposing data")

        self._tof_indices = np.zeros(1, np.uint32)
        self._push_indptr = np.zeros(1, np.int64)

        if self.mmap_detector_events:
            self._push_indices = tm.clone(push_indices)
            self._tof_indptr = tm.clone(tof_indptr)
            self._intensity_values = tm.clone(intensity_values)
        else:
            self._push_indices = push_indices
            self._tof_indptr = tof_indptr
            self._intensity_values = intensity_values

    def _import_data_from_hdf_file(
        self,
        bruker_d_folder_name: str,
        mmap_detector_events: bool = False,
    ):
        raise NotImplementedError("Not implemented yet for TimsTOFTranspose")

    def _import_data_from_hdf_file(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet for TimsTOFTranspose")

    def jitclass(self):
        return TimsTOFTransposeJIT(
            self._accumulation_times,
            self._cycle,
            self._dia_mz_cycle,
            self._dia_precursor_cycle,
            self._frame_max_index,
            self._intensity_corrections,
            self._intensity_max_value,
            self._intensity_min_value,
            self._intensity_values,
            self._max_accumulation_time,
            self._mobility_max_value,
            self._mobility_min_value,
            self._mobility_values,
            self._mz_values,
            self._precursor_indices,
            self._precursor_max_index,
            # self._push_indptr,
            self._quad_indptr,
            self._quad_max_mz_value,
            self._quad_min_mz_value,
            self._quad_mz_values,
            self._raw_quad_indptr,
            self._rt_values,
            self._scan_max_index,
            # self._tof_indices,
            self._tof_max_index,
            self._use_calibrated_mz_values_as_default,
            self._zeroth_frame,
            self._push_indices,
            self._tof_indptr,
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

    def get_frame_indices(self, rt_values: np.array, optimize_size: int = 16):
        """

        Convert an interval of two rt values to a frame index interval.
        The length of the interval is rounded up so that a multiple of 16 cycles are included.

        Parameters
        ----------
        rt_values : np.ndarray, shape = (2,), dtype = float32
            array of rt values

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the precursor cycle to a multiple of 16

        Returns
        -------
        np.ndarray, shape = (2,), dtype = int64
            array of frame indices

        """

        if rt_values.shape != (2,):
            raise ValueError("rt_values must be a numpy array of shape (2,)")

        frame_index = np.searchsorted(self.rt_values, rt_values, "left")

        precursor_cycle_limits = (frame_index + self.zeroth_frame) // self.cycle.shape[
            1
        ]
        precursor_cycle_len = precursor_cycle_limits[1] - precursor_cycle_limits[0]

        # round up to the next multiple of 16
        optimal_len = int(
            optimize_size * math.ceil(precursor_cycle_len / optimize_size)
        )

        # by default, we extend the precursor cycle to the right
        optimal_cycle_limits = np.array(
            [precursor_cycle_limits[0], precursor_cycle_limits[0] + optimal_len],
            dtype=np.int64,
        )

        # if the cycle is too long, we extend it to the left
        if optimal_cycle_limits[1] > self.precursor_cycle_max_index:
            optimal_cycle_limits[1] = self.precursor_cycle_max_index
            optimal_cycle_limits[0] = self.precursor_cycle_max_index - optimal_len

            if optimal_cycle_limits[0] < 0:
                optimal_cycle_limits[0] = (
                    0 if self.precursor_cycle_max_index % 2 == 0 else 1
                )

        # second element is the index of the first whole cycle which should not be used
        # precursor_cycle_limits[1] += 1
        # convert back to frame indices
        frame_limits = optimal_cycle_limits * self.cycle.shape[1] + self.zeroth_frame
        return utils.make_slice_1d(frame_limits)

    def get_frame_indices_tolerance(
        self, rt: float, tolerance: float, optimize_size: int = 16
    ):
        """
        Determine the frame indices for a given retention time and tolerance.
        The frame indices will make sure to include full precursor cycles and will be optimized for fft.

        Parameters
        ----------
        rt : float
            retention time in seconds

        tolerance : float
            tolerance in seconds

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the precursor cycle to a multiple of 16

        Returns
        -------
        np.ndarray, shape = (1, 3,), dtype = int64
            array which contains a slice object for the frame indices [[start, stop step]]

        """

        rt_limits = np.array([rt - tolerance, rt + tolerance], dtype=np.float32)

        return self.get_frame_indices(rt_limits, optimize_size=optimize_size)

    def get_scan_indices(self, mobility_values: np.array, optimize_size: int = 16):
        """convert array of mobility values into scan indices, njit compatible.

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
            if optimal_scan_limits[0] > self.scan_max_index:
                optimal_scan_limits[0] = self.scan_max_index

        return utils.make_slice_1d(optimal_scan_limits)

    def get_scan_indices_tolerance(self, mobility, tolerance, optimize_size=16):
        """
        Determine the scan limits for the elution group based on the mobility and mobility tolerance.

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

        return self.get_scan_indices(mobility_limits, optimize_size=optimize_size)

    def get_tof_indices(
        self,
        mz_values: np.ndarray,
    ):
        """convert array of mobility values into scan indices, njit compatible"""
        return np.searchsorted(self.mz_values, mz_values, "left")

    def get_tof_indices_tolerance(
        self,
        mz_values: np.ndarray,
        tolerance: float,
    ):
        mz_limits = utils.mass_range(mz_values, tolerance)
        return utils.make_slice_2d(self.get_tof_indices(mz_limits))

    def cycle_mask(self, quad_slices: np.ndarray, custom_cycle: np.ndarray = None):
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

    def get_push_indices(
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

    def assemble_push(
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
            (2, n_tof_slices, n_precursor_indices, mobility_len, precursor_cycle_len),
            dtype=np.float32,
        )

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

                            if absolute_masses:
                                new_dim1 = (
                                    accumulated_dim1 * accumulated_intensity
                                    + new_intensity * measured_mz_value
                                ) / (accumulated_intensity + new_intensity)

                            else:
                                new_error = (
                                    (measured_mz_value - library_mz_value)
                                    / library_mz_value
                                    * 10**6
                                )
                                new_dim1 = (
                                    accumulated_dim1 * accumulated_intensity
                                    + new_intensity * new_error
                                ) / (accumulated_intensity + new_intensity)

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

    def assemble_push_intensity(
        self,
        tof_limits,
        mz_values,
        push_query,
        precursor_index,
        frame_limits,
        scan_limits,
        ppm_background,
    ):
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
        tof_limits = utils.make_slice_2d(
            self.get_tof_indices(utils.mass_range(mz_values, mass_tolerance))
        )

        mz_mask = self.cycle_mask(quadrupole_mz, custom_cycle)

        push_query, _absolute_precursor_index = self.get_push_indices(
            frame_limits, scan_limits, mz_mask
        )

        return self.assemble_push(
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
    ):
        tof_limits = utils.make_slice_2d(
            self.get_tof_indices(utils.mass_range(mz_values, mass_tolerance))
        )

        mz_mask = self.cycle_mask(quadrupole_mz, custom_cycle)

        push_query, _absolute_precursor_index = self.get_push_indices(
            frame_limits, scan_limits, mz_mask
        )

        return self.assemble_push_intensity(
            tof_limits,
            mz_values,
            push_query,
            _absolute_precursor_index,
            frame_limits,
            scan_limits,
            mass_tolerance,
        )


@alphatims.utils.pjit()
def transpose_chunk(
    chunk_idx,
    chunks,
    push_indices,
    push_indptr,
    tof_indices,
    tof_indptr,
    values,
    new_values,
    tof_indcount,
):
    tof_index_chunk_start = chunks[chunk_idx]
    tof_index_chunk_stop = chunks[chunk_idx + 1]

    for push_idx in range(len(push_indptr) - 1):
        start_push_indptr = push_indptr[push_idx]
        stop_push_indptr = push_indptr[push_idx + 1]

        for idx in range(start_push_indptr, stop_push_indptr):
            # new row
            tof_index = tof_indices[idx]
            if tof_index_chunk_start <= tof_index and tof_index < tof_index_chunk_stop:
                push_indices[tof_indptr[tof_index] + tof_indcount[tof_index]] = push_idx
                new_values[tof_indptr[tof_index] + tof_indcount[tof_index]] = values[
                    idx
                ]
                tof_indcount[tof_index] += 1


@nb.njit
def build_chunks(number_of_elements, num_chunks):
    # Calculate the number of chunks needed
    chunk_size = (number_of_elements + num_chunks - 1) // num_chunks

    chunks = [0]
    start = 0

    for _ in range(num_chunks):
        stop = min(start + chunk_size, number_of_elements)
        chunks.append(stop)
        start = stop

    return np.array(chunks)


@nb.njit
def transpose(tof_indices, push_indptr, values):
    """
    The default alphatims data format consists of a sparse matrix where pushes are the rows, tof indices (discrete mz values) the columns and intensities the values.
    A lookup starts with a given push index p which points to the row. The start and stop indices of the row are accessed from dia_data.push_indptr[p] and dia_data.push_indptr[p+1].
    The tof indices are then accessed from dia_data.tof_indices[start:stop] and the corresponding intensities from dia_data.intensity_values[start:stop].

    The function transposes the data such that the tof indices are the rows and the pushes are the columns.
    This is usefull when accessing only a small number of tof indices (e.g. when extracting a single precursor) and the number of pushes is large (e.g. when extracting a whole run).

    Parameters
    ----------

    tof_indices : np.ndarray
        column indices (n_values)

    push_indptr : np.ndarray
        start stop values for each row (n_rows +1)

    values : np.ndarray
        values (n_values)

    threads : int
        number of threads to use

    Returns
    -------

    push_indices : np.ndarray
        row indices (n_values)

    tof_indptr : np.ndarray
        start stop values for each row (n_rows +1)

    new_values : np.ndarray
        values (n_values)

    """
    # this is one less than the old col count or the new row count
    max_tof_index = tof_indices.max()

    tof_indcount = np.zeros((max_tof_index + 1), dtype=np.uint32)

    # get new row counts
    for v in tof_indices:
        tof_indcount[v] += 1

    # get new indptr
    tof_indptr = np.zeros((max_tof_index + 1 + 1), dtype=np.int64)

    for i in range(max_tof_index + 1):
        tof_indptr[i + 1] = tof_indptr[i] + tof_indcount[i]

    tof_indcount = np.zeros((max_tof_index + 1), dtype=np.uint32)

    # get new values
    push_indices = np.zeros((len(tof_indices)), dtype=np.uint32)
    new_values = np.zeros_like(values)

    chunks = build_chunks(max_tof_index + 1, 20)

    with nb.objmode:
        alphatims.utils.set_threads(20)

        transpose_chunk(
            range(len(chunks) - 1),
            chunks,
            push_indices,
            push_indptr,
            tof_indices,
            tof_indptr,
            values,
            new_values,
            tof_indcount,
        )

    return push_indices, tof_indptr, new_values

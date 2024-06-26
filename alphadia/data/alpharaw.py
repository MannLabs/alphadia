# native imports
import logging
import math
import os

import numba as nb

# third party imports
import numpy as np
import pandas as pd
from alpharaw import mzml as alpharawmzml

# TODO fix: "import resolves to its containing file"
from alpharaw import sciex as alpharawsciex

# alpha family imports
from alpharaw import (
    thermo as alpharawthermo,
)

# alphadia imports
from alphadia import utils
from alphadia.data.stats import log_stats

logger = logging.getLogger()


@nb.njit(parallel=False, fastmath=True)
def search_sorted_left(slice, value):
    left = 0
    right = len(slice)

    while left < right:
        mid = (left + right) >> 1
        if slice[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@nb.njit(inline="always", fastmath=True)
def search_sorted_refernce_left(array, left, right, value):
    while left < right:
        mid = (left + right) >> 1
        if array[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


def normed_auto_correlation(x: np.ndarray):
    """Calculate the normalized auto correlation of a 1D array.
    Parameters
    ----------

    x : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The normalized auto correlation of the input array.

    """
    x = x - x.mean()
    result = np.correlate(x, x, mode="full")
    result = result[len(result) // 2 :]
    result /= result[0]
    return result


def get_cycle_length(cycle_signature: np.ndarray):
    """Get the cycle length from the cycle signature.

    Parameters
    ----------

    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    Returns
    -------

    cycle_length: int
        The length of the DIA cycle.
    """

    corr = normed_auto_correlation(cycle_signature)

    is_peak = (corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:])

    peak_index = is_peak.nonzero()[0] + 1
    argmax = np.argmax(corr[peak_index])

    return peak_index[argmax]


@nb.njit
def get_cycle_start(
    cycle_signature: np.ndarray,
    cycle_length: int,
):
    """Get the cycle start from the cycle signature.

    Parameters
    ----------

    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    cycle_length: int
        The length of the DIA cycle.

    Returns
    -------

    cycle_start: int
        The index of the first cycle in the signature.
    """
    for i in range(len(cycle_signature) - (2 * cycle_length)):
        if np.all(cycle_signature[i : i + cycle_length] == cycle_signature[i]):
            continue

        if np.all(
            cycle_signature[i : i + cycle_length]
            == cycle_signature[i + cycle_length : i + 2 * cycle_length]
        ):
            return i

    return -1


@nb.njit
def assert_cycle(cycle_signature: np.ndarray, cycle_length: int, cycle_start: int):
    """Assert that the found DIA cycle is valid.

    Parameters
    ----------

    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    cycle_length: int
        The length of the DIA cycle.

    cycle_start: int
        The index of the first cycle in the signature.

    Returns
    -------

    cycle_valid: bool
        True if the cycle is valid, False otherwise.
    """

    cycle_valid = True
    for i in range(len(cycle_signature) - (2 * cycle_length) - cycle_start):
        if not np.all(
            cycle_signature[i + cycle_start : i + cycle_start + cycle_length]
            == cycle_signature[
                i + cycle_start + cycle_length : i + cycle_start + 2 * cycle_length
            ]
        ):
            cycle_valid = False
            break
    return cycle_valid


def determine_dia_cycle(
    spectrum_df: pd.DataFrame,
    subset_for_cycle_detection: int = 10000,
):
    """Determine the DIA cycle and store it in self.cycle.

    Parameters
    ----------

    spectrum_df : pandas.DataFrame
        AlphaRaw compatible spectrum dataframe.

    subset_for_cycle_detection : int, default = 10000
        The number of spectra to use for cycle detection.

    """
    logger.info("Determining DIA cycle")

    cycle_signature = (
        spectrum_df.isolation_lower_mz.values[:subset_for_cycle_detection]
        + spectrum_df.isolation_upper_mz.values[:subset_for_cycle_detection]
    )
    cycle_length = get_cycle_length(cycle_signature)

    cycle_start = get_cycle_start(cycle_signature, cycle_length)

    if cycle_start == -1:
        raise ValueError("Failed to determine start of DIA cycle.")

    if not assert_cycle(cycle_signature, cycle_length, cycle_start):
        raise ValueError(
            f"Cycle with start {spectrum_df.rt.values[cycle_start]:.2f} min and length {cycle_length} detected, but does not consistent."
        )

    logger.info(
        f"Found cycle with start {spectrum_df.rt.values[cycle_start]:.2f} min and length {cycle_length}."
    )

    cycle = np.zeros((1, cycle_length, 1, 2), dtype=np.float64)
    cycle[0, :, 0, 0] = spectrum_df.isolation_lower_mz.values[
        cycle_start : cycle_start + cycle_length
    ]
    cycle[0, :, 0, 1] = spectrum_df.isolation_upper_mz.values[
        cycle_start : cycle_start + cycle_length
    ]

    return cycle, cycle_start, cycle_length


@nb.njit
def calculate_valid_scans(quad_slices: np.ndarray, cycle: np.ndarray):
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


class AlphaRaw(alpharawthermo.MSData_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_mobility = False
        self.has_ms1 = True

    def process_alpharaw(self, **kwargs):
        self.sample_name = os.path.basename(self.raw_file_path)

        # the filter spectra function is implemented in the sub-class
        self.filter_spectra(**kwargs)

        self.rt_values = self.spectrum_df.rt.values.astype(np.float32) * 60
        self.zeroth_frame = 0

        try:
            # determine the DIA cycle
            self.cycle, self.cycle_start, self.cycle_length = determine_dia_cycle(
                self.spectrum_df
            )
        except ValueError:
            logger.warning(
                "Failed to determine DIA cycle, will retry without MS1 spectra."
            )

            self.spectrum_df = self.spectrum_df[self.spectrum_df.ms_level > 1]
            self.cycle, self.cycle_start, self.cycle_length = determine_dia_cycle(
                self.spectrum_df
            )
            self.has_ms1 = False

        self.spectrum_df = self.spectrum_df.iloc[self.cycle_start :]
        self.rt_values = self.spectrum_df.rt.values.astype(np.float32) * 60

        self.precursor_cycle_max_index = len(self.rt_values) // self.cycle.shape[1]
        self.mobility_values = np.array([1e-6, 0], dtype=np.float32)

        self.max_mz_value = self.spectrum_df.precursor_mz.max().astype(np.float32)
        self.min_mz_value = self.spectrum_df.precursor_mz.min().astype(np.float32)

        self.quad_max_mz_value = (
            self.spectrum_df[self.spectrum_df["ms_level"] == 2]
            .isolation_upper_mz.max()
            .astype(np.float32)
        )
        self.quad_min_mz_value = (
            self.spectrum_df[self.spectrum_df["ms_level"] == 2]
            .isolation_lower_mz.min()
            .astype(np.float32)
        )

        self.peak_start_idx_list = self.spectrum_df.peak_start_idx.values.astype(
            np.int64
        )
        self.peak_stop_idx_list = self.spectrum_df.peak_stop_idx.values.astype(np.int64)
        self.mz_values = self.peak_df.mz.values.astype(np.float32)
        self.intensity_values = self.peak_df.intensity.values.astype(np.float32)

        self.scan_max_index = 1
        self.frame_max_index = len(self.rt_values) - 1

    def filter_spectra(self, **kwargs):
        """Filter the spectra.
        This function is implemented in the sub-class.
        """

        pass

    def jitclass(self):
        return AlphaRawJIT(
            self.cycle,
            self.rt_values,
            self.mobility_values,
            self.zeroth_frame,
            self.max_mz_value,
            self.min_mz_value,
            self.quad_max_mz_value,
            self.quad_min_mz_value,
            self.precursor_cycle_max_index,
            self.peak_start_idx_list,
            self.peak_stop_idx_list,
            self.mz_values,
            self.intensity_values,
            self.scan_max_index,
            self.frame_max_index,
        )


class MzML(AlphaRaw, alpharawmzml.MzMLReader):
    def __init__(self, raw_file_path: str, process_count: int = 10, **kwargs):
        super().__init__(process_count=process_count)
        self.load_raw(raw_file_path)
        self.process_alpharaw(**kwargs)
        log_stats(self.rt_values, self.cycle)


class Sciex(AlphaRaw, alpharawsciex.SciexWiffData):
    def __init__(self, raw_file_path: str, process_count: int = 10, **kwargs):
        super().__init__(process_count=process_count)
        self.load_raw(raw_file_path)
        self.process_alpharaw(**kwargs)
        log_stats(self.rt_values, self.cycle)


class Thermo(AlphaRaw, alpharawthermo.ThermoRawData):
    def __init__(self, raw_file_path: str, process_count: int = 10, **kwargs):
        super().__init__(process_count=process_count)
        self.load_raw(raw_file_path)
        self.process_alpharaw(**kwargs)
        log_stats(self.rt_values, self.cycle)

    def filter_spectra(self, cv: float = None, astral_ms1: bool = False, **kwargs):
        """
        Filter the spectra for MS1 or MS2 spectra.
        """

        # filter for Astral or Orbitrap MS1 spectra
        if astral_ms1:
            self.spectrum_df = self.spectrum_df[self.spectrum_df["nce"] > 0.1]
            self.spectrum_df.loc[self.spectrum_df["nce"] < 1.1, "ms_level"] = 1
            self.spectrum_df.loc[self.spectrum_df["nce"] < 1.1, "precursor_mz"] = -1.0
            self.spectrum_df.loc[
                self.spectrum_df["nce"] < 1.1, "isolation_lower_mz"
            ] = -1.0
            self.spectrum_df.loc[
                self.spectrum_df["nce"] < 1.1, "isolation_upper_mz"
            ] = -1.0
        else:
            self.spectrum_df = self.spectrum_df[
                (self.spectrum_df["nce"] < 0.1) | (self.spectrum_df["nce"] > 1.1)
            ]

        # filter for cv values if multiple cv values are present
        if cv is not None and "cv" in self.spectrum_df.columns:
            # use np.isclose to account for floating point errors
            logger.info(f"Filtering for CV {cv}")
            logger.info(f"Before: {len(self.spectrum_df)}")
            self.spectrum_df = self.spectrum_df[
                np.isclose(self.spectrum_df["cv"], cv, atol=0.1)
            ]
            logger.info(f"After: {len(self.spectrum_df)}")

        self.spectrum_df["spec_idx"] = np.arange(len(self.spectrum_df))


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

        frame_index = np.zeros(len(rt_values), dtype=np.int64)
        for i, rt in enumerate(rt_values):
            frame_index[i] = search_sorted_left(self.rt_values, rt)

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
        return np.array([[0, 2, 1]], dtype=np.int64)

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
        """
        Get a dense representation of the data for a given set of parameters.

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
        mz_query_slices = utils.mass_range(mz_query_list, mass_tolerance)
        n_tof_slices = len(mz_query_slices)

        cycle_length = self.cycle.shape[1]

        # (n_precursors) array of precursor indices, the precursor index refers to each scan within the cycle
        precursor_idx_list = calculate_valid_scans(quadrupole_mz, self.cycle)
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
                    rel_idx = search_sorted_left(
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
        """
        Get a dense representation of the data for a given set of parameters.

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
        mz_query_slices = utils.mass_range(mz_query_list, mass_tolerance)
        n_tof_slices = len(mz_query_slices)

        cycle_length = self.cycle.shape[1]

        # (n_precursors) array of precursor indices, the precursor index refers to each scan within the cycle
        precursor_idx_list = calculate_valid_scans(quadrupole_mz, self.cycle)
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
                    idx = search_sorted_refernce_left(
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


def get_dense_intensity(
    cycle,
    peak_start_idx_list,
    peak_stop_idx_list,
    mz_values,
    intensity_values,
    frame_limits,
    scan_limits,
    mz_query_list,
    mass_tolerance,
    quadrupole_mz,
    absolute_masses=False,
    custom_cycle=None,
):
    """
    Get a dense representation of the data for a given set of parameters.

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
    mz_query_slices = utils.mass_range(mz_query_list, mass_tolerance)
    n_tof_slices = len(mz_query_slices)

    cycle_length = cycle.shape[1]

    # (n_precursors) array of precursor indices, the precursor index refers to each scan within the cycle
    precursor_idx_list = calculate_valid_scans(quadrupole_mz, cycle)
    n_precursor_indices = len(precursor_idx_list)
    # print('n_precursor_indices', n_precursor_indices)

    precursor_cycle_start = frame_limits[0, 0] // cycle_length
    precursor_cycle_stop = frame_limits[0, 1] // cycle_length
    precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

    dense_output = np.zeros(
        (1, n_tof_slices, n_precursor_indices, 2, precursor_cycle_len),
        dtype=np.float32,
    )

    # print('precursor_idx_list', precursor_idx_list)

    for i, cycle_idx in enumerate(range(precursor_cycle_start, precursor_cycle_stop)):
        for j, precursor_idx in enumerate(precursor_idx_list):
            scan_idx = precursor_idx + cycle_idx * cycle_length

            peak_start_idx = peak_start_idx_list[scan_idx]
            peak_stop_idx = peak_stop_idx_list[scan_idx]

            idx = peak_start_idx

            # above:
            # 0.58.0 0.0128s
            # 0.59.0 0.0135s

            for k, (mz_query_start, mz_query_stop) in enumerate(mz_query_slices):
                idx = search_sorted_refernce_left(
                    mz_values, idx, peak_stop_idx, mz_query_start
                )

                # above:
                # 0.59.0 8.24s
                # 0.58.0 3.17s

                while idx < peak_stop_idx and mz_values[idx] <= mz_query_stop:
                    accumulated_intensity = dense_output[0, k, j, 0, i]
                    # accumulated_dim1 = dense_output[1, k, j, 0, i]

                    new_intensity = intensity_values[idx]

                    dense_output[0, k, j, 0, i] = accumulated_intensity + new_intensity
                    dense_output[0, k, j, 1, i] = accumulated_intensity + new_intensity

                    idx += 1

    return dense_output, precursor_idx_list

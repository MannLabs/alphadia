# native imports
import math
import os
import logging

logger = logging.getLogger()

# alphadia imports
from alphadia import utils

# alpha family imports
from alpharaw import thermo as alpharawthermo

# third party imports
import numpy as np
import numba as nb


def normed_auto_correlation(x):
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


def calculate_cycle(spectrum_df):
    """Calculate the DIA cycle quadrupole schedule.
    This function will try to find a repeating pattern of quadrupole isolation windows in the data.
    The pattern is found by calculating the auto correlation of the isolation window m/z values.
    The cycle length is then determined by the first peak in the auto correlation.

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        The spectrum dataframe.

    Returns
    -------
    np.ndarray
        The DIA cycle quadrupole mask. (1, n_precursor, 1, 2)

    """
    # the cycle length is calculated by using the auto correlation of the isolation window m/z values
    x = (
        spectrum_df.isolation_lower_mz.values[:10000]
        + spectrum_df.isolation_upper_mz.values[:10000]
    )
    corr = normed_auto_correlation(x)
    corr[0] = 0
    cycle_length = np.argmax(corr)

    # check that the cycles really match
    first_cycle = (
        spectrum_df.isolation_lower_mz.values[:cycle_length]
        + spectrum_df.isolation_upper_mz.values[:cycle_length]
    )
    second_cycle = (
        spectrum_df.isolation_lower_mz.values[cycle_length : 2 * cycle_length]
        + spectrum_df.isolation_upper_mz.values[cycle_length : 2 * cycle_length]
    )
    if not np.allclose(first_cycle, second_cycle):
        raise ValueError("No DIA cycle pattern found in the data.")

    cycle = np.zeros((1, cycle_length, 1, 2), dtype=np.float64)
    cycle[0, :, 0, 0] = spectrum_df.isolation_lower_mz.values[:cycle_length]
    cycle[0, :, 0, 1] = spectrum_df.isolation_upper_mz.values[:cycle_length]

    return cycle


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


class Thermo(alpharawthermo.ThermoRawData):
    has_mobility = False

    def __init__(self, path, astral_ms1=False, cv=None, **kwargs):
        super().__init__(**kwargs)
        self.load_raw(path)

        self.sample_name = os.path.basename(self.raw_file_path)

        self.astral_ms1 = astral_ms1
        self.cv = cv
        self.filter_spectra()

        self.cycle = calculate_cycle(self.spectrum_df)
        self.rt_values = self.spectrum_df.rt.values.astype(np.float32) * 60
        self.zeroth_frame = 0
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

    def filter_spectra(self):
        print(self.cv, "cv" in self.spectrum_df.columns)

        # filter for astral MS1
        if self.astral_ms1:
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

        # filter for cv
        if self.cv is not None:
            if "cv" in self.spectrum_df.columns:
                # use np.isclose to account for floating point errors
                logger.info(f"Filtering for CV {self.cv}")
                logger.info(f"Before: {len(self.spectrum_df)}")
                self.spectrum_df = self.spectrum_df[
                    np.isclose(self.spectrum_df["cv"], self.cv, atol=0.1)
                ]
                logger.info(f"After: {len(self.spectrum_df)}")

        self.spectrum_df["spec_idx"] = np.arange(len(self.spectrum_df))

    def jitclass(self):
        return ThermoJIT(
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
class ThermoJIT(object):
    """Numba compatible Thermo data structure."""

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
        """Numba compatible Thermo data structure."""

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
                    rel_idx = np.searchsorted(
                        self.mz_values[idx:peak_stop_idx], mz_query_start, "left"
                    )

                    idx += rel_idx

                    while idx < peak_stop_idx and self.mz_values[idx] <= mz_query_stop:
                        accumulated_intensity = dense_output[0, k, j, 0, i]
                        accumulated_dim1 = dense_output[1, k, j, 0, i]

                        new_intensity = self.intensity_values[idx]
                        new_mz_value = self.mz_values[idx]

                        if absolute_masses:
                            new_dim1 = (
                                accumulated_dim1 * accumulated_intensity
                                + new_intensity * new_mz_value
                                + 1e-6
                            ) / (accumulated_intensity + new_intensity + 1e-6)
                        else:
                            new_error = (
                                (new_mz_value - mz_query_list[k])
                                / mz_query_list[k]
                                * 10**6
                            )
                            new_dim1 = (
                                accumulated_dim1 * accumulated_intensity
                                + new_intensity * new_error
                                + 1e-6
                            ) / (accumulated_intensity + new_intensity + 1e-6)

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

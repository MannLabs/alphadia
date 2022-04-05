"""A module to perform smoothing of TOF data."""

import alphatims.utils
import alphadia.tempmmap as tm
import numpy as np
import logging


@alphatims.utils.njit(nogil=True)
def get_connections_within_cycle(
    scan_tolerance: int,
    scan_max_index: int,
    dia_mz_cycle: np.ndarray,
    exclude_self: bool = False,
    multiple_frames: bool = False,
    ms1: bool = True,
    ms2: bool = False,
) -> tuple:
    """Determine how individual pushes in a cycle are connected.

    Parameters
    ----------
    scan_tolerance : int
        Maximum scan distance for two pushes to be connected
    scan_max_index : int
        The maximum scan index (dia_data.scan_max_index).
    dia_mz_cycle : np.ndarray
        An np.float64[:, 2] array with upper and lower quadrupole boundaries
        per push of a cycle.
    exclude_self : bool
        Excluded connections between equal push indices
        (the default is False).
    multiple_frames : bool
        Connect scans between different frames a cycle
        (the default is False).
    ms1 : bool
        Allow connections between MS1 pushes
        (the default is True).
    ms2 : bool
        OAllow connections between MS2 pushes
        (the default is False).

    Returns
    -------
    tuple
        A tuple with indptr and indices defining the (sparse) connections.
    """
    connections = []
    connection_count = 0
    connection_counts = [connection_count]
    shape = (
        scan_max_index,
        len(dia_mz_cycle) // scan_max_index
    )
    if multiple_frames:
        frame_iterator = range(shape[1])
    for self_frame in range(shape[1]):
        if not multiple_frames:
            frame_iterator = range(self_frame, self_frame + 1)
        for self_scan in range(shape[0]):
            index = self_scan + self_frame * shape[0]
            low_quad, high_quad = dia_mz_cycle[index]
            if (not ms1) and (low_quad == -1):
                connection_counts.append(connection_count)
                continue
            if (not ms2) and (low_quad != -1):
                connection_counts.append(connection_count)
                continue
            for other_frame in frame_iterator:
                for other_scan in range(
                    self_scan - scan_tolerance,
                    self_scan + scan_tolerance + 1
                ):
                    if not (0 <= other_scan < scan_max_index):
                        continue
                    other_index = other_scan + other_frame * shape[0]
                    if exclude_self and (index == other_index):
                        continue
                    other_low_quad, other_high_quad = dia_mz_cycle[other_index]
                    if low_quad > other_high_quad:
                        continue
                    if high_quad < other_low_quad:
                        continue
                    connection_count += 1
                    connections.append(other_index)
            connection_counts.append(connection_count)
    return np.array(connection_counts), np.array(connections)


@alphatims.utils.njit(nogil=True)
def calculate_cyclic_scan_blur(
    connection_indices: np.ndarray,
    connection_indptr: np.ndarray,
    scan_max_index: int,
    sigma: float = 1,
) -> np.ndarray:
    """Short summary.

    Parameters
    ----------
    connection_indices : np.ndarray
        Connections indices from .get_connections_within_cycle.
    connection_indptr : np.ndarray
        Connections indptr from .get_connections_within_cycle.
    scan_max_index : int
        The maximum scan index (dia_data.scan_max_index).
    sigma : float
        The sigma for the Gaussian blur (default is 1).
        To make sure there are no large dropoffs, this sigma should be at most
        scan_max_index / 3 (see get_connections_within_cycle).

    Returns
    -------
    np.ndarray
        The blurred weight for all the connection_indices.

    """
    scan_blur = np.repeat(
        np.arange(len(connection_indptr) - 1),
        np.diff(connection_indptr),
    ) % scan_max_index - connection_indices % scan_max_index
    scan_blur = np.exp(-(scan_blur / sigma)**2 / 2)
    for i, start in enumerate(connection_indptr[:-1]):
        end = connection_indptr[i + 1]
        scan_blur[start: end] /= np.sum(scan_blur[start: end])
    return scan_blur


@alphatims.utils.pjit
def smooth(
    cycle_index,
    indptr,
    tof_indices,
    intensity_values,
    tof_tolerance,
    scan_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    smooth_intensity_values,
):
    len_dia_mz_cycle = len(connection_counts) - 1
    push_offset = len_dia_mz_cycle * cycle_index + zeroth_frame * scan_max_index
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        connection_end = connection_counts[self_connection_index + 1]
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
            cycle_blur = gauss_correction(cycle_offset)
            for other_connection_index in connections[connection_start: connection_end]:
                connection_blur = gauss_correction(
                    self_connection_index % scan_max_index - other_connection_index % scan_max_index
                )
                other_push_index = push_offset + other_connection_index + len_dia_mz_cycle * cycle_offset
                if other_push_index == self_push_index:
                    continue
                if other_push_index >= len(indptr):
                    continue
                other_start = indptr[other_push_index]
                other_end = indptr[other_push_index + 1]
                if other_start == other_end:
                    continue
                self_index = self_start
                other_index = other_start
                while (self_index < self_end) and (other_index < other_end):
                    self_tof = tof_indices[self_index]
                    other_tof = tof_indices[other_index]
                    if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                        # self_intensity = intensity_values[self_index]
                        other_intensity = intensity_values[other_index]
                        tof_blur = gauss_correction(int(self_tof) - int(other_tof))
                        smooth_intensity_values[self_index] += other_intensity * cycle_blur * connection_blur * tof_blur
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1


@alphatims.utils.njit(nogil=True)
def gauss_correction(x=0, sigma=1):
    return np.exp(-(x / sigma)**2 / 2)


@alphatims.utils.pjit
def find_seeds(
    cycle_index,
    indptr,
    tof_indices,
    smooth_intensity_values,
    tof_tolerance,
    scan_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    peaks,
):
    len_dia_mz_cycle = len(connection_counts) - 1
    push_offset = len_dia_mz_cycle * cycle_index + zeroth_frame * scan_max_index
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        connection_end = connection_counts[self_connection_index + 1]
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
            for other_connection_index in connections[connection_start: connection_end]:
                other_push_index = push_offset + other_connection_index + len_dia_mz_cycle * cycle_offset
                if other_push_index <= self_push_index:
                    continue
                if other_push_index >= len(indptr):
                    continue
                other_start = indptr[other_push_index]
                other_end = indptr[other_push_index + 1]
                if other_start == other_end:
                    continue
                self_index = self_start
                other_index = other_start
                while (self_index < self_end) and (other_index < other_end):
                    self_tof = tof_indices[self_index]
                    other_tof = tof_indices[other_index]
                    if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                        self_intensity = smooth_intensity_values[self_index]
                        other_intensity = smooth_intensity_values[other_index]
                        if self_intensity < other_intensity:
                            peaks[self_index] = False
                        if self_intensity > other_intensity:
                            peaks[other_index] = False
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1

#
#
# # @alphatims.utils.pjit
# @alphatims.utils.njit
# def create_inet(
#     self_push_index,
#     indptr,
#     tof_indices,
#     intensity_values,
#     dia_mz_cycle,
#     tof_tolerance,
#     scan_max_index,
#     tof_max_index,
#     zeroth_frame,
#     connection_counts,
#     connections,
#     cycle_tolerance,
#     is_signal,
#     # mz_values,
# ):
#     intensity_buffer = np.zeros(tof_max_index, dtype=np.float32)
#     new_tof_indices = []
#     new_intensity_values = []
#     index_offset = (self_push_index - zeroth_frame * scan_max_index) % len(dia_mz_cycle)
#     cycle_index = (self_push_index - zeroth_frame * scan_max_index) // len(dia_mz_cycle)
#     push_offset = len(dia_mz_cycle) * cycle_index + zeroth_frame * scan_max_index
#     current_tof_indices = []
#     connection_start = connection_counts[index_offset]
#     connection_end = connection_counts[index_offset + 1]
#     for connection_index in connections[connection_start: connection_end]:
#         for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
#             other_push_index = push_offset + connection_index + len(dia_mz_cycle) * cycle_offset
#             if other_push_index < 0:
#                 continue
#                 # Check mz
#             if other_push_index >= len(indptr):
#                 continue
#             for index in range(
#                 indptr[other_push_index],
#                 indptr[other_push_index + 1]
#             ):
#                 if not is_signal[index]:
#                     continue
#                 tof_index = tof_indices[index]
#                 if intensity_buffer[tof_index] == 0:
#                     current_tof_indices.append(tof_index)
#                 intensity_buffer[tof_index] += intensity_values[index]
#     if len(tof_indices) == 0:
#         return
#     current_tof_indices = sorted(current_tof_indices)
#     last_tof_index = tof_indices[0]
#     last_tof_index = -(1 + tof_tolerance)
#     summed_intensity = intensity_buffer[last_tof_index]
#     summed_tof = last_tof_index
#     count = 1
#     intensity_buffer[last_tof_index] = 0
#     for tof_index in current_tof_indices[1:]:
#         intensity = intensity_buffer[tof_index]
#         if (tof_index - last_tof_index) >= tof_tolerance:
#             if last_tof_index >= 0:
#                 new_tof_indices.append(summed_tof // count)
#                 new_intensity_values.append(summed_intensity)
#             summed_intensity = intensity
#             summed_tof = tof_index
#             count = 1
#         else:
#             summed_intensity += intensity
#             summed_tof += last_tof_index
#             count += 1
#         intensity_buffer[tof_index] = 0
#         last_tof_index = tof_index
#     return (
#         np.array(new_tof_indices),
#         np.array(new_intensity_values),
#     )
#


@alphatims.utils.pjit
def inet_counts(
    cycle_index,
    indptr,
    tof_indices,
    intensity_values,
    tof_tolerance,
    scan_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    inet_indptr,
    peaks,
):
    len_dia_mz_cycle = len(connection_counts) - 1
    push_offset = len_dia_mz_cycle * cycle_index + zeroth_frame * scan_max_index
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        connection_end = connection_counts[self_connection_index + 1]
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
            for other_connection_index in connections[connection_start: connection_end]:
                other_push_index = push_offset + other_connection_index + len_dia_mz_cycle * cycle_offset
                if other_push_index >= len(indptr):
                    continue
                other_start = indptr[other_push_index]
                other_end = indptr[other_push_index + 1]
                if other_start == other_end:
                    continue
                self_index = self_start
                other_index = other_start
                while (self_index < self_end) and (other_index < other_end):
                    self_tof = tof_indices[self_index]
                    other_tof = tof_indices[other_index]
                    if peaks[self_index] and peaks[other_index]:
                        inet_indptr[self_index] += 1
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1


class Analysis(object):

    def __init__(
        self,
        dia_data,
        tof_tolerance=3,
        cycle_tolerance=3,
        scan_tolerance=6,
        multiple_frames_per_cycle=False,
        ms1=True,
        ms2=True,
    ):
        self.dia_data = dia_data
        self.tof_tolerance = tof_tolerance
        self.cycle_tolerance = cycle_tolerance
        self.scan_tolerance = scan_tolerance
        self.multiple_frames_per_cycle = multiple_frames_per_cycle
        self.ms1 = ms1
        self.ms2 = ms2
        logging.info("Setting connections")
        self.connection_counts, self.connections = get_connections_within_cycle(
            scan_tolerance=self.scan_tolerance,
            scan_max_index=self.dia_data.scan_max_index,
            dia_mz_cycle=self.dia_data.dia_mz_cycle,
            multiple_frames=self.multiple_frames_per_cycle,
            ms1=self.ms1,
            ms2=self.ms2,
        )

    def smooth(self):
        logging.info("Smoothing peaks")
        self.smooth_intensity_values = tm.zeros(
            self.dia_data.intensity_values.shape,
            dtype=np.float32
        )
        smooth(
            range(len(self.dia_data.push_indptr) // len(self.dia_data.dia_mz_cycle) + 1),
            self.dia_data.push_indptr,
            self.dia_data.tof_indices,
            self.dia_data.intensity_values,
            self.tof_tolerance,
            self.dia_data.scan_max_index,
            self.dia_data.zeroth_frame,
            self.connection_counts,
            self.connections,
            self.cycle_tolerance,
            self.smooth_intensity_values,
        )

    def find_peaks(self):
        logging.info("Finding peaks")
        self.potential_peaks = tm.array(
            shape=self.smooth_intensity_values.shape,
            dtype=np.bool
        )
        self.potential_peaks[:] = self.smooth_intensity_values > 0
        find_seeds(
            range(len(self.dia_data.push_indptr) // len(self.dia_data.dia_mz_cycle) + 1),
            self.dia_data.push_indptr,
            self.dia_data.tof_indices,
            self.dia_data.intensity_values + self.smooth_intensity_values,
            self.tof_tolerance,
            self.dia_data.scan_max_index,
            self.dia_data.zeroth_frame,
            self.connection_counts,
            self.connections,
            self.cycle_tolerance,
            self.potential_peaks,
        )
        self.smooth_intensity_values += self.dia_data.intensity_values

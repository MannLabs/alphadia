"""A module to perform smoothing of TOF data."""

import logging

import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.preprocessing


import alphatims.utils
import alphatims.tempmmap as tm
import alphabase.io
import alphadia.prefilter


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
    neighbor_types,
    density_values,
    cycle_sigma,
    scan_sigma,
    tof_sigma,
):
    len_dia_mz_cycle = len(connection_counts) - 1
    push_offset = len_dia_mz_cycle * cycle_index + zeroth_frame * scan_max_index
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        connection_end = connection_counts[self_connection_index + 1]
        if connection_end == connection_start:
            continue
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        self_scan = self_connection_index % scan_max_index
        max_neighbor_count = 0
        for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
            cycle_blur = gauss_correction(cycle_offset, cycle_sigma)
            for other_connection_index in connections[connection_start: connection_end]:
                other_scan = other_connection_index % scan_max_index
                connection_blur = gauss_correction(
                    self_connection_index % scan_max_index - other_connection_index % scan_max_index,
                    scan_sigma,
                )
                other_push_index = push_offset + other_connection_index + len_dia_mz_cycle * cycle_offset
                if other_push_index == self_push_index:
                    continue
                if other_push_index >= len(indptr):
                    continue
                max_neighbor_count += 1
                other_start = indptr[other_push_index]
                other_end = indptr[other_push_index + 1]
                if other_start == other_end:
                    continue
                self_index = self_start
                other_index = other_start
                neighbor_type = determine_neighbor_type(
                    cycle_offset,
                    self_scan,
                    other_scan,
                )
                while (self_index < self_end) and (other_index < other_end):
                    self_tof = tof_indices[self_index]
                    other_tof = tof_indices[other_index]
                    if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                        other_intensity = intensity_values[other_index]
                        tof_blur = gauss_correction(
                            int(self_tof) - int(other_tof),
                            tof_sigma,
                        )
                        smooth_intensity_values[self_index] += other_intensity * cycle_blur * connection_blur * tof_blur
                        neighbor_types[self_index] |= neighbor_type
                        density_values[self_index] += 1
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1
        for self_index in range(self_start, self_end):
            density_values[self_index] /= max_neighbor_count


@alphatims.utils.njit(nogil=True)
def gauss_correction(x=0, sigma=1):
    return np.exp(-(x / sigma)**2 / 2)


@alphatims.utils.njit(nogil=True)
def determine_neighbor_type(
    cycle_offset,
    self_scan,
    other_scan,
):
    if cycle_offset < 0:
        if self_scan < other_scan:
            return 2**0
        elif self_scan == other_scan:
            return 2**1
        else:
            return 2**2
    elif cycle_offset == 0:
        if self_scan < other_scan:
            return 2**7
        elif self_scan == other_scan:
            pass  # cannot happen because scan is fully equal?
        else:
            return 2**3
    else:
        if self_scan < other_scan:
            return 2**6
        elif self_scan == other_scan:
            return 2**5
        else:
            return 2**4


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
                # self_index = self_start
                # other_index = other_start
                # while (self_index < self_end) and (other_index < other_end):
                #     # if self_index == other_index:
                #     #     self_index += 1
                #     self_tof = tof_indices[self_index]
                #     other_tof = tof_indices[other_index]
                #     if peaks[self_index] and peaks[other_index]:
                #         inet_indptr[self_index] += 1
                #     if self_tof < other_tof:
                #         self_index += 1
                #     else:
                #         other_index += 1
                count = np.sum(peaks[other_start: other_end])
                # for self_index in range(self_start, self_end):
                #     inet_indptr[self_index] += count
                inet_indptr[self_start] += count


def create_inet(
    dia_data,
    tof_tolerance,
    connection_counts,
    connections,
    cycle_tolerance,
    potential_peaks,
):
    import multiprocessing

    def starfunc(cycle_index):
        return get_inet(
            cycle_index,
            dia_data.push_indptr,
            dia_data.tof_indices,
            tof_tolerance,
            dia_data.scan_max_index,
            dia_data.zeroth_frame,
            connection_counts,
            connections,
            cycle_tolerance,
            potential_peaks,
        )

    iterable = range(len(dia_data.push_indptr) // len(dia_data.dia_mz_cycle) + 1)
    # self.inet_indptr = tm.zeros(
    #     self.dia_data.intensity_values.shape,
    #     dtype=np.int64
    # )
    # iterable = range(500, 520)
    self_connections = []
    other_connections = []
    with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
        for cycle_index, (
            self_connection,
            other_connection,
        ) in alphatims.utils.progress_callback(
            enumerate(pool.imap(starfunc, iterable)),
            total=len(iterable),
            include_progress_callback=True
        ):
            self_connections.append(np.concatenate(self_connection))
            other_connections.append(np.concatenate(other_connection))
    return self_connections, other_connections

# @alphatims.utils.pjit
@alphatims.utils.njit(nogil=True)
def get_inet(
    cycle_index,
    indptr,
    tof_indices,
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
    self_connections = []
    other_connections = []
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
        self_connection = []
        other_connection = []
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
                    # if self_index == other_index:
                    #     self_index += 1
                    self_tof = tof_indices[self_index]
                    other_tof = tof_indices[other_index]
                    if peaks[self_index] and peaks[other_index]:
                        self_connection.append(self_index)
                        other_connection.append(other_index)
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1
        self_connection = np.array(self_connection, dtype=np.int64)
        other_connection = np.array(other_connection, dtype=np.int64)
        order = np.argsort(self_connection)
        self_connections.append(self_connection[order])
        other_connections.append(other_connection[order])
    return self_connections, other_connections


def create_isotopic_pairs(
    analysis,
    difference,
    mz_tolerance,
    # tof_tolerance,
    # connection_counts,
    # connections,
    # cycle_tolerance,
    # potential_peaks,
):
    import multiprocessing

    def starfunc(cycle_index):
        return get_isotopic_pairs(
            cycle_index,
            analysis.peak_collection.indptr,
            analysis.dia_data.mz_values[
                analysis.dia_data.tof_indices[
                    analysis.peak_collection.indices
                ]
            ],
            mz_tolerance,
            analysis.dia_data.scan_max_index,
            analysis.dia_data.zeroth_frame,
            analysis.connection_counts,
            analysis.connections,
            analysis.cycle_tolerance,
            difference,
        )

    iterable = analysis.cycle_range
    # iterable = range(500, 520)
    self_connections = []
    other_connections = []
    with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
        for cycle_index, (
            self_connection,
            other_connection,
        ) in alphatims.utils.progress_callback(
            enumerate(pool.imap(starfunc, iterable)),
            total=len(iterable),
            include_progress_callback=True
        ):
            self_connections.append(self_connection)
            other_connections.append(other_connection)
    return np.concatenate(self_connections), np.concatenate(other_connections)



# @alphatims.utils.pjit
@alphatims.utils.njit(nogil=True)
def get_isotopic_pairs(
    cycle_index,
    indptr,
    mz_values,
    mz_tolerance,
    scan_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    difference,
    # peaks,
):
    len_dia_mz_cycle = len(connection_counts) - 1
    push_offset = len_dia_mz_cycle * cycle_index + zeroth_frame * scan_max_index
    self_connections = []
    other_connections = []
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
                    # if self_index == other_index:
                    #     self_index += 1
                    self_tof = mz_values[self_index]
                    other_tof = mz_values[other_index]
                    if (self_tof - mz_tolerance) <= (other_tof - difference) <= (self_tof + mz_tolerance):
                        self_connections.append(self_index)
                        other_connections.append(other_index)
                    if self_tof < (other_tof - difference - mz_tolerance):
                        self_index += 1
                    else:
                        other_index += 1
    return np.array(self_connections), np.array(other_connections)


@alphatims.utils.njit(nogil=True)
def create_precursor_centric_ion_network(
    cycle_index,
    indices,
    indptr,
    zeroth_frame,
    scan_max_index,
    scan_tolerance,
    cycle_tolerance,
    mz_windows,
    mz_values,
    tof_indices,
    is_mono,
):
    cycle_length = len(mz_windows)
    frame_count = cycle_length // scan_max_index
    push_offset = cycle_length * cycle_index + zeroth_frame * scan_max_index
    precursor_indices = []
    precursor_count = []
    fragment_indices = []
    for self_push_offset in np.flatnonzero(mz_windows[:, 0] == -1):
        self_push_index = push_offset + self_push_offset
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        self_scan = self_push_offset % scan_max_index
        for precursor_index_ in range(self_start, self_end):
            if not is_mono[precursor_index_]:
                continue
            precursor_index = indices[precursor_index_]
            precursor_mz = mz_values[tof_indices[precursor_index]]
            hits = 0
            for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
                for frame_offset in range(frame_count):
                    for scan_offset in range(-scan_tolerance, scan_tolerance + 1):
                        other_scan = self_scan + scan_offset
                        if not (0 <= other_scan < scan_max_index):
                            continue
                        other_push_offset = frame_offset * scan_max_index + other_scan
                        low_mz, high_mz = mz_windows[other_push_offset]
                        if not (low_mz <= precursor_mz < high_mz):
                            continue
                        other_push_index = push_offset + other_push_offset + cycle_length * cycle_offset
                        if not (0 <= other_push_index < len(indptr)):
                            continue
                        other_start = indptr[other_push_index]
                        other_end = indptr[other_push_index + 1]
                        for fragment_index_ in range(other_start, other_end):
                            fragment_index = indices[fragment_index_]
                            fragment_indices.append(fragment_index)
                            hits += 1
            if hits > 0:
                precursor_indices.append(precursor_index)
                precursor_count.append(hits)
    return (
        np.array(precursor_indices),
        np.array(precursor_count),
        np.array(fragment_indices),
    )


@alphatims.utils.njit(nogil=True)
def find_unfragmented_precursors(
    cycle_index,
    indices,
    indptr,
    zeroth_frame,
    scan_max_index,
    scan_tolerance,
    cycle_tolerance,
    mz_windows,
    tof_indices,
    tof_tolerance,
):
    cycle_length = len(mz_windows)
    frame_count = cycle_length // scan_max_index
    push_offset = cycle_length * cycle_index + zeroth_frame * scan_max_index
    unfragmented_precursor_indices = []
    for self_push_offset in np.flatnonzero(mz_windows[:, 0] == -1):
        self_push_index = push_offset + self_push_offset
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        self_scan = self_push_offset % scan_max_index
        for precursor_index_ in range(self_start, self_end):
            precursor_index = indices[precursor_index_]
            precursor_tof = tof_indices[precursor_index]
            for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
                for frame_offset in range(frame_count):
                    for scan_offset in range(-scan_tolerance, scan_tolerance + 1):
                        other_scan = self_scan + scan_offset
                        if not (0 <= other_scan < scan_max_index):
                            continue
                        other_push_offset = frame_offset * scan_max_index + other_scan
                        low_mz, high_mz = mz_windows[other_push_offset]
                        if low_mz == -1:
                            continue
                        other_push_index = push_offset + other_push_offset + cycle_length * cycle_offset
                        if not (0 <= other_push_index < len(indptr)):
                            continue
                        other_start = indptr[other_push_index]
                        other_end = indptr[other_push_index + 1]
                        for fragment_index_ in range(other_start, other_end):
                            fragment_index = indices[fragment_index_]
                            fragment_tof = tof_indices[fragment_index]
                            if np.abs(fragment_tof - precursor_tof) < tof_tolerance:
                                unfragmented_precursor_indices.append(fragment_index)
    return np.array(unfragmented_precursor_indices)


def annotate(
    iterable,
    frag_start_idx,
    frag_end_idx,
    frag_indices,
    frag_frequencies,
    indptr,
    mz_values,
    tof_indices,
    fragment_ppm,
    lower,
    upper,
    y_mzs,
    b_mzs,
    min_size,
    min_hit_count,
    top_n_hits,
):
    import multiprocessing

    def starfunc(index):
        # return alphadia.prefilter.annotate_pool(
        return alphadia.prefilter.annotate_pool2(
            index,
            frag_start_idx,
            frag_end_idx,
            frag_indices,
            frag_frequencies,
            indptr,
            mz_values,
            tof_indices,
            fragment_ppm,
            lower,
            upper,
            y_mzs,
            b_mzs,
            min_size,
            min_hit_count,
            top_n_hits,
        )
    precursor_indices = []
    max_hit_counts = []
    max_frequency_counts = []
    db_indices = []
    precursor_indptr = []
    with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
        for (
            precursor_index,
            hit_count,
            frequency_count,
            db_indices_,
        ) in alphatims.utils.progress_callback(
            pool.imap(starfunc, iterable),
            total=len(iterable),
            include_progress_callback=True
        ):
            # if hit_count >= min_hit_count:
            if True:
                precursor_indices.append(precursor_index)
                precursor_indptr.append(len(db_indices_))
                max_hit_counts.append(hit_count)
                max_frequency_counts.append(frequency_count)
                db_indices.append(db_indices_)
    return (
        np.array(precursor_indices),
        np.array(precursor_indptr),
        # np.array(max_hit_counts),
        np.concatenate(max_hit_counts),
        np.concatenate(max_frequency_counts),
        np.concatenate(db_indices),
    )


@alphatims.utils.pjit
# @alphatims.utils.njit(nogil=True)
def update_annotation(
    index,
    database_indices,
    database_frag_starts,
    database_frag_ends,
    database_y_mzs,
    database_b_mzs,
    database_y_ints,
    database_b_ints,
    inet_indices,
    precursor_indptr,
    fragment_indices,
    tof_indices,
    intensity_values,
    mz_values,
    fragment_ppm,
    b_hit_counts,
    y_hit_counts,
    b_mean_ppm,
    y_mean_ppm,
    relative_found_b_int,
    relative_missed_b_int,
    relative_found_y_int,
    relative_missed_y_int,
    relative_found_int,
    relative_missed_int,
    pearsons,
    pearsons_log,
    pseudo_int,
):
    if index >= len(database_indices):
        return
    database_index = database_indices[index]
    db_frag_start_idx = database_frag_starts[database_index]
    db_frag_end_idx = database_frag_ends[database_index]
    db_y_mzs = database_y_mzs[db_frag_start_idx: db_frag_end_idx][::-1]
    db_b_mzs = database_b_mzs[db_frag_start_idx: db_frag_end_idx]
    db_y_ints = database_y_ints[db_frag_start_idx: db_frag_end_idx][::-1]
    db_b_ints = database_b_ints[db_frag_start_idx: db_frag_end_idx]
    if pseudo_int > 0:
        db_y_ints = db_y_ints + pseudo_int
        db_b_ints = db_b_ints + pseudo_int
    precursor_index = inet_indices[index]
    frag_start_idx = precursor_indptr[precursor_index]
    frag_end_idx = precursor_indptr[precursor_index + 1]
    frags = fragment_indices[frag_start_idx: frag_end_idx]
    fragment_tofs = tof_indices[frags]
    order = np.argsort(fragment_tofs)
    fragment_mzs = mz_values[fragment_tofs][order]
    fragment_ints = intensity_values[frags][order]
    fragment_b_hits, db_b_hits = find_hits(
        fragment_mzs,
        db_b_mzs,
        fragment_ppm,
    )
    total_b_int = np.sum(db_b_ints)
    if total_b_int == 0:
        total_b_int = 1
    if len(db_b_hits) > 0:
        b_ppm = np.mean(
            (db_b_mzs[db_b_hits] - fragment_mzs[fragment_b_hits]) / db_b_mzs[db_b_hits] * 10**6
        )
        found_b_int = np.sum(db_b_ints[db_b_hits])
        min_b_int = np.min(db_b_ints[db_b_hits])
    else: # TODO defaults are not reflective of good/bad scores
        b_ppm = fragment_ppm
        found_b_int = 0
        min_b_int = -1
    fragment_y_hits, db_y_hits = find_hits(
        fragment_mzs,
        db_y_mzs,
        fragment_ppm,
    )
    total_y_int = np.sum(db_y_ints)
    if total_y_int == 0:
        total_y_int = 1
    if len(db_y_hits) > 0:
        y_ppm = np.mean(
            (db_y_mzs[db_y_hits] - fragment_mzs[fragment_y_hits]) / db_y_mzs[db_y_hits] * 10**6
        )
        found_y_int = np.sum(db_y_ints[db_y_hits])
        min_y_int = np.min(db_y_ints[db_y_hits])
    else: # TODO defaults are not reflective of good/bad scores
        y_ppm = fragment_ppm
        found_y_int = 0
        min_y_int = -1
    missed_b_int = np.sum(
        np.array([intsy for i, intsy in enumerate(db_b_ints) if (i not in db_b_hits) and (intsy > min_b_int)])
    )
    missed_y_int = np.sum(
        np.array([intsy for i, intsy in enumerate(db_y_ints) if (i not in db_y_hits) and (intsy > min_y_int)])
    )
    # all_frags = fragment_ints
    b_hit_counts[index] = len(db_b_hits)
    y_hit_counts[index] = len(db_y_hits)
    b_mean_ppm[index] = b_ppm
    y_mean_ppm[index] = y_ppm
    relative_found_b_int[index] = found_b_int / total_b_int
    relative_missed_b_int[index] = missed_b_int / total_b_int
    relative_found_y_int[index] = found_y_int / total_y_int
    relative_missed_y_int[index] = missed_y_int / total_y_int
    relative_found_int[index] = (found_b_int + found_y_int) / (total_b_int + total_y_int)
    relative_missed_int[index] = (missed_b_int + missed_y_int) / (total_b_int + total_y_int)
    all_db_ints = []
    all_frag_ints = []
    for b_int in db_b_ints[db_b_hits]:
        all_db_ints.append(b_int)
    for y_int in db_y_ints[db_y_hits]:
        all_db_ints.append(y_int)
    for frag_int in fragment_ints[fragment_b_hits]:
        all_frag_ints.append(frag_int)
    for frag_int in fragment_ints[fragment_y_hits]:
        all_frag_ints.append(frag_int)
    pearsons[index] = np.corrcoef(all_db_ints, all_frag_ints)[0, 1]
    pearsons_log[index] = np.corrcoef(
        np.log(np.array(all_db_ints)),
        np.log(np.array(all_frag_ints)),
    )[0, 1]

    # return (
    #     len(db_b_hits),
    #     len(db_y_hits),
    #     b_ppm,
    #     y_ppm,
    #     found_b_int / total_b_int,
    #     missed_b_int / total_b_int,
    #     found_y_int / total_y_int,
    #     missed_y_int / total_y_int,
    #     (found_b_int + found_y_int) / (total_b_int + total_y_int),
    #     (missed_b_int + missed_y_int) / (total_b_int + total_y_int),
    #     # pearson,
    # )


@alphatims.utils.njit(nogil=True)
def find_hits(
    fragment_mzs,
    database_mzs,
    fragment_ppm,
):
    fragment_index = 0
    database_index = 0
    fragment_hits = []
    db_hits = []
    while (fragment_index < len(fragment_mzs)) and (database_index < len(database_mzs)):
        fragment_mz = fragment_mzs[fragment_index]
        database_mz = database_mzs[database_index]
        if fragment_mz < (database_mz / (1 + 10**-6 * fragment_ppm)):
            fragment_index += 1
        elif database_mz < (fragment_mz / (1 + 10**-6 * fragment_ppm)):
            database_index += 1
        else:
            fragment_hits.append(fragment_index)
            db_hits.append(database_index)
            fragment_index += 1
            database_index += 1
    return np.array(fragment_hits), np.array(db_hits)


def quick_annotation_stats(analysis1, pseudo_int=10**-6):
    logging.info("Appending stats to quick annotation")
    b_hit_counts = np.zeros(len(analysis1.quick_annotation))
    y_hit_counts = np.zeros(len(analysis1.quick_annotation))
    b_mean_ppm = np.zeros(len(analysis1.quick_annotation))
    y_mean_ppm = np.zeros(len(analysis1.quick_annotation))
    relative_found_b_int = np.zeros(len(analysis1.quick_annotation))
    relative_missed_b_int = np.zeros(len(analysis1.quick_annotation))
    relative_found_y_int = np.zeros(len(analysis1.quick_annotation))
    relative_missed_y_int = np.zeros(len(analysis1.quick_annotation))
    relative_found_int = np.zeros(len(analysis1.quick_annotation))
    relative_missed_int = np.zeros(len(analysis1.quick_annotation))
    pearsons = np.zeros(len(analysis1.quick_annotation))
    pearsons_log = np.zeros(len(analysis1.quick_annotation))
    update_annotation(
        range(len(analysis1.quick_annotation)),
        analysis1.quick_annotation.db_index.values,
        analysis1.predicted_library_df.frag_start_idx.values,
        analysis1.predicted_library_df.frag_end_idx.values,
        analysis1.y_mzs,
        analysis1.b_mzs,
        analysis1.y_ions_intensities,
        analysis1.b_ions_intensities,
        analysis1.quick_annotation.inet_index.values,
        analysis1.precursor_indptr,
        analysis1.fragment_indices,
        analysis1.dia_data.tof_indices,
        # analysis1.dia_data.intensity_values,#.astype(np.float64),
        analysis1.smooth_intensity_values,#.astype(np.float64),
        analysis1.dia_data.mz_values * (1 + analysis1.ppm_mean * 10**-6),
        analysis1.ppm_width,
        b_hit_counts,
        y_hit_counts,
        b_mean_ppm,
        y_mean_ppm,
        relative_found_b_int,
        relative_missed_b_int,
        relative_found_y_int,
        relative_missed_y_int,
        relative_found_int,
        relative_missed_int,
        pearsons,
        pearsons_log,
        np.float32(pseudo_int),
    )
    analysis1.quick_annotation["b_hit_counts"] = b_hit_counts
    analysis1.quick_annotation["y_hit_counts"] = y_hit_counts
    analysis1.quick_annotation["b_mean_ppm"] = b_mean_ppm
    analysis1.quick_annotation["y_mean_ppm"] = y_mean_ppm
    analysis1.quick_annotation["relative_found_b_int"] = relative_found_b_int
    analysis1.quick_annotation["relative_missed_b_int"] = relative_missed_b_int
    analysis1.quick_annotation["relative_found_y_int"] = relative_found_y_int
    analysis1.quick_annotation["relative_missed_y_int"] = relative_missed_y_int
    analysis1.quick_annotation["relative_found_int"] = relative_found_int
    analysis1.quick_annotation["relative_missed_int"] = relative_missed_int
    pearsons[~np.isfinite(pearsons)] = 0
    analysis1.quick_annotation["pearsons"] = pearsons
    pearsons_log[~np.isfinite(pearsons_log)] = 0
    analysis1.quick_annotation["pearsons_log"] = pearsons_log


@alphatims.utils.pjit
# @alphatims.utils.njit(nogil=True)
def cluster_peaks(
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
    connected_ions,
    is_internal,
):
    len_dia_mz_cycle = len(connection_counts) - 1
    push_offset = len_dia_mz_cycle * cycle_index + zeroth_frame * scan_max_index
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        connection_end = connection_counts[self_connection_index + 1]
        if connection_end == connection_start:
            continue
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        for other_connection_index in connections[connection_start: connection_end]:
            other_push_index = push_offset + other_connection_index
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
                if is_internal[self_index] & is_internal[other_index]:
                    if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                        pointer = connected_ions[self_index]
                        to_merge = True
                        # print(self_index, other_index)
                        while pointer != self_index:
                            if pointer == other_index:
                                to_merge = False
                                break
                            pointer = connected_ions[pointer]
                        if to_merge:
                            connected_ions[self_index], connected_ions[other_index] = connected_ions[other_index], connected_ions[self_index]
                if self_tof < other_tof:
                    self_index += 1
                else:
                    other_index += 1
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        for self_index in range(self_start, self_end):
            pointer = connected_ions[self_index]
            if pointer == self_index:
                connected_ions[self_index] = 0
                # TODO
            else:
                while pointer > 0:
                    new_pointer = connected_ions[pointer]
                    connected_ions[pointer] = -self_index
                    pointer = new_pointer


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
        tof_sigma=3,
        cycle_sigma=3,
        scan_sigma=6,
    ):
        if isinstance(dia_data, str):
            dia_data = alphatims.bruker.TimsTOF(
                dia_data,
            )
        self.dia_data = dia_data
        self.tof_tolerance = tof_tolerance
        self.cycle_tolerance = cycle_tolerance
        self.scan_tolerance = scan_tolerance
        self.multiple_frames_per_cycle = multiple_frames_per_cycle
        self.ms1 = ms1
        self.ms2 = ms2
        self.cycle_sigma = cycle_sigma
        self.scan_sigma = scan_sigma
        self.tof_sigma = tof_sigma
        self.cycle_range = range(len(dia_data.push_indptr) // len(dia_data.dia_mz_cycle) + 1)
        self.cycle_length = len(dia_data.dia_mz_cycle) // dia_data.scan_max_index
        logging.info("Setting connections")
        self.connect()

    def connect(self):
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
        self.density_values = tm.zeros(
            self.dia_data.intensity_values.shape,
            dtype=np.float32
        )
        self.neighbor_types = tm.zeros(
            self.dia_data.intensity_values.shape,
            dtype=np.uint8
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
            self.neighbor_types,
            self.density_values,
            self.cycle_sigma,
            self.scan_sigma,
            self.tof_sigma,
        )
        self.smooth_intensity_values += self.dia_data.intensity_values

    def find_peaks(self):
        logging.info("Finding peaks")
        self.potential_peaks = tm.empty(
            shape=self.smooth_intensity_values.shape,
            dtype=np.bool_
        )
        self.valid_neighborhood = np.ones(2**8, dtype=np.bool_)
        for index in range(2**8):
            bin_repr = '{:08b}'.format(index)
            if "00" in bin_repr:
                self.valid_neighborhood[index] = False
            if (index < 2**7) and (index % 2 == 0):
                self.valid_neighborhood[index] = False
        self.potential_peaks[:] = self.valid_neighborhood[self.neighbor_types]
        find_seeds(
            range(len(self.dia_data.push_indptr) // len(self.dia_data.dia_mz_cycle) + 1),
            self.dia_data.push_indptr,
            self.dia_data.tof_indices,
            self.smooth_intensity_values,
            self.tof_tolerance,
            self.dia_data.scan_max_index,
            self.dia_data.zeroth_frame,
            self.connection_counts,
            self.connections,
            self.cycle_tolerance,
            self.potential_peaks,
        )
        self.peak_collection = PeakCollection(
            self.dia_data.push_indptr,
            self.potential_peaks,
        )

    def cluster(self):
        self.find_cluster_paths()
        self.assign_internal_points()
        self.cluster_from_paths()
        self.find_ambiguous_cluster_overlaps()
        self.assemble_clusters()
        self.assign_quantifiable_clusters()
        self.peak_collection = alphadia.smoothing.PeakCollection(
            self.dia_data.push_indptr,
            self.peaks,
        )

    def find_cluster_paths(self):
        logging.info("Finding cluster paths")
        self.cluster_path_pointers = tm.clone(np.arange(len(self.dia_data)))
        cluster_to_max_peaks_(
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
            self.cluster_path_pointers,
        )

    def cluster_from_paths(self):
        logging.info("Clustering from paths")
        self.cluster_pointers = tm.clone(self.cluster_path_pointers)
        walk_cluster_path(np.arange(10))
        walk_cluster_path(self.cluster_pointers)

    def find_ambiguous_cluster_overlaps(self):
        logging.info("Detecting cluster ambiguities")
        self.nonambiguous_ions = tm.ones(len(self.dia_data), dtype=np.bool_)
        find_unique_peaks_(
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
            self.cluster_pointers,
            self.nonambiguous_ions,
        )
        logging.info("Removing cluster ambiguities")
        walk_unique_cluster_path(
            np.arange(10),
            np.zeros(10, dtype=np.bool_),
            np.ones(10, dtype=np.bool_),
        )
        to_visit = np.ones_like(self.nonambiguous_ions)
        walk_unique_cluster_path(
            # range(len(to_visit)),
            self.cluster_path_pointers,
            self.nonambiguous_ions,
            to_visit,
        )

    def assemble_clusters(self):
        logging.info("Assembling clusters")
        self.cluster_assemblies = tm.clone(self.cluster_pointers)
        assemble_clusters(
            self.cluster_pointers,
            self.nonambiguous_ions,
            self.cluster_assemblies,
        )

    def assign_internal_points(self):
        logging.info("Assigning internal points")
        self.internal_points = tm.empty(
            shape=self.smooth_intensity_values.shape,
            dtype=np.bool_
        )
        self.valid_neighborhood = np.ones(2**8, dtype=np.bool_)
        for index in range(2**8):
            bin_repr = '{:08b}'.format(index)
            if "00" in bin_repr:
                self.valid_neighborhood[index] = False
            if (index < 2**7) and (index % 2 == 0):
                self.valid_neighborhood[index] = False
        self.internal_points[:] = self.valid_neighborhood[self.neighbor_types]

    def assign_quantifiable_clusters(self):
        logging.info("Assigning quantifiable clusters")
        unique_peaks = np.unique(self.cluster_pointers)
        self.peaks = unique_peaks[
            (self.nonambiguous_ions & self.internal_points)[unique_peaks]
        ]

    def create_precursor_centric_ion_network(self):
        import multiprocessing
        logging.info("Creating net")

        def starfunc(cycle_index):
            return create_precursor_centric_ion_network(
                cycle_index,
                self.peak_collection.indices,
                self.peak_collection.indptr,
                self.dia_data.zeroth_frame,
                self.dia_data.scan_max_index,
                self.scan_tolerance,
                self.cycle_tolerance,
                self.dia_data.dia_mz_cycle,
                self.dia_data.mz_values,
                self.dia_data.tof_indices,
                np.isin(self.peak_collection.indices, self.mono_isotopes)
            )

        precursor_indices = []
        precursor_counts = [[0]]
        fragment_indices = []

        with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
            for (
                precursor_indices_,
                precursor_counts_,
                fragment_indices_,
            ) in alphatims.utils.progress_callback(
                pool.imap(starfunc, self.cycle_range),
                total=len(self.cycle_range),
                include_progress_callback=True
            ):
                precursor_indices.append(precursor_indices_)
                precursor_counts.append(precursor_counts_)
                fragment_indices.append(fragment_indices_)

        # for cycle_index in alphatims.utils.progress_callback(self.cycle_range):
        #     (
        #         precursor_indices_,
        #         precursor_counts_,
        #         fragment_indices_,
        #     ) = create_precursor_centric_ion_network(
        #         cycle_index,
        #         self.peak_collection.indices,
        #         self.peak_collection.indptr,
        #         self.dia_data.zeroth_frame,
        #         self.dia_data.scan_max_index,
        #         self.scan_tolerance,
        #         self.cycle_tolerance,
        #         self.dia_data.dia_mz_cycle,
        #         self.dia_data.mz_values,
        #         self.dia_data.tof_indices,
        #         np.isin(self.peak_collection.indices, self.mono_isotopes)
        #     )
        #     precursor_indices.append(precursor_indices_)
        #     precursor_counts.append(precursor_counts_)
        #     fragment_indices.append(fragment_indices_)

        precursor_indices = np.concatenate(precursor_indices)
        precursor_counts = np.cumsum(np.concatenate(precursor_counts))
        fragment_indices = np.concatenate(fragment_indices)
        self.precursor_indices = tm.clone(precursor_indices)
        self.precursor_indptr = tm.clone(precursor_counts)
        self.fragment_indices = tm.clone(fragment_indices)
        self.set_fragment_weights()

    def find_unfragmented_precursors(self):
        unfragmented_precursors = []

        for cycle_index in alphatims.utils.progress_callback(self.cycle_range):
            unfragmented_precursors_ = find_unfragmented_precursors(
                cycle_index,
                self.peak_collection.indices,
                self.peak_collection.indptr,
                self.dia_data.zeroth_frame,
                self.dia_data.scan_max_index,
                self.scan_tolerance,
                self.cycle_tolerance,
                self.dia_data.dia_mz_cycle,
                self.dia_data.tof_indices,
                self.tof_tolerance,
            )
            unfragmented_precursors.append(unfragmented_precursors_)

        unfragmented_precursors = np.concatenate(unfragmented_precursors)
        self.unfragmented_precursors = tm.clone(unfragmented_precursors)

    def get_inet_counts(self):
        self.inet_indptr = tm.zeros(
            self.dia_data.intensity_values.shape,
            dtype=np.int64
        )
        inet_counts(
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
            self.inet_indptr,
            self.potential_peaks,
        )

    def determine_mono_isotopes(self, isotope_mz_tolerance=0.01):
        logging.info("Determining mono isotopes")
        self.isotope_mz_tolerance = isotope_mz_tolerance
        logging.info("Charge 2")
        left_connection, right_connection = create_isotopic_pairs(
            self,
            difference=1/2,
            mz_tolerance=isotope_mz_tolerance,
        )
        self.mono_isotopes_charge2 = tm.clone(
            self.peak_collection.indices[
                left_connection[
                    ~np.isin(
                        left_connection, right_connection
                    ) & np.isin(
                        right_connection, left_connection
                    )
                ]
            ]
        )
        logging.info("Charge 3")
        left_connection, right_connection = create_isotopic_pairs(
            self,
            difference=1/3,
            mz_tolerance=isotope_mz_tolerance,
        )
        self.mono_isotopes_charge3 = tm.clone(
            self.peak_collection.indices[
                left_connection[
                    ~np.isin(
                        left_connection, right_connection
                    ) & np.isin(
                        right_connection, left_connection
                    )
                ]
            ]
        )
        self.mono_isotopes = np.unique(
            np.concatenate(
                [
                    self.mono_isotopes_charge2,
                    self.mono_isotopes_charge3,
                ]
            )
        )

    def set_fragment_weights(self):
        logging.info("Setting fragment weights")
        dia_data = self.dia_data
        fdf = pd.DataFrame(
            dia_data.convert_from_indices(
                self.fragment_indices,
                return_scan_indices=True,
                return_push_indices=True,
            )
        )
        pdf = pd.DataFrame(
            dia_data.convert_from_indices(
                np.repeat(
                    self.precursor_indices,
                    np.diff(self.precursor_indptr)
                ),
                return_scan_indices=True,
                return_push_indices=True,
            )
        )
        pdf["cycle"] = (pdf.push_indices - dia_data.zeroth_frame * dia_data.scan_max_index) // dia_data.dia_mz_cycle.shape[0]
        fdf["cycle"] = (fdf.push_indices - dia_data.zeroth_frame * dia_data.scan_max_index) // dia_data.dia_mz_cycle.shape[0]
        self.fragment_frequencies = (
            np.exp(
                -((pdf.scan_indices - fdf.scan_indices) / self.scan_sigma)**2 / 2
            ) * np.exp(
                -((pdf.cycle - fdf.cycle) / self.cycle_sigma)**2 / 2
            )
        ).values

    def add_library(self, library_file_name):
        logging.info("Loading library")
        self.library_file_name = library_file_name
        self.lib = alphabase.io.hdf.HDF_File(
            self.library_file_name
        #     read_only=False
        )

        predicted_library_df = self.lib.library.precursor_df[...]
        # predicted_library_df.sort_values(by=["rt_pred", "mobility_pred"], inplace=True)
        predicted_library_df.sort_values(by="precursor_mz", inplace=True)
        predicted_library_df.reset_index(level=0, inplace=True)
        predicted_library_df.rename(columns={"index": "original_index"}, inplace=True)
        predicted_library_df.decoy = predicted_library_df.decoy.astype(np.bool_)

        self.y_mzs = self.lib.library.fragment_mz_df.y_z1.mmap
        self.b_mzs = self.lib.library.fragment_mz_df.b_z1.mmap
        self.y_ions_intensities = self.lib.library.fragment_intensity_df.y_z1.mmap
        self.b_ions_intensities = self.lib.library.fragment_intensity_df.b_z1.mmap

        self.predicted_library_df = predicted_library_df

    def quick_annotate(
        self,
        precursor_ppm=50,
        fragment_ppm=50,
        min_size=10,
        ppm_mean=0,
        min_hit_count=1,
        append_stats=True,
        top_n_hits=1,
    ):
        logging.info(f"Quick library annotation of mono isotopes with {ppm_mean=} and {precursor_ppm=}")
        o = np.argsort(self.dia_data.tof_indices[self.precursor_indices])
        mz_values = self.dia_data.mz_values * (1 + ppm_mean * 10**-6)
        p_mzs = mz_values[
            self.dia_data.tof_indices[self.precursor_indices][o]
        ]
        lower = np.empty(len(self.precursor_indices), dtype=np.int64)
        upper = np.empty(len(self.precursor_indices), dtype=np.int64)
        lower[o] = np.searchsorted(
            self.predicted_library_df.precursor_mz.values,
            p_mzs / (1 + precursor_ppm * 10**-6)
        )
        upper[o] = np.searchsorted(
            self.predicted_library_df.precursor_mz.values,
            p_mzs * (1 + precursor_ppm * 10**-6)
        )
        logging.info(
            f"PSMs to test: {np.sum(((upper - lower) * (np.diff(self.precursor_indptr) >= min_size)))}"
        )
        (
            precursor_indices,
            precursor_indptr,
            hit_counts,
            frequency_counts,
            db_indices,
        ) = annotate(
            range(len(lower)),
            self.predicted_library_df.frag_start_idx.values,
            self.predicted_library_df.frag_end_idx.values,
            self.fragment_indices,
            self.fragment_frequencies,
            self.precursor_indptr,
            mz_values,
            self.dia_data.tof_indices,
            fragment_ppm,
            lower,
            upper,
            self.y_mzs,
            self.b_mzs,
            min_size,
            min_hit_count,
            top_n_hits,
        )

        precursor_selection = np.repeat(precursor_indices, precursor_indptr)
        hits = self.dia_data.as_dataframe(self.precursor_indices[precursor_selection])
        hits["inet_index"] = precursor_selection
        hits["candidates"] = (upper - lower)[precursor_selection]
        hits["total_peaks"] = np.diff(self.precursor_indptr)[precursor_selection]
        hits["db_index"] = db_indices.astype(np.int64)
        # hits["counts"] = np.repeat(hit_counts, precursor_indptr)
        hits["counts"] = hit_counts
        hits["frequency_counts"] = frequency_counts
        self.quick_annotation = hits
        self.quick_annotation["smooth_intensity"] = self.smooth_intensity_values[
            self.quick_annotation.raw_indices
        ]
        self.quick_annotation = self.quick_annotation.join(self.predicted_library_df, on="db_index")
        self.quick_annotation["im_diff"] = self.quick_annotation.mobility_pred - self.quick_annotation.mobility_values
        self.quick_annotation["mz_diff"] = self.quick_annotation.precursor_mz - self.quick_annotation.mz_values
        self.quick_annotation["ppm_diff"] = self.quick_annotation.mz_diff / self.quick_annotation.precursor_mz * 10**6
        self.quick_annotation["target"] = ~self.quick_annotation.decoy
        self.quick_annotation.reset_index(drop=True, inplace=True)
        if append_stats:
            quick_annotation_stats(self)

    def estimate_mz_tolerance(self):
        ppm_diffs = self.quick_annotation.ppm_diff
        order = np.argsort(ppm_diffs.values)

        decoys, targets = np.bincount(self.quick_annotation.decoy.values)
        distribution = np.cumsum(
            [
                1 / targets if i else -1 / decoys for i in self.quick_annotation.decoy.values[order]
            ]
        )
        low = ppm_diffs[order[np.argmin(distribution)]]
        high = ppm_diffs[order[np.argmax(distribution)]]
        self.ppm_mean = (low + high) / 2
        self.ppm_width = abs(high - low)
        # plt.plot(
        #     ppm_diffs[order],
        #     distribution,
        # )
        # sns.histplot(
        #     data=self.quick_annotation,
        #     x="ppm_diff",
        #     hue="decoy",
        # )

    def quick_calibration(
        self,
        fdr=0.01,
        train_fdr_level_pre_calibration=0.1,
        train_fdr_level_post_calibration=0.1,
        n_neighbors=4,
        test_size=0.8,
        random_state=0,
    ):
        val_names = [
            "counts",
            "frequency_counts",
            "ppm_diff",
            "im_diff",
            "charge",
            "total_peaks",
            "nAA",
            "b_hit_counts",
            "y_hit_counts",
            "b_mean_ppm",
            "y_mean_ppm",
            "relative_found_b_int",
            "relative_missed_b_int",
            "relative_found_y_int",
            "relative_missed_y_int",
            "relative_found_int",
            "relative_missed_int",
            "pearsons",
            "pearsons_log",
            "candidates",
        ]
        logging.info("Calculating quick log odds")
        score_df = self.quick_annotation.copy()
        log_odds = calculate_log_odds_product(
            score_df,
            val_names,
        )
        score_df["log_odds"] = log_odds
        # score_df = alphadia.prefilter.train_and_score(
        #     score_df,
        #     val_names,
        #     ini_score="log_odds",
        #     train_fdr_level=train_fdr_level_pre_calibration,
        # ).reset_index(drop=True)
        score_df = alphadia.library.get_q_values(score_df, "log_odds", 'decoy', drop=True)
        score_df_above_fdr = score_df[
            (score_df.q_value < fdr) & (score_df.target)
        ].reset_index(drop=True)
        logging.info(f"Found {len(score_df_above_fdr)} targets for calibration")
        score_df_above_fdr["im_pred"] = score_df_above_fdr.mobility_pred
        score_df_above_fdr["im_values"] = score_df_above_fdr.mobility_values
        self.predictors = {}
        for dimension in ["rt", "im"]:
            X = score_df_above_fdr[f"{dimension}_pred"].values.reshape(-1, 1)
            y = score_df_above_fdr[f"{dimension}_values"].values
            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = sklearn.model_selection.train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
            self.predictors[dimension] = sklearn.neighbors.KNeighborsRegressor(
                n_neighbors=n_neighbors,
                # weights="distance",
                n_jobs=alphatims.utils.set_threads(alphatims.utils.MAX_THREADS)
            )
            self.predictors[dimension].fit(X_train, y_train)
            score_df_above_fdr[f"{dimension}_calibrated"] = self.predictors[dimension].predict(
                score_df_above_fdr[f"{dimension}_pred"].values.reshape(-1, 1)
            )
            score_df_above_fdr[f"{dimension}_diff"] = score_df_above_fdr[f"{dimension}_values"] - score_df_above_fdr[f"{dimension}_calibrated"]
        score_df["rt_calibrated"] = self.predictors["rt"].predict(
            score_df.rt_pred.values.reshape(-1, 1)
        )
        score_df["im_calibrated"] = self.predictors["im"].predict(
            score_df.mobility_pred.values.reshape(-1, 1)
        )
        ppm_mean = np.mean(score_df_above_fdr.ppm_diff.values)
        score_df["mz_calibrated"] = score_df.precursor_mz * (
            1 - ppm_mean * 10**-6
        )

        score_df["ppm_diff_calibrated"] = (score_df.mz_calibrated - score_df.mz_values) / score_df.mz_calibrated * 10**6
        score_df["rt_diff_calibrated"] = score_df.rt_calibrated - score_df.rt_values
        score_df["im_diff_calibrated"] = score_df.im_calibrated - score_df.mobility_values
        self.score_df = alphadia.prefilter.train_and_score(
            # score_df[np.abs(score_df.rt_diff_calibrated) < 250].reset_index(drop=True),
            score_df,
            [
                "counts",
                "frequency_counts",
                "ppm_diff_calibrated",
                "im_diff_calibrated",
                "rt_diff_calibrated",
                "charge",
                "total_peaks",
                "nAA",
                "b_hit_counts",
                "y_hit_counts",
                "b_mean_ppm",
                "y_mean_ppm",
                "relative_found_b_int",
                "relative_missed_b_int",
                "relative_found_y_int",
                "relative_missed_y_int",
                "relative_found_int",
                "relative_missed_int",
                "pearsons",
                "pearsons_log",
                "candidates",
                # "log_odds",
            ],
            ini_score="log_odds",
            train_fdr_level=train_fdr_level_post_calibration,
        ).reset_index(drop=True)

        self.score_df["target_type"] = np.array([-1, 0])[
            self.score_df.target.astype(np.int)
        ]
        self.score_df["target_type"][
            (self.score_df.q_value < fdr) & (self.score_df.target)
        ] = 1

    def preprocess(
        self
    ):
        self.smooth()
        self.find_peaks()
        self.determine_mono_isotopes()
        self.create_precursor_centric_ion_network()
        # self.save()

    def save(self):
        hdf = alphabase.io.hdf.HDF_File(
            f"sandbox_{self.dia_data.sample_name}_analysis.hdf",
            read_only=False,
            truncate=True,
        )
        hdf.preprocessing = {
            "smooth_intensity_values": self.smooth_intensity_values,
            "neighbor_types": self.neighbor_types,
            "density_values": self.density_values,
            "potential_peaks": self.potential_peaks,
            "peak_collection": {
                "indptr": self.peak_collection.indptr,
                "indices": self.peak_collection.indices,
            },
            "isotopes": {
                "mono_isotopes_charge2": self.mono_isotopes_charge2,
                "mono_isotopes_charge3": self.mono_isotopes_charge3,
                "mono_isotopes": self.mono_isotopes,
            },
            "pseudo_msms_spectra": {
                "precursor_indices": self.precursor_indices,
                "precursor_indptr": self.precursor_indptr,
                "fragment_indices": self.fragment_indices,
            },
            "connections": {
                "connection_counts": self.connection_counts,
                "connections": self.connections,
            },
            "tof_tolerance": self.tof_tolerance,
            "cycle_tolerance": self.cycle_tolerance,
            "scan_tolerance": self.scan_tolerance,
            "multiple_frames_per_cycle": self.multiple_frames_per_cycle,
            "ms1": self.ms1,
            "ms2": self.ms2,
            "cycle_sigma": self.cycle_sigma,
            "scan_sigma": self.scan_sigma,
            "tof_sigma": self.tof_sigma,
            # "cycle_range": self.cycle_range,
            "cycle_length": self.cycle_length,
        }
        hdf.annotation = {
            "fragment_frequencies": self.fragment_frequencies,
            "quick_annotation": self.quick_annotation,
            "score_df": self.score_df,
            "ppm_width": self.ppm_width,
            "ppm_mean": self.ppm_mean,
        }


class PeakCollection(object):

    def __init__(
        self,
        indptr: np.ndarray,
        peaks: np.ndarray,
    ):
        if peaks.dtype == np.bool_:
            self.indices = tm.clone(np.flatnonzero(peaks))
        else:
            self.indices = tm.clone(peaks)
        self.indptr = tm.empty(indptr.shape, indptr.dtype)
        set_peak_indptr(indptr, self.indptr, self.indices)


@alphatims.utils.njit
def set_peak_indptr(old_indptr, new_indptr, indices):
    count = 0
    offset = 0
    for index in indices:
        while index >= old_indptr[offset]:
            new_indptr[offset] = count
            offset += 1
        count += 1
    while index >= old_indptr[offset]:
        new_indptr[offset] = count
        offset += 1
    new_indptr[offset:] = count


@alphatims.utils.pjit
# @alphatims.utils.njit(nogil=True)
def match(
    index,
    indices1,
    indices2,
    indptr1,
    indptr2,
    fragments1,
    fragments2,
    tof_indices1,
    tof_indices2,
    mz_values1,
    mz_values2,
    fragment_ppm,
    overlaps,
    fragment_hits1,
    fragment_hits2,
):
    precursor1 = indices1[index]
    start1 = indptr1[precursor1]
    end1 = indptr1[precursor1 + 1]
    frags1 = fragments1[start1: end1]
    tofs1 = tof_indices1[frags1]
    mzs1 = mz_values1[tofs1]
    order1 = np.argsort(mzs1)
    precursor2 = indices2[index]
    start2 = indptr2[precursor2]
    end2 = indptr2[precursor2 + 1]
    frags2 = fragments2[start2: end2]
    tofs2 = tof_indices2[frags2]
    mzs2 = mz_values2[tofs2]
    order2 = np.argsort(mzs2)
    index1 = 0
    index2 = 0
    hits = 0
    while (index1 < len(mzs1)) and (index2 < len(mzs2)):
        fragment_mz = mzs1[order1[index1]]
        database_mz = mzs2[order2[index2]]
        if fragment_mz < (database_mz / (1 + 10**-6 * fragment_ppm)):
            index1 += 1
        elif database_mz < (fragment_mz / (1 + 10**-6 * fragment_ppm)):
            index2 += 1
        else:
            hits += 1
            fragment_hits1[start1 + order1[index1]] = True
            fragment_hits2[start2 + order2[index2]] = True
            index1 += 1
            index2 += 1
    overlaps[index] = hits



@alphatims.utils.njit(nogil=True)
def rough_match2_count_only(
    fragment_mzs,
    database_mzs,
    fragment_ppm,
):
    fragment_index = 0
    database_index = 0
    hits = 0
    while (fragment_index < len(fragment_mzs)) and (database_index < len(database_mzs)):
        fragment_mz = fragment_mzs[fragment_index]
        database_mz = database_mzs[database_index]
        if fragment_mz < (database_mz / (1 + 10**-6 * fragment_ppm)):
            fragment_index += 1
        elif database_mz < (fragment_mz / (1 + 10**-6 * fragment_ppm)):
            database_index += 1
        else:
            hits += 1
            fragment_index += 1
            database_index += 1
    return hits


def align(
    analysis1,
    analysis2,
    ppm=30,
    fragment_ppm=30
):
    logging.info("Aligning samples")
    df1 = analysis1.dia_data.as_dataframe(
        analysis1.precursor_indices
    )
    df1.sort_values(by="mz_values", inplace=True)
    df1.reset_index(inplace=True)
    df2 = analysis2.dia_data.as_dataframe(
        analysis2.precursor_indices
    )
    df2.sort_values(by="mz_values", inplace=True)
    df2.reset_index(inplace=True)
    mz1 = df1.mz_values.values
    mz2 = df2.mz_values.values
    lower = np.searchsorted(mz1, mz2 / (1 + ppm*10**-6))
    upper = np.searchsorted(mz1, mz2 * (1 + ppm*10**-6))
    indices2 = np.repeat(df2["index"].values, upper - lower)
    indices1 = np.concatenate(
        [
            df1["index"].values[l:h] for l, h in zip(lower, upper)
        ]
    )
    overlaps = tm.empty(len(indices1), dtype=np.int16)
    fragment_hits1 = tm.zeros(len(analysis1.fragment_indices), dtype=np.bool_)
    fragment_hits2 = tm.zeros(len(analysis2.fragment_indices), dtype=np.bool_)
    match(
        range(len(overlaps)),
        indices1,
        indices2,
        analysis1.precursor_indptr,
        analysis2.precursor_indptr,
        analysis1.fragment_indices,
        analysis2.fragment_indices,
        analysis1.dia_data.tof_indices,
        analysis2.dia_data.tof_indices,
        analysis1.dia_data.mz_values,
        analysis2.dia_data.mz_values,
        fragment_ppm,
        overlaps,
        fragment_hits1,
        fragment_hits2,
    )
    alignment_indptr = np.empty(len(upper) + 1, dtype=np.int64)
    alignment_indptr[1:] = np.cumsum(upper - lower)
    alignment_indptr[0] = 0
    best = np.array(
        [
            start + np.argmax(overlaps[start:end]) for start, end in zip(
                alignment_indptr[:-1],
                alignment_indptr[1:]
            ) if end > start
        ]
    )
    overlaps = tm.empty(len(indices1), dtype=np.int16)
    fragment_hits1 = tm.zeros(len(analysis1.fragment_indices), dtype=np.bool_)
    fragment_hits2 = tm.zeros(len(analysis2.fragment_indices), dtype=np.bool_)
    match(
        range(len(best)),
        indices1[best],
        indices2[best],
        analysis1.precursor_indptr,
        analysis2.precursor_indptr,
        analysis1.fragment_indices,
        analysis2.fragment_indices,
        analysis1.dia_data.tof_indices,
        analysis2.dia_data.tof_indices,
        analysis1.dia_data.mz_values,
        analysis2.dia_data.mz_values,
        fragment_ppm,
        overlaps,
        fragment_hits1,
        fragment_hits2,
    )
    return fragment_hits1, fragment_hits2


def run_flow(file_name):
    analysis1 = alphadia.smoothing.Analysis(
        file_name,
        tof_tolerance=3,
        cycle_tolerance=3,
        scan_tolerance=6,
        multiple_frames_per_cycle=False,
        ms1=True,
        ms2=True,
        tof_sigma=3,
        cycle_sigma=3,
        scan_sigma=6,
    )
    analysis1.preprocess()
    analysis1.fragment_frequencies = np.ones(len(analysis1.fragment_indices))
    analysis1.add_library(
        "/Users/swillems/Data/peptide_centric/FZW_predicted_spec_libs/human_reviewed_fasta_regular_w_decoy.speclib.hdf"
    )
    analysis1.quick_annotate(
        precursor_ppm=50,
        fragment_ppm=50,
        min_size=5,
        min_hit_count=3,
        append_stats=False,
    )
    analysis1.estimate_mz_tolerance()
    analysis1.quick_annotate(
        precursor_ppm=analysis1.ppm_width,
        fragment_ppm=analysis1.ppm_width,
        ppm_mean=analysis1.ppm_mean,
        min_size=5,
        min_hit_count=3,
    )
    fdr = 0.01
    analysis1.quick_calibration(
        fdr=fdr,
        train_fdr_level_pre_calibration=1,
        train_fdr_level_post_calibration=0.1,
        n_neighbors=4,
        test_size=0.8,
        random_state=0,
    )
    new_lib = analysis1.score_df[
        (analysis1.score_df.q_value < fdr) & (analysis1.score_df.target)
    ]
    return new_lib


def calculate_odds(df, column_name, *, target_name="target", smooth=1, plot=False):
    order = np.argsort(df[column_name].values)
    negatives, positives = np.bincount(df.target.values)
    tp_count = positives - negatives
    n = int(tp_count * smooth)
    forward = np.cumsum(df[target_name].values[order])
    odds = np.zeros_like(forward, dtype=np.float)
    odds[n:-n] = forward[2*n:] - forward[:-2*n]
    odds[:n] = forward[n:2*n]
    odds[-n:] = forward[-1] - forward[-2*n:-n]
    odds[n:-n] /= 2*n
    odds[:n] /= np.arange(n, 2*n)
    odds[-n:] /= np.arange(n, 2*n)[::-1]
    odds /= (1 - odds)
    odds = odds[np.argsort(order)]
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(df[column_name], odds, marker=".")
    return odds


def calculate_log_odds_product(
    df_,
    val_names
):
    df = df_[val_names]
    df = sklearn.preprocessing.StandardScaler().fit_transform(df)
    pca = sklearn.decomposition.PCA(n_components=df.shape[1])
    pca.fit(df)
    df = pd.DataFrame(pca.transform(df))
    df["target"] = df_.target
    negative, positive = np.bincount(df.target)
    log_odds = np.zeros(len(df))
    for val_name in range(df.shape[1] - 1):
        odds = alphadia.smoothing.calculate_odds(df, val_name, smooth=1)
        log_odds += np.log2(odds) * pca.explained_variance_[val_name]
    return log_odds
    # new_df = analysis1.score_df[["decoy", "target"]]
    # new_df['odds'] = log_odds
    # new_df = alphadia.library.get_q_values(new_df, "odds", 'decoy', drop=True)
    # new_df.reset_index(drop=True, inplace=True)


def deconvolute_frame_groups(
    analysis1,
    ppm=20,
    tolerance=1,
):
    import multiprocessing

    def starfunc(index):
        return deconvolute_frame_groups_(
            index,
            analysis1,
            ppm,
            tolerance,
        )
    fragment_indices = []
    precursor_counts = 0
    precursor_indptr = [precursor_counts]
    # iterable = range(100)
    iterable = range(len(analysis1.precursor_indices))
    with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
        for reproducible_fragments in alphatims.utils.progress_callback(
            pool.imap(starfunc, iterable),
            total=len(iterable),
            include_progress_callback=True
        ):
            fragment_indices.append(reproducible_fragments)
            precursor_counts += len(reproducible_fragments)
            precursor_indptr.append(precursor_counts)
    return (
        tm.clone(np.array(precursor_indptr)),
        np.concatenate(fragment_indices),
    )


def deconvolute_frame_groups_(
    prec_index,
    analysis1,
    ppm,
    tolerance
):
    start = analysis1.precursor_indptr[prec_index]
    end = analysis1.precursor_indptr[prec_index + 1]
    frags = analysis1.fragment_indices[start:end]
    # df = dia_data.as_dataframe(frags)
    # df.sort_values(by="mz_values", inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # # df["cycle"] = (df.push_indices - dia_data.zeroth_frame * dia_data.scan_max_index) // dia_data.dia_mz_cycle.shape[0]
    # # df["frame_group"] = df.precursor_indices
    # to_keep = match_frame_groups(
    #     df.mz_values.values,
    #     df.precursor_indices.values,
    #     df.intensity_values.values,
    #     df.raw_indices.values,
    #     ppm,
    # )
    coordinates = analysis1.dia_data.convert_from_indices(
         frags,
         return_raw_indices=True,
         return_precursor_indices=True,
         return_mz_values=True,
         return_intensity_values=True,
         raw_indices_sorted=True,
    )
    order = np.argsort(coordinates["mz_values"])
    to_keep = match_frame_groups(
        coordinates["mz_values"][order],
        coordinates["precursor_indices"][order],
        coordinates["intensity_values"][order],
        coordinates["raw_indices"][order],
        ppm,
        tolerance,
    )
    return to_keep


@alphatims.utils.njit(nogil=True)
def match_frame_groups(
    # df,
    mz_values,
    frame_groups,
    intensity_values,
    raw_indices,
    ppm,
    tolerance
):
    unique_frame_groups = len(np.unique(frame_groups))
    if tolerance < 1:
        tolerance *= unique_frame_groups
    index2 = 0
    to_keep = []
    for index1, mz1 in enumerate(mz_values[:-1]):
        if index1 < index2:
            continue
        prev_mz = mz1
        for index2, mz2 in enumerate(mz_values[index1 + 1:], index1 + 1):
            if ((mz2 - prev_mz) / prev_mz) * 10**6 > ppm:
                break
            prev_mz = mz2
        detected_frame_groups = np.unique(frame_groups[index1: index2])
        if (unique_frame_groups - len(detected_frame_groups)) <= tolerance:
            max_intensity = np.argmax(intensity_values[index1: index2])
            raw_index = raw_indices[index1: index2][max_intensity]
            to_keep.append(raw_index)
    to_keep = np.sort(np.array(to_keep))
    return to_keep



@alphatims.utils.njit(nogil=True)
def match_frame_groups_frequency(
    # df,
    mz_values,
    frame_groups,
    intensity_values,
    raw_indices,
    ppm,
    tolerance
):
    to_keep = []
    index2 = 0
    for index1, mz1 in enumerate(mz_values[:-1]):
        if index1 < index2:
            continue
        prev_mz = mz1
        for index2, mz2 in enumerate(mz_values[index1 + 1:], index1 + 1):
            if ((mz2 - prev_mz) / prev_mz) * 10**6 > ppm:
                break
            prev_mz = mz2
        detected_frame_groups = len(np.unique(frame_groups[index1: index2]))
        for index in range(index1, index2):
            to_keep.append(detected_frame_groups)
    if index2 != len(mz_values):
        to_keep.append(1)
    return np.array(to_keep)


def deconvolute_frame_groups_frequencies_(
    prec_index,
    analysis1,
    ppm,
    tolerance
):
    start = analysis1.precursor_indptr[prec_index]
    end = analysis1.precursor_indptr[prec_index + 1]
    frags = analysis1.fragment_indices[start:end]
    coordinates = analysis1.dia_data.convert_from_indices(
         frags,
         return_raw_indices=True,
         return_precursor_indices=True,
         return_mz_values=True,
         return_intensity_values=True,
         raw_indices_sorted=True,
    )
    frequencies = np.empty(len(frags), dtype=np.int64)
    order = np.argsort(coordinates["mz_values"])
    frequencies[order] = match_frame_groups_frequency(
        coordinates["mz_values"][order],
        coordinates["precursor_indices"][order],
        coordinates["intensity_values"][order],
        coordinates["raw_indices"][order],
        ppm,
        tolerance,
    )
    return frequencies


def deconvolute_frame_groups_frequencies(
    analysis1,
    ppm=20,
    tolerance=1,
):
    import multiprocessing

    def starfunc(index):
        return deconvolute_frame_groups_frequencies_(
            index,
            analysis1,
            ppm,
            tolerance,
        )
    fragment_indices = []
    # iterable = range(100)
    iterable = range(len(analysis1.precursor_indices))
    with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
        for reproducible_fragments in alphatims.utils.progress_callback(
            pool.imap(starfunc, iterable),
            total=len(iterable),
            include_progress_callback=True
        ):
            fragment_indices.append(reproducible_fragments)
    return np.concatenate(fragment_indices)


@alphatims.utils.njit(nogil=True)
def match_ms1_to_ms2_(
    cycle_index,
    indices,
    indptr,
    zeroth_frame,
    scan_max_index,
    ppm_tolerance,
    scan_tolerance,
    cycle_tolerance,
    mz_windows,
    mz_values,
    tof_indices,
    is_mono,
):
    cycle_length = len(mz_windows)
    frame_count = cycle_length // scan_max_index
    push_offset = cycle_length * cycle_index + zeroth_frame * scan_max_index
    precursor_indices = []
    precursor_count = []
    fragment_indices = []
    for self_push_offset in np.flatnonzero(mz_windows[:, 0] == -1):
        self_push_index = push_offset + self_push_offset
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        self_scan = self_push_offset % scan_max_index
        for precursor_index_ in range(self_start, self_end):
            if not is_mono[precursor_index_]:
                continue
            precursor_index = indices[precursor_index_]
            precursor_mz = mz_values[tof_indices[precursor_index]]
            hits = 0
            for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
                for frame_offset in range(frame_count):
                    for scan_offset in range(-scan_tolerance, scan_tolerance + 1):
                        other_scan = self_scan + scan_offset
                        if not (0 <= other_scan < scan_max_index):
                            continue
                    # for other_scan in range(scan_max_index):
                        other_push_offset = frame_offset * scan_max_index + other_scan
                        low_mz, high_mz = mz_windows[other_push_offset]
                        if low_mz == -1:
                            continue
                        other_push_index = push_offset + other_push_offset + cycle_length * cycle_offset
                        if not (0 <= other_push_index < len(indptr)):
                            continue
                        other_start = indptr[other_push_index]
                        other_end = indptr[other_push_index + 1]
                        for fragment_index_ in range(other_start, other_end):
                            fragment_index = indices[fragment_index_]
                            fragment_mz = mz_values[tof_indices[fragment_index]]
                            # if np.abs(fragment_mz - precursor_mz) / precursor_mz * 10**6 < ppm_tolerance:
                            if is_within_ppm_tolerance(fragment_mz, precursor_mz, ppm_tolerance):
                                fragment_indices.append(fragment_index)
                                hits += 1
            if hits > 0:
                precursor_indices.append(precursor_index)
                precursor_count.append(hits)
    return (
        np.array(precursor_indices),
        np.array(precursor_count),
        np.array(fragment_indices),
    )


@alphatims.utils.njit(nogil=True)
def is_within_ppm_tolerance(mz1, mz2, ppm):
    return np.abs(mz1 - mz2) / mz2 * 10**6 < ppm



def match_ms1_to_ms2(self, ppm):
    import multiprocessing
    logging.info("Matching precursors")

    def starfunc(cycle_index):
        return match_ms1_to_ms2_(
            cycle_index,
            self.peak_collection.indices,
            self.peak_collection.indptr,
            self.dia_data.zeroth_frame,
            self.dia_data.scan_max_index,
            ppm,
            self.scan_tolerance,
            self.cycle_tolerance,
            self.dia_data.dia_mz_cycle,
            self.dia_data.mz_values,
            self.dia_data.tof_indices,
            np.isin(self.peak_collection.indices, self.mono_isotopes)
        )

    precursor_indices = []
    precursor_counts = [[0]]
    fragment_indices = []

    with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
        for (
            precursor_indices_,
            precursor_counts_,
            fragment_indices_,
        ) in alphatims.utils.progress_callback(
            pool.imap(starfunc, self.cycle_range),
            total=len(self.cycle_range),
            include_progress_callback=True
        ):
            precursor_indices.append(precursor_indices_)
            precursor_counts.append(precursor_counts_)
            fragment_indices.append(fragment_indices_)

    precursor_indices = np.concatenate(precursor_indices)
    precursor_counts = np.cumsum(np.concatenate(precursor_counts))
    fragment_indices = np.concatenate(fragment_indices)
    return (
        tm.clone(precursor_indices),
        tm.clone(precursor_counts),
        tm.clone(fragment_indices),
    )
    # self.precursor_indices = tm.clone(precursor_indices)
    # self.precursor_indptr = tm.clone(precursor_counts)
    # self.fragment_indices = tm.clone(fragment_indices)


@alphatims.utils.njit
def sort_query_data_fragments_by_mz(indptr, mz_values, intensities):
    for index, start in enumerate(indptr[:-1]):
        end = indptr[index + 1]
        mzs = mz_values[start: end]
        order = np.argsort(mzs)
        mz_values[start:end] = mz_values[start:end][order]
        intensities[start:end] = intensities[start:end][order]


def create_ap_like_query_data(analysis1):
    M_PROTON = 1
    ms1_coordinates = analysis1.dia_data.convert_from_indices(
        analysis1.precursor_indices,
        return_mobility_values=True,
        return_rt_values_min=True,
        return_mz_values=True,
        return_push_indices=True,
    )

    query_data = {}
    query_data['prec_id2'] = analysis1.precursor_indices
    query_data['mono_mzs2'] = ms1_coordinates["mz_values"]
    query_data['rt_list_ms2'] = ms1_coordinates["rt_values_min"]
    query_data['scan_list_ms2'] = ms1_coordinates["push_indices"]
    query_data['mobility2'] = ms1_coordinates["mobility_values"]
    query_data['charge2'] = np.array([2, 3])[np.isin(analysis1.precursor_indices, analysis1.mono_isotopes_charge3).astype(np.int)]
    query_data['prec_mass_list2'] = (query_data['mono_mzs2'] - M_PROTON) * query_data['charge2']
    query_data["indices_ms2"] = analysis1.precursor_indptr
    query_data["mass_list_ms2"] = analysis1.dia_data.mz_values[analysis1.dia_data.tof_indices[analysis1.fragment_indices]]
    query_data["int_list_ms2"] = analysis1.smooth_intensity_values[analysis1.fragment_indices]
    sort_query_data_fragments_by_mz(
        query_data["indices_ms2"],
        query_data["mass_list_ms2"],
        query_data["int_list_ms2"],
    )
    return query_data


def create_ap_like_hdf_file(query_data, file_name):
    import alphabase.io.hdf
    hdf = alphabase.io.hdf.HDF_File(
        file_name,
        read_only=False,
        truncate=True,
    )
    hdf.Raw = {"MS2_scans": query_data}




@alphatims.utils.pjit
def cluster_to_max_peaks_(
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
    clusters,
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
                        self_ref = clusters[self_index]
                        max_intensity = smooth_intensity_values[self_ref]
                        other_intensity = smooth_intensity_values[other_index]
                        if max_intensity < other_intensity:
                            clusters[self_index] = other_index
                        elif max_intensity == other_intensity:
                            if self_index <= other_index:
                                clusters[self_index] = other_index
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1


@alphatims.utils.njit
def walk_cluster_path(
    clusters
):
    for index, pointer in enumerate(clusters):
        initial_index = index
        path_length = 1
        while (pointer >= 0) and (index != pointer):
            index = pointer
            pointer = clusters[index]
            path_length += 1
        if pointer >= 0:
            final_pointer = -(pointer + 1)
        else:
            final_pointer = pointer
        index = initial_index
        for i in range(path_length):
            pointer = clusters[index]
            clusters[index] = final_pointer
            index = pointer
    for index, pointer in enumerate(clusters):
        clusters[index] = -(pointer + 1)


@alphatims.utils.njit
def walk_cluster_path_backup(
    clusters
):
    for index, pointer in enumerate(clusters):
        elements_on_path = []
        while pointer >= 0:
            elements_on_path.append(index)
            if index == pointer:
                pointer = -(pointer + 1)
                break
            index = pointer
            pointer = clusters[index]
        for index in elements_on_path:
            clusters[index] = pointer
    for index, pointer in enumerate(clusters):
        clusters[index] = -(pointer + 1)


@alphatims.utils.pjit
def find_unique_peaks_(
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
    clusters,
    unique_peaks,
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
                        if self_intensity <= other_intensity:
                            if clusters[self_index] != clusters[other_index]:
                                unique_peaks[self_index] = False
                        if self_intensity >= other_intensity:
                            if clusters[self_index] != clusters[other_index]:
                                unique_peaks[other_index] = False
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1


@alphatims.utils.njit
def walk_unique_cluster_path(
    cluster_pointers,
    nonambiguous_elements,
    to_visit
):
    for index, nonambiguous in enumerate(nonambiguous_elements):
        initial_index = index
        path_length = 0
        while nonambiguous:
            path_length += 1
            if not to_visit[index]:
                break
            else:
                to_visit[index] = False
            pointer = cluster_pointers[index]
            if index == pointer:
                break
            index = pointer
            nonambiguous = nonambiguous_elements[index]
        if not nonambiguous:
            index = initial_index
            for i in range(path_length):
                nonambiguous_elements[index] = False
                index = cluster_pointers[index]


@alphatims.utils.njit
def walk_unique_cluster_path_backup(
    clusters,
    uniques,
    to_visit,
):
    for index, pointer in enumerate(clusters):
        elements_on_path = []
        while uniques[index] and to_visit[index]:
            elements_on_path.append(index)
            if index == pointer:
                break
            index = pointer
            pointer = clusters[index]
        unique = uniques[index]
        for index in elements_on_path[::-1]:
            unique &= uniques[index]
            uniques[index] = unique
            to_visit[index] = False


@alphatims.utils.njit
def assemble_clusters(
    cluster_pointers,
    nonambiguous_ions,
    cluster_assemblies,
):
    for index, pointer in enumerate(cluster_pointers):
        if nonambiguous_ions[index]:
            if index != pointer:
                secondary_pointer = cluster_assemblies[pointer]
                cluster_assemblies[index] = secondary_pointer
                cluster_assemblies[pointer] = index




@alphatims.utils.njit(nogil=True)
def create_pseudo_msms_spectra_for_monos(
    precursor_indices,
    precursor_mz_values,
    precursor_push_indices,
    zeroth_frame,
    scan_max_index,
    scan_tolerance,
    cycle_tolerance,
    dia_mz_cycle,
    push_indptr,
    tof_indices,
    intensity_values,
    tof_max_index,
    cycle_sigma,
    scan_sigma,
    spectrum_intensities,
    spectrum_frequencies,
    spectrum_mzs,
    spectrum_indptr,
    tof_sigma,
    tof_tolerance,
    max_peaks_per_spectrum,
):
    cycle_length = len(dia_mz_cycle)
    frame_count = cycle_length // scan_max_index
    hits = np.zeros((2, tof_max_index))
    to_clear = np.zeros(tof_max_index, dtype=np.int32)
    for precursor_index, precursor_mz in enumerate(
        precursor_mz_values[precursor_indices]
    ):
        elements_to_clear = 0
        self_push_index = precursor_push_indices[precursor_index]
        self_scan_index = self_push_index % scan_max_index
        self_cycle_index = (
            self_push_index - zeroth_frame * scan_max_index
        ) // cycle_length
        max_positive_element_count = 0
        for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
            other_cycle_index = self_cycle_index + cycle_offset
            if other_cycle_index < 0:
                continue
            cycle_blur = gauss_correction(cycle_offset, cycle_sigma)
            for frame_index in range(frame_count):
                for scan_offset in range(-scan_tolerance, scan_tolerance + 1):
                    other_scan_index = self_scan_index + scan_offset
                    if not (0 <= other_scan_index < scan_max_index):
                        continue
                    other_push_offset = frame_index * scan_max_index + other_scan_index
                    low_mz, high_mz = dia_mz_cycle[other_push_offset]
                    if not (low_mz <= precursor_mz < high_mz):
                        continue
                    other_push_index = zeroth_frame * scan_max_index
                    other_push_index += other_cycle_index * cycle_length
                    other_push_index += other_push_offset
                    if not (0 <= other_push_index < len(push_indptr)):
                        continue
                    other_start = push_indptr[other_push_index]
                    other_end = push_indptr[other_push_index + 1]
                    scan_blur = gauss_correction(scan_offset, scan_sigma)
                    intensity_weight = cycle_blur * scan_blur
                    max_positive_element_count += intensity_weight
                    for index in range(other_start, other_end):
                        tof_index = tof_indices[index]
                        intensity = intensity_values[index]
                        if hits[0, tof_index] == 0:
                            to_clear[elements_to_clear] = tof_index
                            elements_to_clear += 1
                        hits[0, tof_index] += intensity_weight * intensity
                        hits[1, tof_index] += intensity_weight
        if elements_to_clear == 0:
            spectrum_indptr[precursor_index + 1] = spectrum_indptr[precursor_index]
        centroid_deconvoluted_peak(
            precursor_index,
            hits,
            to_clear,
            elements_to_clear,
            max_positive_element_count,
            spectrum_intensities,
            spectrum_frequencies,
            spectrum_mzs,
            spectrum_indptr,
            tof_tolerance,
            tof_sigma,
            tof_max_index,
            max_peaks_per_spectrum,
        )
        for index, tof_index in enumerate(to_clear[:elements_to_clear]):
            hits[0, tof_index] = 0
            hits[1, tof_index] = 0
            to_clear[index] = 0


@alphatims.utils.njit(nogil=True)
def centroid_deconvoluted_peak(
    precursor_index,
    hits,
    to_clear,
    elements_to_clear,
    max_positive_element_count,
    spectrum_intensities,
    spectrum_frequencies,
    spectrum_mzs,
    spectrum_indptr,
    tof_tolerance,
    tof_sigma,
    tof_max_index,
    max_peaks_per_spectrum,
):
    for tof_index in to_clear[:elements_to_clear]:
        hits[1, tof_index] /= max_positive_element_count
    for tof_index in to_clear[:elements_to_clear]:
        for tof_offset in range(-tof_tolerance, tof_tolerance + 1):
            other_tof = tof_index + tof_offset
            if not (0 <= other_tof < tof_max_index):
                continue
            if tof_offset == 0:
                continue
            tof_blur = gauss_correction(tof_offset, tof_sigma)
            other_intensity = hits[0, other_tof]
            other_frequency = hits[1, other_tof]
            hits[0, tof_index] += tof_blur * other_intensity
            hits[1, tof_index] += tof_blur * other_frequency
    for tof_index in to_clear[:elements_to_clear]:
        for tof_offset in range(-tof_tolerance, tof_tolerance + 1):
            other_tof = tof_index + tof_offset
            if not (0 <= other_tof < tof_max_index):
                continue
            if tof_offset == 0:
                continue
            if hits[1, tof_index] <= hits[1, other_tof]:
                hits[1, tof_index] = 0
                break
    elems0 = to_clear[:elements_to_clear]
    elems = hits[1][elems0]
    order = np.argsort(elems)[::-1]
    hit_offset = spectrum_indptr[precursor_index]
    for element in order[:max_peaks_per_spectrum]:
        tof_index = to_clear[element]
        spectrum_intensities[hit_offset] = hits[0, tof_index]
        spectrum_frequencies[hit_offset] = hits[1, tof_index]
        spectrum_mzs[hit_offset] = tof_index
        hit_offset += 1
    spectrum_indptr[precursor_index + 1] = hit_offset

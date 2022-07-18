"""Connect push indices from dia data."""

import logging

import numpy as np

import alphatims.utils
import alphatims.tempmmap as tm


class Connector:

    def __init__(
        self,
        scan_tolerance=6,
        connect_multiple_frames_per_cycle=False,
        connect_ms1=True,
        connect_ms2=True,
    ):
        self.scan_tolerance = scan_tolerance
        self.connect_multiple_frames_per_cycle = connect_multiple_frames_per_cycle
        self.connect_ms1 = connect_ms1
        self.connect_ms2 = connect_ms2

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def connect(self):
        logging.info("Setting connections")
        connection_counts, connections = get_connections_within_cycle(
            scan_tolerance=self.scan_tolerance,
            scan_max_index=self.dia_data.scan_max_index,
            dia_mz_cycle=self.dia_data.dia_mz_cycle,
            connect_multiple_frames_per_cycle=self.connect_multiple_frames_per_cycle,
            connect_ms1=self.connect_ms1,
            connect_ms2=self.connect_ms2,
        )
        self.connection_counts = tm.clone(connection_counts)
        self.connections = tm.clone(connections)


@alphatims.utils.njit(nogil=True)
def get_connections_within_cycle(
    scan_tolerance: int,
    scan_max_index: int,
    dia_mz_cycle: np.ndarray,
    exclude_self: bool = False,
    connect_multiple_frames_per_cycle: bool = False,
    connect_ms1: bool = True,
    connect_ms2: bool = False,
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
    connect_multiple_frames_per_cycle : bool
        Connect scans between different frames a cycle
        (the default is False).
    connect_ms1 : bool
        Allow connections between MS1 pushes
        (the default is True).
    connect_ms2 : bool
        Allow connections between MS2 pushes
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
    if connect_multiple_frames_per_cycle:
        frame_iterator = range(shape[1])
    for self_frame in range(shape[1]):
        if not connect_multiple_frames_per_cycle:
            frame_iterator = range(self_frame, self_frame + 1)
        for self_scan in range(shape[0]):
            index = self_scan + self_frame * shape[0]
            low_quad, high_quad = dia_mz_cycle[index]
            if (not connect_ms1) and (low_quad == -1):
                connection_counts.append(connection_count)
                continue
            if (not connect_ms2) and (low_quad != -1):
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
                    elif low_quad == other_high_quad:
                        if low_quad != -1:
                            continue
                    if high_quad < other_low_quad:
                        continue
                    elif high_quad == other_low_quad:
                        if high_quad != -1:
                            continue
                    connection_count += 1
                    connections.append(other_index)
            connection_counts.append(connection_count)
    return np.array(connection_counts), np.array(connections)


class PushConnector:

    def __init__(
        self,
        dia_data,
        subcycle_tolerance=3,
        scan_tolerance=6,
    ):
        logging.info("Setting connections")
        cycle = get_cycle(dia_data)
        indptr, indices = get_connections(
            cycle,
            scan_tolerance=scan_tolerance,
            subcycle_tolerance=subcycle_tolerance,
        )
        self.dia_data = dia_data
        self.scan_tolerance = scan_tolerance
        self.subcycle_tolerance = subcycle_tolerance
        self.cycle = cycle
        self.indptr = indptr
        self.indices = indices
        self.connection_counts = self.indptr
        self.connections = self.indices


def get_cycle(dia_data):
    last_window_group = -1
    for max_index, (frame, window_group) in enumerate(
        zip(
            dia_data.fragment_frames.Frame,
            dia_data.fragment_frames.Precursor
        )
    ):
        if window_group < last_window_group:
            break
        else:
            last_window_group = window_group
    frames = dia_data.fragment_frames.Frame[max_index-1]
    frames += dia_data.fragment_frames.Frame[0] == int(dia_data.zeroth_frame)
    sub_cycles = frames - len(np.unique(dia_data.fragment_frames.Frame[:max_index]))
    cycle = np.zeros(
        (
            frames,
            dia_data.scan_max_index,
            2,
        )
    )
    # cycle[:] = -1
    precursor_frames = np.ones(frames, dtype=np.bool_)
    for index, row in dia_data.fragment_frames[:max_index].iterrows():
        frame = int(row.Frame - dia_data.zeroth_frame)
        scan_begin = int(row.ScanNumBegin)
        scan_end = int(row.ScanNumEnd)
        low_mz = row.IsolationMz - row.IsolationWidth / 2
        high_mz = row.IsolationMz + row.IsolationWidth / 2
    #     print(low_mz, high_mz)
        cycle[
            frame,
            scan_begin: scan_end,
        ] = (low_mz, high_mz)
        precursor_frames[frame] = False
    cycle[precursor_frames] = (-1, -1)
    cycle = cycle.reshape(
        (
            sub_cycles,
            frames // sub_cycles,
            *cycle.shape[1:]
        )
    )
    return cycle


def get_connections(
    cycle,
    scan_tolerance,
    subcycle_tolerance,
):
    cycle_size = np.prod(cycle.shape[:-1])
    max_subcycle_count = cycle.shape[0]
    cycle_tolerance = int(np.ceil(subcycle_tolerance / max_subcycle_count))
    pointer_cycle = np.arange(
        cycle_size
    ).reshape(cycle.shape[:-1])
    indices = []
    indptr = np.empty(np.prod(cycle.shape[:-1]) + 1, dtype=np.int64)
    indptr[0] = 0
    push_index = 0
    hit_count = 0
    for subcycle_index, subcycle in enumerate(cycle):
        for frame_index, frame in enumerate(subcycle):
            for scan_index, (low_mz, high_mz) in enumerate(frame):
                low_scan_index = max(0, scan_index - scan_tolerance)
                high_scan_index = scan_index + scan_tolerance + 1
                if low_mz == -1:
                    sub_selection = cycle[
                        :, :, low_scan_index: high_scan_index, 0
                    ] == -1
                else:
                    sub_selection = cycle[
                        :, :, low_scan_index: high_scan_index, 0
                    ] < high_mz
                    sub_selection &= cycle[
                        :, :, low_scan_index: high_scan_index, 1
                    ] > low_mz
                selected_pointers = pointer_cycle[
                    :, :, low_scan_index: high_scan_index
                ][sub_selection]
                elements = []
                for i in range(-cycle_tolerance, cycle_tolerance + 1):
                    elements.append(selected_pointers + i * cycle_size)
                elements = np.concatenate(elements)
                subcycle_offsets = elements // np.prod(cycle.shape[1:-1]) - subcycle_index
                left = np.searchsorted(
                    subcycle_offsets,
                    -subcycle_tolerance,
                    side="left",
                )
                right = np.searchsorted(
                    subcycle_offsets,
                    subcycle_tolerance,
                    side="right",
                )
                selected_elements = elements[left: right]
                selected_offsets = selected_elements - push_index
                indices.append(selected_offsets)
                push_index += 1
                hit_count += len(selected_offsets)
                indptr[push_index] = hit_count
    indices = np.concatenate(indices)
    return indptr, indices

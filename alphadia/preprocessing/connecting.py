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
                    if high_quad < other_low_quad:
                        continue
                    connection_count += 1
                    connections.append(other_index)
            connection_counts.append(connection_count)
    return np.array(connection_counts), np.array(connections)

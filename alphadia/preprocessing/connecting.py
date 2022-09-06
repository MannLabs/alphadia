"""Connect push indices from dia data."""

import logging

import numpy as np

import alphatims.utils
import alphatims.tempmmap as tm


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
    if hasattr(dia_data, "cycle"):
        return dia_data.cycle
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

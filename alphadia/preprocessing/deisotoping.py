"""Deisotope peaks"""

import logging

import numpy as np

import alphatims.utils
import alphatims.tempmmap as tm


class Deisotoper:

    def __init__(
        self,
        isotope_mz_tolerance=0.01,
        cycle_tolerance=3,
    ):
        self.isotope_mz_tolerance = isotope_mz_tolerance
        self.cycle_tolerance = cycle_tolerance

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_connector(self, connector):
        self.connector = connector

    def set_peak_collection(self, peak_collection):
        self.peak_collection = peak_collection

    def deisotope(self):
        logging.info("Determining mono isotopes")
        logging.info("Charge 2")
        left_connection, right_connection = create_isotopic_pairs(
            self,
            difference=1/2,
            mz_tolerance=self.isotope_mz_tolerance,
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
            mz_tolerance=self.isotope_mz_tolerance,
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

def create_isotopic_pairs(
    self,
    difference,
    mz_tolerance,
):
    import multiprocessing

    def starfunc(cycle_index):
        return get_isotopic_pairs(
            cycle_index,
            self.peak_collection.indptr,
            self.dia_data.mz_values[
                self.dia_data.tof_indices[
                    self.peak_collection.indices
                ]
            ],
            mz_tolerance,
            self.dia_data.scan_max_index,
            self.dia_data.zeroth_frame,
            self.connector.connection_counts,
            self.connector.connections,
            self.cycle_tolerance,
            difference,
        )

    iterable = range(
        len(self.dia_data.push_indptr) // np.prod(
            self.connector.cycle.shape[:-1]
        ) + 1
    )
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
        if True:
            for other_connection_offset in connections[connection_start: connection_end]:
                other_push_index = self_push_index + other_connection_offset
                if other_push_index == self_push_index:
                    continue
                if not (0 <= other_push_index < len(indptr)):
                    continue
        # for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
        #     for other_connection_index in connections[connection_start: connection_end]:
        #         other_push_index = push_offset + other_connection_index + len_dia_mz_cycle * cycle_offset
        #         if other_push_index >= len(indptr):
        #             continue
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

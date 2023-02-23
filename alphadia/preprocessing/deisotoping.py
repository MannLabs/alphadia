"""Deisotope peaks"""

import logging

import numpy as np

import alphatims.utils
import alphatims.tempmmap as tm
import alphadia.preprocessing.peakstats


class Deisotoper:

    def __init__(
        self,
        isotope_mz_tolerance=0.01,
        cycle_tolerance=3,
        min_correlation=0.5,
        proton_mass=1.007277,
    ):
        self.isotope_mz_tolerance = isotope_mz_tolerance
        self.cycle_tolerance = cycle_tolerance
        self.min_correlation = min_correlation
        self.proton_mass = proton_mass

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_connector(self, connector):
        self.connector = connector

    def set_peak_collection(self, peak_collection):
        self.peak_collection = peak_collection

    def set_peak_stats_calculator(self, peak_stats_calculator):
        self.peak_stats_calculator = peak_stats_calculator

    def deisotope(self):
        logging.info("Determining mono isotopes")
        logging.info("Charge 2")
        self.mono_isotopes_charge2 = create_isotopic_pairs(
            self,
            difference=self.proton_mass/2,
            mz_tolerance=self.isotope_mz_tolerance,
            min_correlation=self.min_correlation
        )
        logging.info("Charge 3")
        self.mono_isotopes_charge3 = create_isotopic_pairs(
            self,
            difference=self.proton_mass/3,
            mz_tolerance=self.isotope_mz_tolerance,
            min_correlation=self.min_correlation
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
    min_correlation,
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
    left_connection = np.concatenate(self_connections)
    right_connection = np.concatenate(other_connections)
    xic_correlations = np.empty(len(left_connection))
    alphadia.preprocessing.peakstats.set_profile_correlations(
        range(len(left_connection)),
        # range(10),
        # 0,
        left_connection,
        right_connection,
        np.arange(len(left_connection) + 1),
        xic_correlations,
        self.peak_stats_calculator.xic_offset,
        self.peak_stats_calculator.xics,
        self.peak_stats_calculator.xic_indptr,
    )
    mobilogram_correlations = np.empty(len(left_connection))
    alphadia.preprocessing.peakstats.set_profile_correlations(
        range(len(left_connection)),
        # range(10),
        # 0,
        left_connection,
        right_connection,
        np.arange(len(left_connection) + 1),
        mobilogram_correlations,
        self.peak_stats_calculator.mobilogram_offset,
        self.peak_stats_calculator.mobilograms,
        self.peak_stats_calculator.mobilogram_indptr,
    )
    correlation = xic_correlations * mobilogram_correlations
    mono_isotopes_charge = tm.clone(
        self.peak_collection.indices[
            left_connection[correlation > min_correlation][
                ~np.isin(
                    left_connection[correlation > min_correlation], right_connection[correlation > min_correlation]
                # ) & np.isin(
                #     right_connection, left_connection
                )
            ]
        ]
    )
    return mono_isotopes_charge


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
    len_cycle = len(connection_counts) - 1
    push_offset = len_cycle * cycle_index + zeroth_frame * scan_max_index
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
                # if other_push_index == self_push_index:
                #     continue
                if not (0 <= other_push_index < len(indptr)):
                    continue
        # for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
        #     for other_connection_index in connections[connection_start: connection_end]:
        #         other_push_index = push_offset + other_connection_index + len_cycle * cycle_offset
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

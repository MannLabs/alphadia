"""Create MSMS spectra."""

import logging

import numpy as np
import pandas as pd

import alphatims.utils
import alphatims.tempmmap as tm


class MSMSGenerator:

    def __init__(
        self,
        scan_tolerance=6,
        cycle_tolerance=3,
        cycle_sigma=3,
        scan_sigma=6,
    ):
        self.scan_tolerance = scan_tolerance
        self.cycle_tolerance = cycle_tolerance
        self.cycle_sigma = cycle_sigma
        self.scan_sigma = scan_sigma

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_peak_collection(self, peak_collection):
        self.peak_collection = peak_collection

    def set_deisotoper(self, deisotoper):
        self.deisotoper = deisotoper

    def create_msms_spectra(self):
        import multiprocessing
        logging.info("Creating MSMS spectra")

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
                np.isin(self.peak_collection.indices, self.deisotoper.mono_isotopes)
            )

        precursor_indices = []
        precursor_counts = [[0]]
        fragment_indices = []

        iterable = range(
            len(self.dia_data.push_indptr) // len(self.dia_data.dia_mz_cycle) + 1
        )

        with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
            for (
                precursor_indices_,
                precursor_counts_,
                fragment_indices_,
            ) in alphatims.utils.progress_callback(
                pool.imap(starfunc, iterable),
                total=len(iterable),
                include_progress_callback=True
            ):
                precursor_indices.append(precursor_indices_)
                precursor_counts.append(precursor_counts_)
                fragment_indices.append(fragment_indices_)

        precursor_indices = np.concatenate(precursor_indices)
        precursor_counts = np.cumsum(np.concatenate(precursor_counts))
        fragment_indices = np.concatenate(fragment_indices)
        self.precursor_indices = tm.clone(precursor_indices)
        self.precursor_indptr = tm.clone(precursor_counts)
        self.fragment_indices = tm.clone(fragment_indices)
        self.set_fragment_weights()

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

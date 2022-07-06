"""Calculate peak stats."""

import logging

import numpy as np
import pandas as pd

import alphatims.utils
import alphatims.tempmmap as tm


class PeakStatsCalculator:

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_peakfinder(self, peakfinder):
        self.peakfinder = peakfinder

    def calculate_stats(self):
        logging.info("Calculating peak stats")
        import multiprocessing

        iterable = range(len(self.peakfinder.peak_collection.indices))
        # iterable = range(4585132, 4585132+1)

        self.cycle_rt_values = self.dia_data.rt_values[
            int(self.dia_data.zeroth_frame)::len(self.dia_data.dia_mz_cycle)//self.dia_data.scan_max_index
        ]

        self.number_of_ions = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=np.int32
        )

        self.summed_intensity_values = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=np.float64
        )

        xic_indptr = tm.empty(
            len(self.peakfinder.peak_collection.indices) + 1,
            dtype=np.int64
        )
        xic_indptr[0] = 0
        self.xic_offset = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=np.int32
        )
        self.rt_average = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.rt_values.dtype
        )
        self.rt_start = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.rt_values.dtype
        )
        self.rt_end = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.rt_values.dtype
        )

        mobilogram_indptr = tm.empty(
            len(self.peakfinder.peak_collection.indices) + 1,
            dtype=np.int64
        )
        mobilogram_indptr[0] = 0
        self.mobilogram_offset = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=np.int32
        )
        self.mobility_average = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.mobility_values.dtype
        )
        self.mobility_start = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.mobility_values.dtype
        )
        self.mobility_end = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.mobility_values.dtype
        )

        mz_profile_indptr = tm.empty(
            len(self.peakfinder.peak_collection.indices) + 1,
            dtype=np.int64
        )
        mz_profile_indptr[0] = 0
        self.mz_profile_offset = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=np.int32
        )
        self.mz_average = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.mz_values.dtype
        )
        self.mz_start = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.mz_values.dtype
        )
        self.mz_end = tm.empty(
            self.peakfinder.peak_collection.indices.shape,
            dtype=self.dia_data.mz_values.dtype
        )

        def starfunc(index):
            return calculate_stats(
                index,
                self.peakfinder.peak_collection.indices,
                self.dia_data.push_indptr,
                self.dia_data.tof_indices,
                self.dia_data.scan_max_index,
                self.dia_data.zeroth_frame,
                self.dia_data.intensity_values,
                self.dia_data.mz_values,
                self.dia_data.rt_values,
                self.cycle_rt_values,
                self.dia_data.mobility_values,
                self.peakfinder.cluster_assemblies,
                len(self.dia_data.dia_mz_cycle),
                self.number_of_ions,
                self.summed_intensity_values,
                xic_indptr,
                self.xic_offset,
                self.rt_average,
                self.rt_start,
                self.rt_end,
                mobilogram_indptr,
                self.mobilogram_offset,
                self.mobility_average,
                self.mobility_start,
                self.mobility_end,
                mz_profile_indptr,
                self.mz_profile_offset,
                self.mz_average,
                self.mz_start,
                self.mz_end,
            )

        xics = []
        mobilograms = []
        mz_profiles = []
        with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
            for (
                xic,
                mobilogram,
                mz_profile,
            ) in alphatims.utils.progress_callback(
                pool.imap(starfunc, iterable),
                total=len(iterable),
                include_progress_callback=True
            ):
                xics.append(xic)
                mobilograms.append(mobilogram)
                mz_profiles.append(mz_profile)
        self.xics = tm.clone(np.concatenate(xics))
        self.mobilograms = tm.clone(np.concatenate(mobilograms))
        self.mz_profiles = tm.clone(np.concatenate(mz_profiles))
        self.xic_indptr = tm.clone(np.cumsum(xic_indptr))
        self.mobilogram_indptr = tm.clone(np.cumsum(mobilogram_indptr))
        self.mz_profile_indptr = tm.clone(np.cumsum(mz_profile_indptr))

    def as_dataframe(self, selected_indices=Ellipsis, *, append_apices=False):
        raw_indices = self.peakfinder.peak_collection.indices[
            selected_indices
        ]
        df = pd.DataFrame(
            {
                "number_of_ions": self.number_of_ions[
                    selected_indices
                ],
                "summed_intensity_values": self.summed_intensity_values[
                    selected_indices
                ],
                "xic_offset": self.xic_offset[
                    selected_indices
                ],
                "rt_average": self.rt_average[
                    selected_indices
                ],
                "rt_start": self.rt_start[
                    selected_indices
                ],
                "rt_end": self.rt_end[
                    selected_indices
                ],
                "mobilogram_offset": self.mobilogram_offset[
                    selected_indices
                ],
                "mobility_average": self.mobility_average[
                    selected_indices
                ],
                "mobility_start": self.mobility_start[
                    selected_indices
                ],
                "mobility_end": self.mobility_end[
                    selected_indices
                ],
                "mz_profile_offset": self.mz_profile_offset[
                    selected_indices
                ],
                "mz_average": self.mz_average[
                    selected_indices
                ],
                "mz_start": self.mz_start[
                    selected_indices
                ],
                "mz_end": self.mz_end[
                    selected_indices
                ],
            }
        )
        if append_apices:
            apex_df = self.dia_data.as_dataframe(raw_indices)
            df = df.join(apex_df, how="left")
        else:
            df["raw_indices"] = raw_indices
        return df


@alphatims.utils.njit
def calculate_stats(
    index,
    peak_indices,
    push_indptr,
    tof_indices,
    scan_max_index,
    zeroth_frame,
    intensity_values,
    mz_values,
    rt_values,
    cycle_rt_values,
    mobility_values,
    cluster_assemblies,
    cycle_length,
    number_of_ions,
    summed_intensity_values,
    xic_indptr,
    xic_offset,
    rt_average,
    rt_start,
    rt_end,
    mobilogram_indptr,
    mobilogram_offset,
    mobility_average,
    mobility_start,
    mobility_end,
    mz_profile_indptr,
    mz_profile_offset,
    mz_average,
    mz_start,
    mz_end,
):
    peak_index = peak_indices[index]
    raw_ion_indices = get_ions(peak_index, cluster_assemblies)
    number_of_ions[index] = len(raw_ion_indices)
    if len(raw_ion_indices) == 0:
        return (
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )
    raw_intensities = intensity_values[raw_ion_indices]
    summed_intensity = np.sum(raw_intensities)
    summed_intensity_values[index] = summed_intensity
    push_indices = np.searchsorted(
        push_indptr,
        raw_ion_indices,
        "right"
    ) - 1 - zeroth_frame * scan_max_index
    scan_intensities = calculate_rt_stats(
        index,
        push_indices,
        raw_intensities,
        cycle_length,
        xic_indptr,
        xic_offset,
        rt_average,
        rt_start,
        rt_end,
        rt_values,
        cycle_rt_values,
        summed_intensity,
    )
    cycle_intensities = calculate_mobility_stats(
        index,
        push_indices,
        raw_intensities,
        scan_max_index,
        mobilogram_indptr,
        mobilogram_offset,
        mobility_average,
        mobility_start,
        mobility_end,
        mobility_values,
        summed_intensity,
    )
    mz_intensities = calculate_mz_stats(
        index,
        tof_indices[raw_ion_indices],
        raw_intensities,
        mz_profile_indptr,
        mz_profile_offset,
        mz_average,
        mz_start,
        mz_end,
        mz_values,
        summed_intensity,
    )
    return (
        scan_intensities,
        cycle_intensities,
        mz_intensities,
    )


@alphatims.utils.njit
def get_ions(peak_index, cluster_assemblies):
    raw_ions = [peak_index]
    pointer = cluster_assemblies[peak_index]
    while pointer != peak_index:
        raw_ions.append(pointer)
        pointer = cluster_assemblies[pointer]
    return np.array(raw_ions)


@alphatims.utils.njit
def calculate_rt_stats(
    index,
    push_indices,
    raw_intensities,
    cycle_length,
    xic_indptr,
    xic_offset,
    rt_average,
    rt_start,
    rt_end,
    rt_values,
    cycle_rt_values,
    summed_intensity,
):
    scan_indices = push_indices // cycle_length
    lowest_scan, scan_intensities = extract_profile(
        raw_intensities,
        scan_indices
    )
    xic_indptr[index + 1] = len(scan_intensities)
    xic_offset[index] = lowest_scan
    rt_average[index] = np.sum(
        scan_intensities * cycle_rt_values[
            lowest_scan: lowest_scan + len(scan_intensities)
        ]
    ) / summed_intensity
    rt_start[index] = cycle_rt_values[lowest_scan]
    rt_end[index] = cycle_rt_values[lowest_scan + len(scan_intensities) - 1]
    return scan_intensities


@alphatims.utils.njit
def calculate_mobility_stats(
    index,
    push_indices,
    raw_intensities,
    scan_max_index,
    mobilogram_indptr,
    mobilogram_offset,
    mobility_average,
    mobility_start,
    mobility_end,
    mobility_values,
    summed_intensity,
):
    cycle_indices = push_indices % scan_max_index
    lowest_cycle, cycle_intensities = extract_profile(
        raw_intensities,
        cycle_indices
    )
    mobilogram_indptr[index + 1] = len(cycle_intensities)
    mobilogram_offset[index] = lowest_cycle
    mobility_average[index] = np.sum(
        cycle_intensities * mobility_values[
            lowest_cycle: lowest_cycle + len(cycle_intensities)
        ]
    ) / summed_intensity
    mobility_end[index] = mobility_values[lowest_cycle]
    mobility_start[index] = mobility_values[lowest_cycle + len(cycle_intensities) - 1]
    return cycle_intensities


@alphatims.utils.njit
def calculate_mz_stats(
    index,
    tof_indices,
    raw_intensities,
    mz_profile_indptr,
    mz_profile_offset,
    mz_average,
    mz_start,
    mz_end,
    mz_values,
    summed_intensity,
):
    lowest_mz_index, mz_intensities = extract_profile(
        raw_intensities,
        tof_indices
    )
    mz_profile_indptr[index + 1] = len(mz_intensities)
    mz_profile_offset[index] = lowest_mz_index
    mz_average[index] = np.sum(
        mz_intensities * mz_values[
            lowest_mz_index: lowest_mz_index + len(mz_intensities)
        ]
    ) / summed_intensity
    mz_start[index] = mz_values[lowest_mz_index]
    mz_end[index] = mz_values[lowest_mz_index + len(mz_intensities) - 1]
    return mz_intensities


@alphatims.utils.njit
def extract_profile(
    raw_intensities,
    indices
):
    minimum_index = np.min(indices)
    maximum_index = np.max(indices)
    cumulative_intensities = np.zeros(maximum_index - minimum_index + 1)
    for index, intensity in zip(indices, raw_intensities):
        cumulative_intensities[index - minimum_index] += intensity
    return minimum_index, cumulative_intensities

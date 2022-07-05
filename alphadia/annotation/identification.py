"""Annotate pseudo MSMS spectra."""

import logging

import numpy as np

import alphatims.utils


class MSMSIdentifier:

    def __init__(
        self,
        precursor_ppm=50,
        fragment_ppm=50,
        min_size=10,
        ppm_mean=0,
        min_hit_count=1,
        append_stats=True,
        top_n_hits=1,
    ):
        self.precursor_ppm = precursor_ppm
        self.fragment_ppm = fragment_ppm
        self.min_size = min_size
        self.ppm_mean = ppm_mean
        self.min_hit_count = min_hit_count
        self.append_stats = append_stats
        self.top_n_hits = top_n_hits

    def set_preprocessor(self, preprocessing_workflow):
        self.preprocessing_workflow = preprocessing_workflow
        self.dia_data = preprocessing_workflow.dia_data
        self.msms_generator = preprocessing_workflow.msms_generator

    def set_library(self, library):
        self.library = library

    def update_ppm_values_from_stats_calculator(
        self,
        psm_stats_calculator
    ):
        self.ppm_mean = psm_stats_calculator.ppm_mean
        self.fragment_ppm = psm_stats_calculator.ppm_width
        self.precursor_ppm = psm_stats_calculator.ppm_width

    def identify(
        self,
    ):
        logging.info(
            f"Quick library annotation of mono isotopes with {self.ppm_mean=} and {self.precursor_ppm=}"
        )
        o = np.argsort(
            self.dia_data.tof_indices[self.msms_generator.precursor_indices]
        )
        mz_values = self.dia_data.mz_values * (1 + self.ppm_mean * 10**-6)
        p_mzs = mz_values[
            self.dia_data.tof_indices[self.msms_generator.precursor_indices][o]
        ]
        lower = np.empty(
            len(self.msms_generator.precursor_indices),
            dtype=np.int64
            )
        upper = np.empty(
            len(self.msms_generator.precursor_indices),
            dtype=np.int64
            )
        lower[o] = np.searchsorted(
            self.library.predicted_library_df.precursor_mz.values,
            p_mzs / (1 + self.precursor_ppm * 10**-6)
        )
        upper[o] = np.searchsorted(
            self.library.predicted_library_df.precursor_mz.values,
            p_mzs * (1 + self.precursor_ppm * 10**-6)
        )
        logging.info(
            f"PSMs to test: {np.sum(((upper - lower) * (np.diff(self.msms_generator.precursor_indptr) >= self.min_size)))}"
        )
        (
            precursor_indices,
            precursor_indptr,
            hit_counts,
            frequency_counts,
            db_indices,
        ) = annotate(
            range(len(lower)),
            # range(100),
            self.library.predicted_library_df.frag_start_idx.values,
            self.library.predicted_library_df.frag_end_idx.values,
            self.msms_generator.fragment_indices,
            self.msms_generator.fragment_frequencies,
            self.msms_generator.precursor_indptr,
            mz_values,
            self.dia_data.tof_indices,
            self.fragment_ppm,
            lower,
            upper,
            self.library.y_mzs,
            self.library.b_mzs,
            self.min_size,
            self.min_hit_count,
            self.top_n_hits,
        )

        precursor_selection = np.repeat(precursor_indices, precursor_indptr)
        hits = self.dia_data.as_dataframe(self.msms_generator.precursor_indices[precursor_selection])
        hits["inet_index"] = precursor_selection
        hits["candidates"] = (upper - lower)[precursor_selection]
        hits["total_peaks"] = np.diff(self.msms_generator.precursor_indptr)[precursor_selection]
        hits["db_index"] = db_indices.astype(np.int64)
        # hits["counts"] = np.repeat(hit_counts, precursor_indptr)
        hits["counts"] = hit_counts
        hits["frequency_counts"] = frequency_counts
        self.annotation = hits
        self.annotation["smooth_intensity"] = self.preprocessing_workflow.smoother.smooth_intensity_values[
            self.annotation.raw_indices
        ]
        self.annotation = self.annotation.join(self.library.predicted_library_df, on="db_index")
        self.annotation["im_diff"] = self.annotation.mobility_pred - self.annotation.mobility_values
        self.annotation["mz_diff"] = self.annotation.precursor_mz - self.annotation.mz_values
        self.annotation["ppm_diff"] = self.annotation.mz_diff / self.annotation.precursor_mz * 10**6
        self.annotation["target"] = ~self.annotation.decoy
        self.annotation.reset_index(drop=True, inplace=True)


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
        return annotate_pool2(
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


@alphatims.utils.njit(nogil=True)
def annotate_pool2(
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
):
    start = indptr[index]
    end = indptr[index + 1]
    results = [0][1:] # this defines the type
    hit_counts = [0][1:] # this defines the type
    frequency_counts = [0.0][1:] # this defines the type
    if (end - start) < min_size:
        return index, hit_counts, frequency_counts, results
    if (end - start) < min_hit_count:
        return index, hit_counts, frequency_counts, results
    fragment_indices = frag_indices[start: end]
    fragment_tofs = tof_indices[fragment_indices]
    order = np.argsort(fragment_tofs)
    frequencies = frag_frequencies[start: end][order]
    fragment_tofs = fragment_tofs[order]
    fragment_mzs = mz_values[fragment_tofs]
    max_hit_count = min_hit_count
    for db_index in range(lower[index], upper[index]):
        frag_start = frag_start_idx[db_index]
        frag_end = frag_end_idx[db_index]
        y_hits, y_frequency = hit_and_frequency_count(
            fragment_mzs,
            frequencies,
            y_mzs[frag_start: frag_end][::-1],
            fragment_ppm,
        )
        b_hits, b_frequency = hit_and_frequency_count(
            fragment_mzs,
            frequencies,
            b_mzs[frag_start: frag_end],
            fragment_ppm,
        )
        hit_count = b_hits + y_hits
        frequency_count = b_frequency + y_frequency
        if top_n_hits == 1:
            if frequency_count == max_hit_count:
                results.append(db_index)
                hit_counts.append(hit_count)
                frequency_counts.append(frequency_count)
            elif frequency_count > max_hit_count:
                results = [db_index]
                hit_counts = [hit_count]
                frequency_counts = [frequency_count]
                max_hit_count = hit_count
        elif frequency_count >= min_hit_count:
            if len(results) >= top_n_hits:
                for min_index, freq_count in enumerate(frequency_counts):
                    if freq_count == min_hit_count:
                        results[min_index] = db_index
                        hit_counts[min_index] = hit_count
                        frequency_counts[min_index] = frequency_count
                        break
                min_hit_count = min(frequency_counts)
            else:
                results.append(db_index)
                hit_counts.append(hit_count)
                frequency_counts.append(frequency_count)
    # return index, max_hit_count, results
    return index, hit_counts, frequency_counts, results



@alphatims.utils.njit(nogil=True)
def hit_and_frequency_count(
    fragment_mzs,
    frequencies,
    database_mzs,
    fragment_ppm,
):
    fragment_index = 0
    database_index = 0
    hits = 0
    summed_frequency = 0
    while (fragment_index < len(fragment_mzs)) and (database_index < len(database_mzs)):
        fragment_mz = fragment_mzs[fragment_index]
        database_mz = database_mzs[database_index]
        frequency = frequencies[fragment_index]
        if fragment_mz < (database_mz / (1 + 10**-6 * fragment_ppm)):
            fragment_index += 1
        elif database_mz < (fragment_mz / (1 + 10**-6 * fragment_ppm)):
            database_index += 1
        else:
            hits += 1
            summed_frequency += frequency
            fragment_index += 1
            database_index += 1
    return hits, summed_frequency

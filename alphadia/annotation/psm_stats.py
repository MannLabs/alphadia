"""Calculate PSM stats."""

import logging

import numpy as np

import alphatims.utils


class PSMStatsCalculator:

    def __init__(
        self,
        pseudo_int=10**-6,
    ):
        self.pseudo_int = pseudo_int

    def set_ions(self, precursor_df, fragment_df):
        self.precursor_df = precursor_df
        self.fragment_df = fragment_df

    def set_library(self, library):
        self.library = library

    def set_annotation(self, annotation):
        self.annotation = annotation

    def estimate_mz_tolerance(self):
        logging.info("Estimating ppm values")
        ppm_diffs = self.annotation.ppm_diff
        order = np.argsort(ppm_diffs.values)

        decoys, targets = np.bincount(self.annotation.decoy.values)
        distribution = np.cumsum(
            [
                1 / targets if i else -1 / decoys for i in self.annotation.decoy.values[order]
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
        #     data=self.annotation,
        #     x="ppm_diff",
        #     hue="decoy",
        # )

    def update_annotation_stats(self):
        logging.info("Appending stats to quick annotation")
        b_hit_counts = np.zeros(len(self.annotation))
        y_hit_counts = np.zeros(len(self.annotation))
        b_mean_ppm = np.zeros(len(self.annotation))
        y_mean_ppm = np.zeros(len(self.annotation))
        relative_found_b_int = np.zeros(len(self.annotation))
        relative_missed_b_int = np.zeros(len(self.annotation))
        relative_found_y_int = np.zeros(len(self.annotation))
        relative_missed_y_int = np.zeros(len(self.annotation))
        relative_found_int = np.zeros(len(self.annotation))
        relative_missed_int = np.zeros(len(self.annotation))
        pearsons = np.zeros(len(self.annotation))
        pearsons_log = np.zeros(len(self.annotation))
        update_annotation(
            range(len(self.annotation)),
            # 1000,
            self.annotation.db_index.values,
            self.library.predicted_library_df.frag_start_idx.values,
            self.library.predicted_library_df.frag_end_idx.values,
            self.library.y_mzs,
            self.library.b_mzs,
            self.library.y_ions_intensities,
            self.library.b_ions_intensities,
            self.annotation.inet_index.values,

            self.precursor_df.fragment_start.values,
            self.precursor_df.fragment_end.values,
            self.fragment_df.summed_intensity_values.values,
            self.fragment_df.mz_average.values * (1 + self.ppm_mean * 10**-6),
            # self.precursor_indptr,
            # self.fragment_indices,
            # self.tof_indices,
            # self.smooth_intensity_values, #.astype(np.float64),
            # self.mz_values * (1 + self.ppm_mean * 10**-6),


            self.ppm_width,
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
            np.float32(self.pseudo_int),
        )
        self.annotation["b_hit_counts"] = b_hit_counts
        self.annotation["y_hit_counts"] = y_hit_counts
        self.annotation["b_mean_ppm"] = b_mean_ppm
        self.annotation["y_mean_ppm"] = y_mean_ppm
        self.annotation["relative_found_b_int"] = relative_found_b_int
        self.annotation["relative_missed_b_int"] = relative_missed_b_int
        self.annotation["relative_found_y_int"] = relative_found_y_int
        self.annotation["relative_missed_y_int"] = relative_missed_y_int
        self.annotation["relative_found_int"] = relative_found_int
        self.annotation["relative_missed_int"] = relative_missed_int
        pearsons[~np.isfinite(pearsons)] = 0
        self.annotation["pearsons"] = pearsons
        pearsons_log[~np.isfinite(pearsons_log)] = 0
        self.annotation["pearsons_log"] = pearsons_log


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
    fragment_start,
    fragment_end,
    fragment_intensities,
    fragment_mzs,
    # precursor_indptr,
    # fragment_indices,
    # tof_indices,
    # intensity_values,
    # mz_values,
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
    frag_start_idx = fragment_start[precursor_index]
    frag_end_idx = fragment_end[precursor_index]
    fragment_mzs = fragment_mzs[frag_start_idx: frag_end_idx]
    fragment_ints = fragment_intensities[frag_start_idx: frag_end_idx]
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

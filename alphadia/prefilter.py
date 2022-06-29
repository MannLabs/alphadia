"""A module to perform prefilterinf of peptides."""

import alphatims.utils
import numpy as np
import sklearn.neighbors

import alphadia.library
import alpharaw.smoothing


# @alphatims.utils.njit(nogil=True)
@alphatims.utils.pjit
def find_maximal_pushes(
    cycle_index,
    connection_indptr,
    connection_indices,
    cycle_tolerance,
    zeroth_frame,
    scan_max_index,
    intensity_values,
    selected_push_indices,
    maximum_push_indices,
):
    cycle_length = len(connection_indptr) - 1
    for self_connection_index, connection_start in enumerate(
        connection_indptr[:-1]
    ):
        self_push_index = self_connection_index
        self_push_index += cycle_index * cycle_length
        self_push_index += zeroth_frame * scan_max_index
        if not (0 <= self_push_index < len(selected_push_indices)):
            continue
        if not maximum_push_indices[self_push_index]:
            continue
        self_intensity_index = selected_push_indices[self_push_index]
        if self_intensity_index == -1:
            maximum_push_indices[self_push_index] = False
            continue
        self_intensity = intensity_values[self_intensity_index]
        for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
            if cycle_offset == 0:
                continue
            other_push_index = self_push_index + cycle_offset * cycle_length
            if not (0 <= other_push_index < len(selected_push_indices)):
                continue
            other_intensity_index = selected_push_indices[other_push_index]
            if other_intensity_index == -1:
                continue
            other_intensity = intensity_values[other_intensity_index]
            if self_intensity < other_intensity:
                maximum_push_indices[self_push_index] = False
                break
        if not maximum_push_indices[self_push_index]:
            continue
        connection_end = connection_indptr[self_connection_index + 1]
        other_connection_indices = connection_indices[connection_start: connection_end]
        for other_push_index in self_push_index + other_connection_indices:
            if not (0 <= other_push_index < len(selected_push_indices)):
                continue
            other_intensity_index = selected_push_indices[other_push_index]
            if other_intensity_index == -1:
                continue
            other_intensity = intensity_values[other_intensity_index]
            if self_intensity < other_intensity:
                maximum_push_indices[self_push_index] = False
                break


@alphatims.utils.pjit
def find_max_intensity_per_push(
    cycle_index,
    dia_mz_cycle,
    zeroth_frame,
    scan_max_index,
    push_indptr,
    intensity_values,
    tof_indices,
    mz_values,
    selected_push_indices,
    min_peaks,
    min_intensity_value,
    use_ms1,
    use_ms2,
    use_precursor_range,
):
    cycle_length = len(dia_mz_cycle)
    for self_connection_index, (lower_mz, upper_mz) in enumerate(
        dia_mz_cycle
    ):
        push_index = self_connection_index
        push_index += cycle_index * cycle_length
        push_index += zeroth_frame * scan_max_index
        if not (0 <= push_index < len(selected_push_indices)):
            continue
        selected_push_indices[push_index] = -1
        if (not use_ms1) and (lower_mz == -1):
            continue
        if (not use_ms2) and (lower_mz != -1):
            continue
        start_index = push_indptr[push_index]
        end_index = push_indptr[push_index + 1]
        if (end_index - start_index) >= min_peaks:
            current_min_intensity = min_intensity_value
            max_index = -1
            for index, intensity in enumerate(
                intensity_values[start_index:end_index],
                start_index
            ):
                mz_value = mz_values[tof_indices[index]]
                if (not use_precursor_range) and (lower_mz < mz_value < upper_mz):
                    continue
                if intensity >= current_min_intensity:
                    max_index = index
            selected_push_indices[push_index] = max_index


def push_index_to_cycle_index(
    push_index,
    dia_mz_cycle,
    zeroth_frame,
    scan_max_index,
):
    cycle_length = len(dia_mz_cycle)
    push_index_ = push_index - zeroth_frame * scan_max_index
    cycle_index = push_index_ // cycle_length
    cycle_offset = push_index_ % cycle_length
    return cycle_index, cycle_offset


@alphatims.utils.njit(nogil=True)
def push_precursor_borders(
    push_indices,
    push_indptr,
    tof_indices,
    mz_values,
    dia_mz_cycle,
    zeroth_frame,
    scan_max_index,
    precursor_frame,
):
    potential_precursors = np.zeros(
        (len(push_indices), 2),
        dtype=np.int64
    )
    cycle_length = len(dia_mz_cycle)
    for index, push_index in enumerate(push_indices):
        if push_index < zeroth_frame * scan_max_index:
            continue
        push_index_ = push_index - zeroth_frame * scan_max_index
        cycle_index = push_index_ // cycle_length
        cycle_offset = push_index_ % cycle_length
        precursor_index = cycle_offset % scan_max_index + precursor_frame * scan_max_index
        precursor_index += cycle_index * cycle_length
        precursor_index += zeroth_frame * scan_max_index
        mz_borders = dia_mz_cycle[cycle_offset]
        precursor_start = push_indptr[precursor_index]
        precursor_end = push_indptr[precursor_index + 1]
        offsets = mz_values[tof_indices[precursor_start: precursor_end]]
        potential_precursors[index] = precursor_start + np.searchsorted(offsets, mz_borders)
    return potential_precursors


@alphatims.utils.pjit
def find_best_peptide(
    selected_index,
    potential_precursors,
    spectra_of_interest,
    precursor_mzs,
    y_mzs,
    b_mzs,
    frag_start_idxs,
    frag_end_idxs,
    mz_values,
    tof_indices,
    push_indptr,
    intensity_values,
    max_indices,
    max_counts,
    max_precursor_mzs,
    selected_ms1_ions,
    fragment_ppm=50,
    precursor_ppm=50,
):
    selected_precursors = potential_precursors[selected_index]
    selected_fragments = (
        push_indptr[spectra_of_interest[selected_index]],
        push_indptr[spectra_of_interest[selected_index] + 1]
    )
    fragment_mzs = mz_values[
        tof_indices[
            selected_fragments[0]: selected_fragments[1]
        ]
    ]
    if selected_precursors[0] == selected_precursors[1]:
        max_indices[selected_index] = -1
        max_counts[selected_index] = 0
        return
    precursor_intensities = intensity_values[
        selected_precursors[0]: selected_precursors[1]
    ]
    selected_ms1_ion = np.argmax(
        precursor_intensities
    ) + selected_precursors[0]
    precursor_mz = mz_values[
        tof_indices[
            selected_ms1_ion
        ]
    ]
    lower_bound = np.searchsorted(
        precursor_mzs,
        precursor_mz / (1 + 10**-6 * precursor_ppm),
    )
    upper_bound = np.searchsorted(
        precursor_mzs,
        precursor_mz * (1 + 10**-6 * precursor_ppm),
    )
    max_hit_count = 0
    max_index = -1
    for index in range(lower_bound, upper_bound):
        frag_start_idx = frag_start_idxs[index]
        frag_end_idx = frag_end_idxs[index]
        if frag_start_idx == frag_end_idx:
            continue
        y_hit_count = rough_match(
            fragment_mzs,
            y_mzs[frag_start_idx: frag_end_idx][::-1],
            fragment_ppm,
        )
        b_hit_count = rough_match(
            fragment_mzs,
            b_mzs[frag_start_idx: frag_end_idx],
            fragment_ppm,
        )
        hit_count = y_hit_count + b_hit_count
        if hit_count > max_hit_count:
            max_hit_count = hit_count
            max_index = index
    max_indices[selected_index] = max_index
    max_counts[selected_index] = max_hit_count
    max_precursor_mzs[selected_index] = precursor_mz
    selected_ms1_ions[selected_index] = selected_ms1_ion
    # return max_index, max_hit_count


@alphatims.utils.njit(nogil=True)
def rough_match(
    fragment_mzs,
    database_mzs,
    fragment_ppm,
):
    fragment_index = 0
    database_index = 0
    hit_count = 0
    while (fragment_index < len(fragment_mzs)) and (database_index < len(database_mzs)):
        fragment_mz = fragment_mzs[fragment_index]
        database_mz = database_mzs[database_index]
        if fragment_mz < (database_mz / (1 + 10**-6 * fragment_ppm)):
            fragment_index += 1
        elif database_mz < (fragment_mz / (1 + 10**-6 * fragment_ppm)):
            database_index += 1
        else:
            hit_count += 1
            fragment_index += 1
            database_index += 1
    return hit_count


def first_search(
    dia_data,
    y_ions,
    b_ions,
    predicted_library_df,
    scan_tolerance=6,
    multiple_frames_per_cycle=True,
    ms1=True,
    ms2=True,
    cycle_tolerance=3,
    precursor_ppm=50,
    fragment_ppm=50,
    precursor_frame=0,
    train_fdr_level=0.5,
    min_peaks=10,
    min_intensity_value=1000,
    use_ms1=False,
    use_ms2=True,
    use_precursor_range=False,
):
    cycle_count = len(dia_data.push_indptr) // len(dia_data.dia_mz_cycle)
    selected_push_indices = np.empty_like(dia_data.push_indptr)[:-1]
    selected_push_indices[:dia_data.zeroth_frame * dia_data.scan_max_index] = -1
    maximum_push_indices = np.ones_like(dia_data.push_indptr, dtype=np.bool)[:-1]
    maximum_push_indices[:dia_data.zeroth_frame * dia_data.scan_max_index] = False
    find_max_intensity_per_push(
        range(cycle_count + 1),
        dia_data.dia_mz_cycle,
        dia_data.zeroth_frame,
        dia_data.scan_max_index,
        dia_data.push_indptr,
        dia_data.intensity_values,
        dia_data.tof_indices,
        dia_data.mz_values,
        selected_push_indices,
        min_peaks,
        min_intensity_value,
        use_ms1,
        use_ms2,
        use_precursor_range,
    )
    connection_indptr, connection_indices = alpharaw.smoothing.get_connections_within_cycle(
        scan_tolerance=scan_tolerance,
        scan_max_index=dia_data.scan_max_index,
        dia_mz_cycle=dia_data.dia_mz_cycle,
        multiple_frames=multiple_frames_per_cycle,
        ms1=ms1,
        ms2=ms2,
    )
    find_maximal_pushes(
        range(cycle_count + 1),
        connection_indptr,
        connection_indices,
        cycle_tolerance,
        dia_data.zeroth_frame,
        dia_data.scan_max_index,
        dia_data.intensity_values,
        selected_push_indices,
        maximum_push_indices,
    )
    spectra_of_interest = np.flatnonzero(maximum_push_indices)
    potential_precursors = push_precursor_borders(
        spectra_of_interest,
        dia_data.push_indptr,
        dia_data.tof_indices,
        dia_data.mz_values,
        dia_data.dia_mz_cycle,
        dia_data.zeroth_frame,
        dia_data.scan_max_index,
        precursor_frame,
    )
    max_indices = np.zeros_like(spectra_of_interest)
    max_counts = np.zeros_like(spectra_of_interest)
    max_precursor_mzs = np.zeros_like(spectra_of_interest, dtype=np.float64)
    selected_ms1_ions = np.zeros_like(spectra_of_interest)
    find_best_peptide(
        range(len(spectra_of_interest)),
        potential_precursors,
        spectra_of_interest,
        predicted_library_df.precursor_mz.values,
        y_ions,
        b_ions,
        predicted_library_df.frag_start_idx.values,
        predicted_library_df.frag_end_idx.values,
        dia_data.mz_values,
        dia_data.tof_indices,
        dia_data.push_indptr,
        dia_data.intensity_values,
        max_indices,
        max_counts,
        max_precursor_mzs,
        selected_ms1_ions,
        precursor_ppm,
        fragment_ppm,
    )
    selection = max_counts >= 1
    selected = max_indices[selection]
    data_df = dia_data.as_dataframe(dia_data.push_indptr[spectra_of_interest[selection]])
    lib_df = predicted_library_df.iloc[selected]
    lib_df["rt_experimental"] = np.copy(data_df.rt_values_min.values)
    lib_df["mobility_experimental"] = np.copy(data_df.mobility_values.values)
    lib_df["mz_experimental"] = max_precursor_mzs[selection]
    lib_df["raw_ms1_index"] = selected_ms1_ions[selection]
    lib_df["count"] = max_counts[selection]
    lib_df["ppm"] = (lib_df.mz_experimental - lib_df.precursor_mz) / lib_df.precursor_mz * 10**6
    lib_df["delta_im"] = lib_df.mobility_experimental - lib_df.mobility_pred
    lib_df.reset_index(level=0, inplace=True)
    lib_df.rename(
        columns={
            "index": "mz_sorted_index",
        },
        inplace=True,
    )
    lib_df["decoy"] = lib_df["decoy"].astype(np.bool)
    lib_df["target"] = ~lib_df["decoy"]
    lib_df["mz_pred"] = lib_df.precursor_mz
    lib_df["im_pred"] = lib_df.mobility_pred
    lib_df["im_experimental"] = lib_df.mobility_experimental
    return train_and_score(
        lib_df,
        ["count", "ppm", "delta_im"],
        train_fdr_level=train_fdr_level,
    )

def train_and_score(
    scores_df,
    features,
    train_fdr_level: float = 0.1,
    ini_score: str = "count",
    min_train: int = 1000,
    test_size: float = 0.8,
    max_depth: list = [5, 25, 50],
    max_leaf_nodes: list = [150, 200, 250],
    n_jobs: int = -1,
    scoring: str = 'accuracy',
    plot: bool = False,
    random_state: int = 42,
):
    df = scores_df.copy()
    cv = alphadia.library.train_RF(
        df,
        features,
        train_fdr_level=train_fdr_level,
        ini_score=ini_score,
        min_train=min_train,
        test_size=test_size,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        n_jobs=n_jobs,
        scoring=scoring,
        plot=plot,
        random_state=random_state,
    )
    df['score'] = cv.predict_proba(df[features])[:, 1]
    return alphadia.library.get_q_values(df, "score", 'decoy', drop=True)


@alphatims.utils.pjit
def filter_precursor_candidates(
    selected_index,
    potential_precursors,
    spectra_of_interest,
    precursor_mzs,
    y_mzs,
    b_mzs,
    frag_start_idxs,
    frag_end_idxs,
    mz_values,
    tof_indices,
    push_indptr,
    intensity_values,
    max_indices,
    max_counts,
    max_precursor_mzs,
    fragment_ppm=50,
    precursor_ppm=50,
):
    selected_precursors = potential_precursors[selected_index]
    selected_fragments = (
        push_indptr[spectra_of_interest[selected_index]],
        push_indptr[spectra_of_interest[selected_index] + 1]
    )
    fragment_mzs = mz_values[
        tof_indices[
            selected_fragments[0]: selected_fragments[1]
        ]
    ]
    if selected_precursors[0] == selected_precursors[1]:
        max_indices[selected_index] = -1
        max_counts[selected_index] = 0
        return
    precursor_intensities = intensity_values[
        selected_precursors[0]: selected_precursors[1]
    ]
    precursor_mz = mz_values[
        tof_indices[
            np.argmax(precursor_intensities) + selected_precursors[0]
        ]
    ]
    lower_bound = np.searchsorted(
        precursor_mzs,
        precursor_mz / (1 + 10**-6 * precursor_ppm),
    )
    upper_bound = np.searchsorted(
        precursor_mzs,
        precursor_mz * (1 + 10**-6 * precursor_ppm),
    )
    max_hit_count = 0
    max_index = -1
    for index in range(lower_bound, upper_bound):
        frag_start_idx = frag_start_idxs[index]
        frag_end_idx = frag_end_idxs[index]
        if frag_start_idx == frag_end_idx:
            continue
        y_hit_count = rough_match(
            fragment_mzs,
            y_mzs[frag_start_idx: frag_end_idx][::-1],
            fragment_ppm,
        )
        b_hit_count = rough_match(
            fragment_mzs,
            b_mzs[frag_start_idx: frag_end_idx],
            fragment_ppm,
        )
        hit_count = y_hit_count + b_hit_count
        if hit_count > max_hit_count:
            max_hit_count = hit_count
            max_index = index
    max_indices[selected_index] = max_index
    max_counts[selected_index] = max_hit_count
    max_precursor_mzs[selected_index] = precursor_mz
    # return max_index, max_hit_count


def calibrate_hits(
    first_hits,
    n_neighbors=4,
    test_size=0.8,
    fdr=0.01,
):
    lib_df = first_hits[first_hits.q_value < fdr]
    for dimension in ["rt", "im"]:
        X = lib_df[f"{dimension}_pred"].values.reshape(-1, 1)
        y = lib_df[f"{dimension}_experimental"].values
        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = sklearn.model_selection.train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=0,
        )
        neigh = sklearn.neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights="distance",
            n_jobs=alphatims.utils.set_threads(alphatims.utils.MAX_THREADS)
        )
        neigh.fit(
            X_train,
            y_train,
        )
        first_hits[f"{dimension}_calibrated"] = neigh.predict(
            first_hits[f"{dimension}_pred"].values.reshape(-1, 1)
        )
        first_hits[f"{dimension}_diff"] = first_hits[f"{dimension}_experimental"] - first_hits[f"{dimension}_calibrated"]
    return first_hits


@alphatims.utils.pjit
def best_transitions(
    index,
    start_indices,
    end_indices,
    fragment_intensities,
    best_transitions,
    remove_outer_ions,
):
    start = start_indices[index] + remove_outer_ions
    end = end_indices[index] - remove_outer_ions
    if (start + remove_outer_ions) < (end - remove_outer_ions):
        start += remove_outer_ions
        end -= remove_outer_ions
    max_index = start + np.argmax(fragment_intensities[start: end])
    best_transitions[index] = max_index


@alphatims.utils.njit
def merge_best_transitions(
    b_intensities,
    b_transitions,
    b_mzs,
    y_intensities,
    y_transitions,
    y_mzs,
):
    # best_transation_indices = np.empty_like(y_transitions)
    best_transation_mzs = np.empty_like(y_transitions, dtype=np.float64)
    for index, y_index in enumerate(y_transitions):
        b_index = b_transitions[index]
        if b_intensities[b_index] > y_intensities[y_index]:
            # best_transation_indices[index] = b_index
            best_transation_mzs[index] = b_mzs[b_index]
        else:
            # best_transation_indices[index] = y_index
            best_transation_mzs[index] = y_mzs[y_index]
    # return best_transation_indices, best_transation_mzs
    return best_transation_mzs


def find_seeds(
    dia_data,
    predicted_library_df,
    cycle_index,
    offsets,
    idxs,
    library_precursor_rt_values,
    library_precursor_im_values,
    rt_diff_std,
    im_diff_std,
    precursor_cycle,
    precursor_frame,
    min_fragments,
    potential_peaks,
):
    import multiprocessing

    def starfunc(cycle_index):
        return find_candidates(
            cycle_index,
            offsets,
            idxs,
            library_precursor_rt_values,
            library_precursor_im_values,
            predicted_library_df.precursor_index_lower.values,
            predicted_library_df.precursor_index_upper.values,
            rt_diff_std,
            im_diff_std,
            dia_data.push_indptr,
            dia_data.tof_indices,
            dia_data.mz_values,
            dia_data.rt_values / 60,
            dia_data.mobility_values,
            precursor_cycle,
            dia_data.zeroth_frame,
            dia_data.scan_max_index,
            precursor_frame,
            # final_candidates,
            min_fragments,
            potential_peaks,
        )

    iterable = range(len(dia_data.push_indptr) // len(dia_data.dia_mz_cycle) + 1)
    # iterable = range(500, 520)
    seeds = []
    counts = [[0] * (1 + dia_data.zeroth_frame * dia_data.scan_max_index)]
    with multiprocessing.pool.ThreadPool(alphatims.utils.MAX_THREADS) as pool:
        for cycle_index, (
            push_counts,
            seed_candidates,
        ) in alphatims.utils.progress_callback(
            enumerate(pool.imap(starfunc, iterable)),
            total=len(iterable),
            include_progress_callback=True
        ):
            counts.append(push_counts)
            seeds.append(seed_candidates)
    # return seeds, counts
    return (
        np.cumsum(np.concatenate(counts))[:len(dia_data.push_indptr)],
        np.concatenate(seeds)
    )


@alphatims.utils.njit(nogil=True)
def find_candidates(
    cycle_index,
    library_precursor_offsets,
    library_precursor_indices,
    library_precursor_rt_values,
    library_precursor_im_values,
    library_lower_indices,
    library_upper_indices,
    rt_tolerance,
    im_tolerance,
    push_indptr,
    tof_indices,
    mz_values,
    rt_values,
    im_values,
    precursor_cycle,
    zeroth_frame,
    scan_max_index,
    precursor_frame,
    # final_candidates,
    min_fragments,
    potential_peaks,
):
    cycle_length = len(precursor_cycle)
    push_offset = cycle_length * cycle_index + zeroth_frame * scan_max_index
    frame = push_offset // scan_max_index
    rt = rt_values[frame]
    start_offsets, end_offsets = filter_offsets_by_rt(
        library_precursor_offsets,
        library_precursor_rt_values,
        rt,
        rt_tolerance,
    )
    seed_candidates = []
    push_counts = np.zeros(len(precursor_cycle), dtype=np.int64)
    # peptide_buffer = np.zeros_like(library_lower_indices, dtype=np.bool_)
    for cycle_offset, (low_precursor, high_precursor) in enumerate(
        precursor_cycle
    ):
        if low_precursor == high_precursor:
            continue
        scan_index = cycle_offset % scan_max_index
        im = im_values[scan_index]
        precursor_push_index = (
            push_offset + precursor_frame * scan_max_index + scan_index
        )
        if precursor_push_index > len(push_indptr):
            continue
        push_index = push_offset + cycle_offset
        if push_index > len(push_indptr):
            continue
        precursor_push_index_start = push_indptr[precursor_push_index]
        precursor_push_index_end = push_indptr[precursor_push_index + 1]
        if precursor_push_index_start == precursor_push_index_end:
            continue
        push_index_start = push_indptr[push_index]
        push_index_end = push_indptr[push_index + 1]
        if (push_index_end - push_index_start) < min_fragments:
            continue
        candidate_precursors = []
        for index, tof_index in enumerate(
            tof_indices[push_index_start: push_index_end],
            push_index_start,
        ):
            if not potential_peaks[index]:
                continue
            low_index = start_offsets[tof_index]
            high_index = end_offsets[tof_index]
            candidate_ims = library_precursor_im_values[
                low_index: high_index
            ]
            # final_candidates[cycle_index] += high_index - low_index
            for candidate_index, candidate_im in enumerate(
                candidate_ims,
                low_index
            ):
                candidate_precursor = library_precursor_indices[candidate_index]
                if library_upper_indices[candidate_precursor] < low_precursor:
                    continue
                if library_lower_indices[candidate_precursor] > high_precursor:
                    continue
                # if peptide_buffer[candidate_precursor]:
                #     continue
                # if final_candidates[candidate_precursor]:
                #     continue
                if abs(candidate_im - im) < im_tolerance:
                    candidate_precursors.append(candidate_precursor)
                    # final_candidates[candidate_precursor] = True
        if len(candidate_precursors) == 0:
            continue
        # return locals()
        candidate_precursors = sorted(candidate_precursors)
        candidate_index = 0
        for precursor_tof in tof_indices[
            precursor_push_index_start: precursor_push_index_end
        ]:
            while candidate_index < len(candidate_precursors):
                candidate_precursor = candidate_precursors[candidate_index]
                lower_tof = library_lower_indices[candidate_precursor]
                if precursor_tof < lower_tof:
                    break
                upper_tof = library_upper_indices[candidate_precursor]
                if precursor_tof < upper_tof:
                    # check other fragments?
                    # peptide_buffer[candidate_precursor] = True
                    # final_candidates[candidate_precursor] = True
                    seed_candidates.append(candidate_precursor)
                    push_counts[cycle_offset] += 1
                    # final_candidates[cycle_index] += 1
                candidate_index += 1
                # else:
                #     final_candidates[candidate_precursor] = True
                #     candidate_index += 1
    # for candidate_index, candidate_precursor in enumerate(peptide_buffer):
    #     if candidate_precursor:
    #         final_candidates[candidate_index] = True
    return push_counts, np.array(seed_candidates)


@alphatims.utils.njit(nogil=True)
def filter_offsets_by_rt(
    offsets,
    library_precursor_rt_values,
    rt,
    rt_tolerance,
):
    start_offsets = np.empty(len(offsets) - 1, dtype=offsets.dtype)
    end_offsets = np.empty(len(offsets) - 1, dtype=offsets.dtype)
    for index, start in enumerate(offsets[:-1]):
        end = offsets[index + 1]
        start_offsets[index], end_offsets[index] = start + np.searchsorted(
            library_precursor_rt_values[start: end],
            [rt - rt_tolerance, rt + rt_tolerance]
        )
    return start_offsets, end_offsets


def remove_unreachable_precursors(
    predicted_library_df,
    dia_data,
):
    lower_mz_index, upper_mz_index = np.searchsorted(
        predicted_library_df.precursor_mz.values,
        [dia_data.quad_mz_min_value, dia_data.quad_mz_max_value],
    )
    return predicted_library_df[lower_mz_index: upper_mz_index].reset_index(
        drop=True,
    )



@alphatims.utils.pjit
def annotate_seeds(
    push_index,
    y_mzs,
    b_mzs,
    frag_start_idxs,
    frag_end_idxs,
    mz_values,
    tof_indices,
    push_indptr,
    # max_indices,
    # max_counts,
    # max_precursor_mzs,
    push_counts,
    seed_candidates,
    fragment_ppm,
    # ppm_offset,
    hit_counts,
    intensity_values,
    dia_mz_cycle,
    tof_tolerance,
    scan_max_index,
    tof_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    is_signal,
):
    seed_start = push_counts[push_index]
    seed_end = push_counts[push_index + 1]
    if seed_start == seed_end:
        return
    push_start = push_indptr[push_index]
    push_end = push_indptr[push_index + 1]
    fragment_tofs = tof_indices[push_start: push_end]
    fragment_tofs, fragment_intensities = alpharaw.smoothing.merge_pushes2(
        push_index,
        push_indptr,
        tof_indices,
        intensity_values,
        dia_mz_cycle,
        tof_tolerance,
        scan_max_index,
        tof_max_index,
        zeroth_frame,
        connection_counts,
        connections,
        cycle_tolerance,
        is_signal,
        # mz_values,
    )
    fragment_mzs = mz_values[fragment_tofs]
    # smooth from connections?
    for seed_index, seed in enumerate(
        seed_candidates[seed_start: seed_end],
        seed_start
    ):
        frag_start_idx = frag_start_idxs[seed]
        frag_end_idx = frag_end_idxs[seed]
        if frag_start_idx == frag_end_idx:
            continue
        y_hit_count = rough_match(
            fragment_mzs,
            y_mzs[frag_start_idx: frag_end_idx][::-1],
            fragment_ppm,
        )
        b_hit_count = rough_match(
            fragment_mzs,
            b_mzs[frag_start_idx: frag_end_idx],
            fragment_ppm,
        )
        hit_count = y_hit_count + b_hit_count
        hit_counts[seed_index] = hit_count


@alphatims.utils.pjit
# @alphatims.utils.njit
def annotate_seeds2(
    push_index,
    y_mzs,
    b_mzs,
    frag_start_idxs,
    frag_end_idxs,
    mz_values,
    tof_indices,
    push_indptr,
    # max_indices,
    # max_counts,
    # max_precursor_mzs,
    push_counts,
    seed_candidates,
    fragment_ppm,
    # ppm_offset,
    b_hit_counts,
    y_hit_counts,
    summed_b_hit_ints,
    summed_y_hit_ints,
    mean_b_hit_ints,
    mean_y_hit_ints,
    std_b_hit_ints,
    std_y_hit_ints,
    intensity_values,
    dia_mz_cycle,
    tof_tolerance,
    scan_max_index,
    tof_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    is_signal,
):
    seed_start = push_counts[push_index]
    seed_end = push_counts[push_index + 1]
    if seed_start == seed_end:
        return
    # push_start = push_indptr[push_index]
    # push_end = push_indptr[push_index + 1]
    # fragment_tofs = tof_indices[push_start: push_end]
    fragment_tofs, fragment_intensities = alpharaw.smoothing.merge_pushes2(
        push_index,
        push_indptr,
        tof_indices,
        intensity_values,
        dia_mz_cycle,
        tof_tolerance,
        scan_max_index,
        tof_max_index,
        zeroth_frame,
        connection_counts,
        connections,
        cycle_tolerance,
        is_signal,
        # mz_values,
    )
    fragment_mzs = mz_values[fragment_tofs]
    # smooth from connections?
    for seed_index, seed in enumerate(
        seed_candidates[seed_start: seed_end],
        seed_start
    ):
        frag_start_idx = frag_start_idxs[seed]
        frag_end_idx = frag_end_idxs[seed]
        if frag_start_idx == frag_end_idx:
            continue
        b_hits = rough_match2(
            fragment_mzs,
            y_mzs[frag_start_idx: frag_end_idx][::-1],
            fragment_ppm,
        )
        y_hits = rough_match2(
            fragment_mzs,
            b_mzs[frag_start_idx: frag_end_idx],
            fragment_ppm,
        )
        b_hit_counts[seed_index] = len(b_hits)
        y_hit_counts[seed_index] = len(y_hits)
        if len(b_hits) > 0:
            mean_b_hit_ints[seed_index] = sum(b_hits) / len(b_hits)
            for i in b_hits:
                std_b_hit_ints[seed_index] += (i - mean_b_hit_ints[seed_index])**2
            std_b_hit_ints[seed_index] /= len(b_hits)
        if len(y_hits) > 0:
            mean_y_hit_ints[seed_index] = sum(y_hits) / len(y_hits)
            for i in y_hits:
                std_y_hit_ints[seed_index] += (i - mean_y_hit_ints[seed_index])**2
            std_y_hit_ints[seed_index] /= len(y_hits)


@alphatims.utils.njit(nogil=True)
def rough_match2(
    fragment_mzs,
    database_mzs,
    fragment_ppm,
):
    fragment_index = 0
    database_index = 0
    hits = []
    while (fragment_index < len(fragment_mzs)) and (database_index < len(database_mzs)):
        fragment_mz = fragment_mzs[fragment_index]
        database_mz = database_mzs[database_index]
        if fragment_mz < (database_mz / (1 + 10**-6 * fragment_ppm)):
            fragment_index += 1
        elif database_mz < (fragment_mz / (1 + 10**-6 * fragment_ppm)):
            database_index += 1
        else:
            ppm = (fragment_mz - database_mz) / fragment_mz * 10**6
            hits.append(ppm)
            fragment_index += 1
            database_index += 1
    return hits


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


@alphatims.utils.pjit
def annotate(
    index,
    frag_start_idx,
    frag_end_idx,
    frag_indices,
    indptr,
    precursor_indices,
    mz_values,
    tof_indices,
    fragment_ppm,
    lower,
    upper,
    y_mzs,
    b_mzs,
    max_hit_counts,
    max_db_indices,
    min_size,
):
    start = indptr[index]
    end = indptr[index + 1]
    if (end - start) < min_size:
        return
    frags = frag_indices[start: end]
    fragment_tofs = np.sort(tof_indices[frags])
    fragment_mzs = mz_values[fragment_tofs]
    for db_index in range(lower[index], upper[index]):
        frag_start = frag_start_idx[db_index]
        frag_end = frag_end_idx[db_index]
        # b_hits = rough_match2(
        y_hits = rough_match2_count_only(
            fragment_mzs,
            y_mzs[frag_start: frag_end][::-1],
            fragment_ppm,
        )
        # y_hits = rough_match2(
        b_hits = rough_match2_count_only(
            fragment_mzs,
            b_mzs[frag_start: frag_end],
            fragment_ppm,
        )
        # hit_count = len(b_hits) + len(y_hits)
        hit_count = b_hits + y_hits
        # if hit_count == max_hit_counts[index]:
        #     max_db_indices[index] = db_index
        if hit_count > max_hit_counts[index]:
            max_db_indices[index] = db_index
            max_hit_counts[index] = hit_count


@alphatims.utils.njit(nogil=True)
def annotate_pool(
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
    frags = frag_indices[start: end]
    fragment_tofs = tof_indices[frags]
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


@alphatims.utils.njit
def trim_seeds(
    push_counts,
    seed_candidates,
    hit_counts,
    threshold=3,
):
    new_push_counts = np.empty_like(push_counts)
    new_push_counts[0] = 0
    total_sum = 0
    for index, start in enumerate(push_counts[:-1]):
        end = push_counts[index + 1]
        total_sum += np.sum(hit_counts[start: end] >= threshold)
        new_push_counts[index + 1] = total_sum
    new_seed_candidates = seed_candidates[hit_counts >= threshold]
    return new_push_counts, new_seed_candidates





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
    fragment_tofs = frag_indices[start: end]
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

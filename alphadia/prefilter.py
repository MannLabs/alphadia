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


@alphatims.utils.njit
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
    for index, push_index in enumerate(push_indices):
        if push_index < zeroth_frame * scan_max_index:
            continue
        cycle_length = len(dia_mz_cycle)
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


@alphatims.utils.njit
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
    return alphadia.library.get_q_values(df, "score", 'decoy')


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
    for dimension in ["rt", "im", "mz"]:
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
    first_hits["ppm_diff"] = first_hits["mz_diff"] / first_hits["mz_pred"] * 10**6
    return first_hits

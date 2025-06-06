# native imports

# alphadia imports
import numba as nb

# alpha family imports
# third party imports
import numpy as np

from alphadia import utils
from alphadia.numba import numeric
from alphadia.scoring.utils import (
    correlation_coefficient,
    median_axis,
    normalize_profiles,
)
from alphadia.utils import USE_NUMBA_CACHING


@nb.njit(cache=USE_NUMBA_CACHING)
def weighted_center_of_mass(
    single_dense_representation,
):
    intensity = [0.0]

    scans, frames = np.nonzero(single_dense_representation > 0)

    if len(scans) == 0:
        return 0, 0, 0, 0

    for scan, frame in zip(scans, frames):
        intensity.append(single_dense_representation[scan, frame])

    intensity_arr = np.array(intensity)[1:]
    intensity_sum = np.sum(intensity_arr)

    scan_mean = (
        np.sum(scans * intensity_arr) / intensity_sum if intensity_sum > 0 else 0
    )
    frame_mean = (
        np.sum(frames * intensity_arr) / intensity_sum if intensity_sum > 0 else 0
    )
    frame_var_weighted = np.sum((frames - frame_mean) ** 2 * intensity_arr)
    scan_var_weighted = np.sum((scans - scan_mean) ** 2 * intensity_arr)

    frame_var_weighted = frame_var_weighted / intensity_sum if intensity_sum > 0 else 0
    scan_var_weighted = scan_var_weighted / intensity_sum if intensity_sum > 0 else 0

    return scan_mean, frame_mean, scan_var_weighted, frame_var_weighted


@nb.njit(cache=USE_NUMBA_CACHING)
def weighted_center_of_mass_1d(
    dense_representation,
):
    scan = np.zeros(dense_representation.shape[0])
    frame = np.zeros(dense_representation.shape[0])
    frame_var_weighted = np.zeros(dense_representation.shape[0])
    scan_var_weighted = np.zeros(dense_representation.shape[0])

    for i in range(dense_representation.shape[0]):
        (
            scan[i],
            frame[i],
            scan_var_weighted[i],
            frame_var_weighted[i],
        ) = weighted_center_of_mass(dense_representation[i])
    return scan, frame, scan_var_weighted, frame_var_weighted


@nb.njit(cache=USE_NUMBA_CACHING)
def weighted_center_mean(single_dense_representation, scan_center, frame_center):
    values = 0
    weights = 0

    scans, frames = np.nonzero(single_dense_representation > 0)
    if len(scans) == 0:
        return 0

    for scan, frame in zip(scans, frames):
        value = single_dense_representation[scan, frame]
        distance = np.sqrt((scan - scan_center) ** 2 + (frame - frame_center) ** 2)
        weight = np.exp(-0.1 * distance)
        values += value * weight
        weights += weight

    return values / weights if weights > 0 else 0


@nb.njit(cache=USE_NUMBA_CACHING)
def weighted_center_mean_2d(dense_representation, scan_center, frame_center):
    values = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    for i in range(dense_representation.shape[0]):
        for j in range(dense_representation.shape[1]):
            values[i, j] = weighted_center_mean(
                dense_representation[i, j], scan_center[i, j], frame_center[i, j]
            )

    return values


float_array = nb.types.float32[:]  # TODO duplicated in plexscoring


@nb.njit(cache=USE_NUMBA_CACHING)
def cosine_similarity_a1(template_intensity, fragments_intensity):
    fragment_norm = np.sqrt(np.sum(np.power(fragments_intensity, 2), axis=-1))
    template_norm = np.sqrt(np.sum(np.power(template_intensity, 2), axis=-1))

    div = (fragment_norm * template_norm) + 0.0001

    return np.sum(fragments_intensity * template_intensity, axis=-1) / div


@nb.njit(cache=USE_NUMBA_CACHING)
def frame_profile_2d(x):
    return np.sum(x, axis=2)


@nb.njit(cache=USE_NUMBA_CACHING)
def frame_profile_1d(x):
    return np.sum(x, axis=1)


@nb.njit(cache=USE_NUMBA_CACHING)
def scan_profile_2d(x):
    return np.sum(x, axis=3)


@nb.njit(cache=USE_NUMBA_CACHING)
def scan_profile_1d(x):
    return np.sum(x, axis=2)


@nb.njit(cache=USE_NUMBA_CACHING)
def or_envelope_1d(x):
    res = x.copy()
    for a0 in range(x.shape[0]):
        for i in range(1, x.shape[1] - 1):
            if (x[a0, i] < x[a0, i - 1]) or (x[a0, i] < x[a0, i + 1]):
                res[a0, i] = (x[a0, i - 1] + x[a0, i + 1]) / 2
    return res


@nb.njit(cache=USE_NUMBA_CACHING)
def or_envelope_2d(x):
    res = x.copy()
    for a0 in range(x.shape[0]):
        for a1 in range(x.shape[1]):
            for i in range(1, x.shape[2] - 1):
                if (x[a0, a1, i] < x[a0, a1, i - 1]) or (
                    x[a0, a1, i] < x[a0, a1, i + 1]
                ):
                    res[a0, a1, i] = (x[a0, a1, i - 1] + x[a0, a1, i + 1]) / 2
    return res


@nb.njit(inline="always", cache=USE_NUMBA_CACHING)
def _odd_center_envelope(x: np.ndarray):
    """
    Applies an interference correction envelope to a collection of odd-length 1D arrays.
    Numba function which operates in place.

    Parameters
    ----------
    x: np.ndarray
        Array of shape (a, b) where a is the number of arrays and b is the length of each array.
        It is mandatory that dimension b is odd.

    """
    center_index = x.shape[1] // 2

    for a0 in range(x.shape[0]):
        left_intensity = (x[a0, center_index - 1] + x[a0, center_index]) * 0.5
        right_intensity = (x[a0, center_index + 1] + x[a0, center_index]) * 0.5

        for i in range(1, center_index + 1):
            x[a0, center_index - i] = min(left_intensity, x[a0, center_index - i])

            left_intensity = (
                x[a0, center_index - i] + x[a0, center_index - i + 1]
            ) * 0.5

            x[a0, center_index + i] = min(right_intensity, x[a0, center_index + i])
            right_intensity = (
                x[a0, center_index + i] + x[a0, center_index + i - 1]
            ) * 0.5


@nb.njit(inline="always", cache=USE_NUMBA_CACHING)
def _even_center_envelope(x: np.ndarray):
    """
    Applies an interference correction envelope to a collection of even-length 1D arrays.
    Numba function which operates in place.

    Parameters
    ----------
    x: np.ndarray
        Array of shape (a, b) where a is the number of arrays and b is the length of each array.
        It is mandatory that dimension b is even.

    """
    center_index_right = x.shape[1] // 2
    center_index_left = center_index_right - 1

    for a0 in range(x.shape[0]):
        left_intensity = x[a0, center_index_left]
        right_intensity = x[a0, center_index_right]

        for i in range(1, center_index_left + 1):
            x[a0, center_index_left - i] = min(
                left_intensity, x[a0, center_index_left - i]
            )

            left_intensity = (
                x[a0, center_index_left - i] + x[a0, center_index_left - i + 1]
            ) * 0.5

            x[a0, center_index_right + i] = min(
                right_intensity, x[a0, center_index_right + i]
            )
            right_intensity = (
                x[a0, center_index_right + i] + x[a0, center_index_right + i - 1]
            ) * 0.5


@nb.njit(cache=USE_NUMBA_CACHING)
def center_envelope_1d(x: np.ndarray):
    """
    Applies an interference correction envelope to a collection of 1D arrays.
    Numba function which operates in place.

    Parameters
    ----------
    x: np.ndarray
        Array of shape (a, b) where a is the number of arrays and b is the length of each array.
        It is mandatory that dimension b is odd.

    """

    is_even = x.shape[1] % 2 == 0

    if is_even:
        _even_center_envelope(x)
    else:
        _odd_center_envelope(x)


@nb.njit(cache=USE_NUMBA_CACHING)
def weighted_mean_a1(array, weight_mask):
    """
    takes an array of shape (a, b) and a mask of shape (a, b)
    and returns an array of shape (a) where each element is the weighted mean of the corresponding masked row in the array.

    Parameters
    ----------

    array: np.ndarray
        array of shape (a, b)

    weight_mask: np.ndarray
        array of shape (a, b)

    Returns
    -------
    np.ndarray
        array of shape (a)
    """

    mask = weight_mask > 0

    mean = np.zeros(array.shape[0])
    for i in range(array.shape[0]):
        masked_array = array[i][mask[i]]
        if len(masked_array) > 0:
            local_weight_mask = weight_mask[i][mask[i]] / np.sum(
                weight_mask[i][mask[i]]
            )
            mean[i] = np.sum(masked_array * local_weight_mask)
        else:
            mean[i] = 0
    return mean


@nb.njit(cache=USE_NUMBA_CACHING)
def precursor_features(
    isotope_mz: np.ndarray,
    isotope_intensity: np.ndarray,
    dense_precursors: np.ndarray,
    observation_importance,
    template: np.ndarray,
    feature_array: np.ndarray,
):
    n_isotopes = isotope_intensity.shape[0]
    n_observations = dense_precursors.shape[2]

    # ============= PRECURSOR FEATURES =============

    # (1, n_observations)
    observation_importance_reshaped = observation_importance.reshape(1, -1)

    # (n_isotopes, n_observations)
    sum_precursor_intensity = np.sum(
        np.sum(dense_precursors[0], axis=-1), axis=-1
    ).astype(np.float32)

    # (n_isotopes)
    weighted_sum_precursor_intensity = np.sum(
        sum_precursor_intensity * observation_importance_reshaped, axis=-1
    ).astype(np.float32)

    # mono_ms1_intensity
    feature_array[4] = weighted_sum_precursor_intensity[0]

    # top_ms1_intensity
    feature_array[5] = weighted_sum_precursor_intensity[np.argmax(isotope_intensity)]

    # sum_ms1_intensity
    feature_array[6] = np.sum(weighted_sum_precursor_intensity)

    # weighted_ms1_intensity
    feature_array[7] = np.sum(weighted_sum_precursor_intensity * isotope_intensity)

    expected_scan_center = utils.tile(
        dense_precursors.shape[3], n_isotopes * n_observations
    ).reshape(n_isotopes, -1)
    expected_frame_center = utils.tile(
        dense_precursors.shape[2], n_isotopes * n_observations
    ).reshape(n_isotopes, -1)

    # (n_isotopes)
    observed_precursor_height = weighted_center_mean_2d(
        dense_precursors[0], expected_scan_center, expected_frame_center
    )[:, 0]

    # (n_isotopes)
    observed_precursor_mz = weighted_center_mean_2d(
        dense_precursors[1], expected_scan_center, expected_frame_center
    )[:, 0]

    mz_mask = observed_precursor_mz > 0

    # (n_isotopes)
    mass_error_array = (observed_precursor_mz - isotope_mz) / isotope_mz * 1e6
    weighted_mass_error = np.sum(mass_error_array[mz_mask] * isotope_intensity[mz_mask])

    # weighted_mass_deviation
    feature_array[8] = weighted_mass_error

    # weighted_mass_error
    feature_array[9] = np.abs(weighted_mass_error)

    # mz_observed
    feature_array[10] = isotope_mz[0] + weighted_mass_error * 1e-6 * isotope_mz[0]

    # mono_ms1_height
    feature_array[11] = observed_precursor_height[0]

    # top_ms1_height
    feature_array[12] = observed_precursor_height[np.argmax(isotope_intensity)]

    # sum_ms1_height
    feature_array[13] = np.sum(observed_precursor_height)

    # weighted_ms1_height
    feature_array[14] = np.sum(observed_precursor_height * isotope_intensity)

    # isotope_intensity_correlation
    feature_array[15] = numeric.save_corrcoeff(
        isotope_intensity, np.sum(sum_precursor_intensity, axis=-1)
    )

    # isotope_height_correlation
    feature_array[16] = numeric.save_corrcoeff(
        isotope_intensity, observed_precursor_height
    )


@nb.njit(cache=USE_NUMBA_CACHING)
def location_features(
    jit_data,
    scan_start,
    scan_stop,
    scan_center,
    frame_start,
    frame_stop,
    frame_center,
    feature_array,
):
    # base_width_mobility
    feature_array[0] = (
        jit_data.mobility_values[scan_start] - jit_data.mobility_values[scan_stop - 1]
    )

    # base_width_rt
    feature_array[1] = (
        jit_data.rt_values[frame_stop - 1] - jit_data.rt_values[frame_start]
    )

    # rt_observed
    feature_array[2] = jit_data.rt_values[frame_center]

    # mobility_observed
    feature_array[3] = jit_data.mobility_values[scan_center]


nb_float32_array = nb.types.Array(nb.types.float32, 1, "C")


@nb.njit(cache=USE_NUMBA_CACHING)
def fragment_features(
    dense_fragments: np.ndarray,
    fragments_frame_profile: np.ndarray,
    frame_rt: np.ndarray,
    observation_importance: np.ndarray,
    template: np.ndarray,
    fragments: np.ndarray,
    feature_array: nb_float32_array,
    quant_window: nb.uint32 = 3,
    quant_all: nb.boolean = False,
):
    n_observations = observation_importance.shape[0]
    n_fragments = dense_fragments.shape[1]
    feature_array[17] = float(n_observations)

    # (1, n_observations)
    observation_importance_reshaped = observation_importance.reshape(1, -1)

    # (n_fragments)
    fragment_intensity_norm = fragments.intensity / np.sum(fragments.intensity)

    if n_fragments == 0:
        print(n_fragments)

    # (n_observations)
    (
        expected_scan_center,
        expected_frame_center,
        expected_scan_variance,
        expected_frame_variance,
    ) = weighted_center_of_mass_1d(template)

    # expand the expected center of mass to the number of fragments
    # (n_fragments, n_observations)
    f_expected_scan_center = utils.tile(expected_scan_center, n_fragments).reshape(
        n_fragments, -1
    )
    f_expected_frame_center = utils.tile(expected_frame_center, n_fragments).reshape(
        n_fragments, -1
    )

    if quant_all:
        best_profile = np.sum(fragments_frame_profile, axis=1)

    else:
        # most intense observation across all observations
        best_observation = np.argmax(observation_importance)

        # (n_fragments, n_frames)
        best_profile = fragments_frame_profile[:, best_observation]

    center_envelope_1d(best_profile)

    # handle rare case where the best observation is at the edge of the frame
    quant_window = min((best_profile.shape[1] // 2) - 1, quant_window)

    # center the profile around the expected frame center
    center = best_profile.shape[1] // 2
    # (n_fragments, quant_window * 2 + 1)
    best_profile = best_profile[:, center - quant_window : center + quant_window + 1]

    # (quant_window * 2 + 1)
    frame_rt_quant = frame_rt[center - quant_window : center + quant_window + 1]

    # (quant_window * 2)
    delta_rt = frame_rt_quant[1:] - frame_rt_quant[:-1]

    # (n_fragments)
    fragment_area = np.sum(
        (best_profile[:, 1:] + best_profile[:, :-1]) * delta_rt.reshape(1, -1) * 0.5,
        axis=-1,
    )
    fragment_area_norm = fragment_area * quant_window

    observed_fragment_intensity = np.sum(best_profile, axis=-1)

    # create fragment masks for filtering
    fragment_profiles = np.sum(dense_fragments[0], axis=-1)
    # (n_fragments, n_observations)
    sum_fragment_intensity = np.sum(fragment_profiles, axis=-1)

    # create fragment intensity mask
    # fragment_intensity_mask_2d = sum_fragment_intensity > 0
    # fragment_intensity_weights_2d = (
    #    fragment_intensity_mask_2d * observation_importance_reshaped
    # )

    # (n_fragments, n_observations)
    # normalize rows to 1
    # fragment_intensity_weights_2d = fragment_intensity_weights_2d / (
    #    np.sum(fragment_intensity_weights_2d, axis=-1).reshape(-1, 1) + 1e-20
    # )

    # (n_fragments)
    # observed_fragment_intensity = weighted_mean_a1(
    #    sum_fragment_intensity, fragment_intensity_weights_2d
    # )

    # (n_observations)
    sum_template_intensity = np.sum(np.sum(template, axis=-1), axis=-1)

    # get the observed fragment mz and intensity
    # (n_fragments, n_observations)
    observed_fragment_mz = weighted_center_mean_2d(
        dense_fragments[1], f_expected_scan_center, f_expected_frame_center
    )

    # (n_fragments, n_observations)
    o_fragment_height = weighted_center_mean_2d(
        dense_fragments[0], f_expected_scan_center, f_expected_frame_center
    )

    # (n_fragments, n_observations)
    fragment_height_mask_2d = o_fragment_height > 0

    # (n_fragments)
    fragment_height_mask_1d = np.sum(fragment_height_mask_2d, axis=-1) > 0

    # (n_fragments, n_observations)
    fragment_height_weights_2d = (
        fragment_height_mask_2d * observation_importance_reshaped
    )

    # (n_fragments, n_observations)
    # normalize rows to 1
    fragment_height_weights_2d = fragment_height_weights_2d / (
        np.sum(fragment_height_weights_2d, axis=-1).reshape(-1, 1) + 1e-20
    )

    # (n_fragments)
    observed_fragment_mz_mean = weighted_mean_a1(
        observed_fragment_mz, fragment_height_weights_2d
    )

    # (n_fragments)
    observed_fragment_height = weighted_mean_a1(
        o_fragment_height, fragment_height_weights_2d
    )

    if np.sum(fragment_height_mask_1d) > 0.0:
        feature_array[18] = np.corrcoef(fragment_area_norm, fragment_intensity_norm)[
            0, 1
        ]

    if np.sum(observed_fragment_height) > 0.0:
        feature_array[19] = np.corrcoef(
            observed_fragment_height, fragment_intensity_norm
        )[0, 1]

    feature_array[20] = np.sum(observed_fragment_intensity > 0.0) / n_fragments
    feature_array[21] = np.sum(observed_fragment_height > 0.0) / n_fragments

    feature_array[22] = np.sum(
        fragment_intensity_norm[observed_fragment_intensity > 0.0]
    )
    feature_array[23] = np.sum(fragment_intensity_norm[observed_fragment_height > 0.0])

    fragment_mask = observed_fragment_intensity > 0

    if np.sum(fragment_mask) > 0:
        sum_template_intensity_expanded = sum_template_intensity.reshape(1, -1)
        observation_score = cosine_similarity_a1(
            sum_template_intensity_expanded, sum_fragment_intensity[fragment_mask]
        ).astype(np.float32)
        feature_array[24] = np.mean(observation_score)

    # ============= FRAGMENT TYPE FEATURES =============

    b_ion_mask = fragments.type == 98
    y_ion_mask = fragments.type == 121

    weighted_b_ion_intensity = observed_fragment_intensity[b_ion_mask]
    weighted_y_ion_intensity = observed_fragment_intensity[y_ion_mask]

    feature_array[25] = (
        np.log(np.sum(weighted_b_ion_intensity) + 1)
        if len(weighted_b_ion_intensity) > 0
        else 0.0
    )
    feature_array[26] = (
        np.log(np.sum(weighted_y_ion_intensity) + 1)
        if len(weighted_y_ion_intensity) > 0
        else 0.0
    )
    feature_array[27] = feature_array[25] - feature_array[26]

    # ============= FRAGMENT FEATURES =============

    mass_error = (observed_fragment_mz_mean - fragments.mz) / fragments.mz * 1e6

    fragment_idx_sorted = np.argsort(fragments.intensity)[::-1]
    top_3_idxs = fragment_idx_sorted[:3]

    # top_3_ms2_mass_error
    feature_array[41] = mass_error[top_3_idxs].mean()

    # mean_ms2_mass_error
    feature_array[42] = mass_error.mean()

    # ============= FRAGMENT intersection =============

    is_b = fragments.type == 98
    is_y = fragments.type == 121

    if np.sum(is_b) > 0 and np.sum(is_y) > 0:
        min_y = fragments.position[is_y].min()
        max_b = fragments.position[is_b].max()
        overlapping = (is_y & (fragments.position < max_b)) | (
            is_b & (fragments.position > min_y)
        )

        # n_overlapping
        feature_array[43] = overlapping.sum()

        if feature_array[43] > 0:
            # mean_overlapping_intensity
            feature_array[44] = np.mean(fragment_area_norm[overlapping])
            # mean_overlapping_mass_error
            feature_array[45] = np.mean(mass_error[overlapping])
        else:
            feature_array[44] = 0
            feature_array[45] = 15

    return (
        observed_fragment_mz_mean,
        mass_error,
        observed_fragment_height,
        fragment_area_norm,
    )


@nb.njit(cache=USE_NUMBA_CACHING)
def fragment_mobility_correlation(
    fragments_scan_profile,
    template_scan_profile,
    observation_importance,
    fragment_intensity,
):
    n_observations = len(observation_importance)

    fragment_mask_1d = np.sum(np.sum(fragments_scan_profile, axis=-1), axis=-1) > 0
    if np.sum(fragment_mask_1d) < 3:
        return 0, 0

    non_zero_fragment_norm = fragment_intensity[fragment_mask_1d] / np.sum(
        fragment_intensity[fragment_mask_1d]
    )

    # (n_observations, n_fragments, n_fragments)
    fragment_scan_correlation_masked = numeric.fragment_correlation(
        fragments_scan_profile[fragment_mask_1d],
    )

    # (n_fragments, n_fragments)
    fragment_scan_correlation_maked_reduced = np.sum(
        fragment_scan_correlation_masked * observation_importance.reshape(-1, 1, 1),
        axis=0,
    )
    fragment_scan_correlation_list = np.dot(
        fragment_scan_correlation_maked_reduced, non_zero_fragment_norm
    )

    # fragment_scan_correlation
    fragment_scan_correlation = np.mean(fragment_scan_correlation_list)

    # (n_observation, n_fragments)
    fragment_template_scan_correlation = numeric.fragment_correlation_different(
        fragments_scan_profile[fragment_mask_1d],
        template_scan_profile.reshape(1, n_observations, -1),
    ).reshape(n_observations, -1)

    # (n_fragments)
    fragment_template_scan_correlation_reduced = np.sum(
        fragment_template_scan_correlation * observation_importance.reshape(-1, 1),
        axis=0,
    )
    # template_scan_correlation
    template_scan_correlation = np.dot(
        fragment_template_scan_correlation_reduced, non_zero_fragment_norm
    )

    return fragment_scan_correlation, template_scan_correlation


@nb.njit(cache=USE_NUMBA_CACHING)
def profile_features(
    dia_data,
    fragment_intensity,
    fragment_type,
    observation_importance,
    fragments_scan_profile,
    fragments_frame_profile,
    template_scan_profile,
    template_frame_profile,
    scan_start,
    scan_stop,
    frame_start,
    frame_stop,
    feature_array,
    experimental_xic,
):
    n_observations = len(observation_importance)
    fragment_idx_sorted = np.argsort(fragment_intensity)[::-1]

    # ============= FRAGMENT RT CORRELATIONS =============
    top_3_idxs = fragment_idx_sorted[:3]

    if experimental_xic:
        # New correlation method
        intensity_slice = fragments_frame_profile.sum(axis=1)
        normalized_intensity_slice = normalize_profiles(intensity_slice, 1)
        median_profile = median_axis(normalized_intensity_slice, 0)
        fragment_frame_correlation_list = correlation_coefficient(
            median_profile, intensity_slice
        ).astype(np.float32)
        top3_fragment_frame_correlation = fragment_frame_correlation_list[
            top_3_idxs
        ].mean()
    else:
        # Original correlation method
        fragment_frame_correlation_masked = numeric.fragment_correlation(
            fragments_frame_profile,
        )
        fragment_frame_correlation_maked_reduced = np.sum(
            fragment_frame_correlation_masked
            * observation_importance.reshape(-1, 1, 1),
            axis=0,
        )
        fragment_frame_correlation_list = np.dot(
            fragment_frame_correlation_maked_reduced, fragment_intensity
        )

        top3_fragment_frame_correlation = fragment_frame_correlation_maked_reduced[
            top_3_idxs, :
        ][:, top_3_idxs].mean()

    feature_array[31] = np.mean(fragment_frame_correlation_list)

    # (3)
    feature_array[32] = top3_fragment_frame_correlation

    # (n_observation, n_fragments)
    fragment_template_frame_correlation = numeric.fragment_correlation_different(
        fragments_frame_profile,
        template_frame_profile.reshape(1, n_observations, -1),
    ).reshape(n_observations, -1)

    # (n_fragments)
    fragment_template_frame_correlation_reduced = np.sum(
        fragment_template_frame_correlation * observation_importance.reshape(-1, 1),
        axis=0,
    )

    # template_frame_correlation
    feature_array[33] = np.dot(
        fragment_template_frame_correlation_reduced, fragment_intensity
    )

    # ============= FRAGMENT TYPE FEATURES =============

    b_ion_mask = fragment_type == 98
    y_ion_mask = fragment_type == 121

    b_ion_index_sorted = fragment_idx_sorted[b_ion_mask]
    y_ion_index_sorted = fragment_idx_sorted[y_ion_mask]

    if len(b_ion_index_sorted) > 0:
        b_ion_limit = min(len(b_ion_index_sorted), 3)
        # 'top3_b_ion_correlation'
        feature_array[34] = fragment_frame_correlation_list[
            b_ion_index_sorted[:b_ion_limit]
        ].mean()
        feature_array[35] = float(len(b_ion_index_sorted))

    if len(y_ion_index_sorted) > 0:
        y_ion_limit = min(len(y_ion_index_sorted), 3)
        feature_array[36] = fragment_frame_correlation_list[
            y_ion_index_sorted[:y_ion_limit]
        ].mean()
        feature_array[37] = float(len(y_ion_index_sorted))

    # ============= FWHM RT =============

    # (n_fragments, n_observations)
    cycle_fwhm = np.zeros(
        (
            fragments_frame_profile.shape[0],
            fragments_frame_profile.shape[1],
        ),
        dtype=np.float32,
    )

    rt_width = dia_data.rt_values[frame_stop - 1] - dia_data.rt_values[frame_start]

    for i_fragment in range(fragments_frame_profile.shape[0]):
        for i_observation in range(fragments_frame_profile.shape[1]):
            max_intensity = np.max(fragments_frame_profile[i_fragment, i_observation])
            half_max = max_intensity / 2
            n_values_above = np.sum(
                fragments_frame_profile[i_fragment, i_observation] > half_max
            )
            fraction_above = n_values_above / len(
                fragments_frame_profile[i_fragment, i_observation]
            )

            cycle_fwhm[i_fragment, i_observation] = fraction_above * rt_width

    cycle_fwhm_mean_list = np.sum(
        cycle_fwhm * observation_importance.reshape(1, -1), axis=-1
    )
    cycle_fwhm_mean_agg = np.sum(cycle_fwhm_mean_list * fragment_intensity)

    feature_array[38] = cycle_fwhm_mean_agg

    # ============= FWHM MOBILITY =============

    # will be skipped if no mobility dimension is present
    if dia_data.has_mobility:
        # (n_fragments, n_observations)
        mobility_fwhm = np.zeros(
            (
                fragments_scan_profile.shape[0],
                fragments_scan_profile.shape[1],
            ),
            dtype=np.float32,
        )

        mobility_width = (
            dia_data.mobility_values[scan_start]
            - dia_data.mobility_values[scan_stop - 1]
        )

        for i_fragment in range(fragments_scan_profile.shape[0]):
            for i_observation in range(fragments_scan_profile.shape[1]):
                max_intensity = np.max(
                    fragments_scan_profile[i_fragment, i_observation]
                )
                half_max = max_intensity / 2
                n_values_above = np.sum(
                    fragments_scan_profile[i_fragment, i_observation] > half_max
                )
                fraction_above = n_values_above / len(
                    fragments_scan_profile[i_fragment, i_observation]
                )

                mobility_fwhm[i_fragment, i_observation] = (
                    fraction_above * mobility_width
                )

        mobility_fwhm_mean_list = np.sum(
            mobility_fwhm * observation_importance.reshape(1, -1), axis=-1
        )
        mobility_fwhm_mean_agg = np.sum(mobility_fwhm_mean_list * fragment_intensity)

        feature_array[39] = mobility_fwhm_mean_agg

    # ============= RT SHIFT =============

    # (n_fragments, n_observations)
    frame_peak = np.argmax(fragments_frame_profile, axis=2)

    # (n_observations)
    median_frame_peak = np.zeros((n_observations), dtype=np.float32)
    for i_observation in range(n_observations):
        median_frame_peak[i_observation] = np.median(frame_peak[:, i_observation])

    # (n_observations)
    delta_frame_peak = median_frame_peak - np.floor(
        fragments_frame_profile.shape[-1] / 2
    )
    feature_array[40] = np.sum(delta_frame_peak * observation_importance)

    return fragment_frame_correlation_list


@nb.njit(cache=USE_NUMBA_CACHING)
def reference_features(
    reference_observation_importance,
    reference_fragments_scan_profile,
    reference_fragments_frame_profile,
    reference_template_scan_profile,
    reference_template_frame_profile,
    observation_importance,
    fragments_scan_profile,
    fragments_frame_profile,
    template_scan_profile,
    template_frame_profile,
    fragment_lib_intensity,
):
    feature_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type, value_type=nb.types.float32
    )

    fragment_idx_sorted = np.argsort(fragment_lib_intensity)[::-1]

    if (
        reference_fragments_scan_profile.shape[0] == 0
        or fragments_scan_profile.shape[0] == 0
        or reference_fragments_scan_profile.shape[0] != fragments_scan_profile.shape[0]
    ):
        feature_dict["reference_intensity_correlation"] = 0

        feature_dict["mean_reference_scan_cosine"] = 0
        feature_dict["top3_reference_scan_cosine"] = 0
        feature_dict["mean_reference_frame_cosine"] = 0
        feature_dict["top3_reference_frame_cosine"] = 0
        feature_dict["mean_reference_template_scan_cosine"] = 0
        feature_dict["top3_reference_template_scan_cosine"] = 0
        feature_dict["mean_reference_template_frame_cosine"] = 0
        feature_dict["top3_reference_template_frame_cosine"] = 0

        return feature_dict

    # ============= Fragment Intensity =============

    reference_fragment_intensity = np.sum(
        np.sum(reference_fragments_scan_profile, axis=-1)
        * reference_observation_importance.reshape(1, -1),
        axis=-1,
    )
    fragment_intensity = np.sum(
        np.sum(fragments_scan_profile, axis=-1) * observation_importance.reshape(1, -1),
        axis=-1,
    )

    total_fragment_intensity = np.sum(fragment_intensity)

    reference_intensity_correlation = 0

    if total_fragment_intensity > 1 and np.sum(reference_fragment_intensity) > 1:
        # print('reference_fragment_intensity',reference_fragment_intensity, reference_fragment_intensity.shape)
        # print('fragment_intensity',fragment_intensity, fragment_intensity.shape)
        reference_intensity_correlation = np.corrcoef(
            reference_fragment_intensity, fragment_intensity
        )[0, 1]

    feature_dict["reference_intensity_correlation"] = reference_intensity_correlation

    # ============= Fragment Profile =============

    reference_scan_profile = np.sum(
        reference_fragments_scan_profile
        * reference_observation_importance.reshape(1, -1, 1),
        axis=1,
    )
    scan_profile = np.sum(
        fragments_scan_profile * observation_importance.reshape(1, -1, 1), axis=1
    )

    scan_similarity = cosine_similarity_a1(reference_scan_profile, scan_profile)

    feature_dict["mean_reference_scan_cosine"] = np.mean(scan_similarity)
    feature_dict["top3_reference_scan_cosine"] = scan_similarity[
        fragment_idx_sorted[:3]
    ].mean()

    reference_frame_profile = np.sum(
        reference_fragments_frame_profile
        * reference_observation_importance.reshape(1, -1, 1),
        axis=1,
    )
    frame_profile = np.sum(
        fragments_frame_profile * observation_importance.reshape(1, -1, 1), axis=1
    )

    frame_similarity = cosine_similarity_a1(reference_frame_profile, frame_profile)

    feature_dict["mean_reference_frame_cosine"] = np.mean(frame_similarity)
    feature_dict["top3_reference_frame_cosine"] = frame_similarity[
        fragment_idx_sorted[:3]
    ].mean()

    # ============= Template Profile =============

    reference_template_scan_profile = np.sum(
        reference_template_scan_profile
        * reference_observation_importance.reshape(1, -1, 1),
        axis=1,
    )
    template_scan_profile = np.sum(
        template_scan_profile * observation_importance.reshape(1, -1, 1), axis=1
    )

    scan_similarity = cosine_similarity_a1(
        reference_template_scan_profile, template_scan_profile
    )

    feature_dict["mean_reference_template_scan_cosine"] = np.mean(scan_similarity)

    reference_template_frame_profile = np.sum(
        reference_template_frame_profile
        * reference_observation_importance.reshape(1, -1, 1),
        axis=1,
    )
    template_frame_profile = np.sum(
        template_frame_profile * observation_importance.reshape(1, -1, 1), axis=1
    )

    frame_similarity = cosine_similarity_a1(
        reference_template_frame_profile, template_frame_profile
    )

    feature_dict["mean_reference_template_frame_cosine"] = np.mean(frame_similarity)

    return feature_dict

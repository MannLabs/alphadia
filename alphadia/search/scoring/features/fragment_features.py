"""Feature extraction for fragment ions."""

import numba as nb
import numpy as np

from alphadia.search.scoring.features.features_utils import (
    cosine_similarity_a1,
    weighted_center_mean_2d,
)
from alphadia.search.scoring.utils import (
    fragment_correlation,
    fragment_correlation_different,
    tile,
)
from alphadia.utils import USE_NUMBA_CACHING

nb_float32_array = nb.types.Array(nb.types.float32, 1, "C")


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
    f_expected_scan_center = tile(expected_scan_center, n_fragments).reshape(
        n_fragments, -1
    )
    f_expected_frame_center = tile(expected_frame_center, n_fragments).reshape(
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
    fragment_scan_correlation_masked = fragment_correlation(
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
    fragment_template_scan_correlation = fragment_correlation_different(
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

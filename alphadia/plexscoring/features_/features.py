import numba as nb
import numpy as np

from alphadia import utils
from alphadia.numba import numeric
from alphadia.plexscoring.features_.features_utils import (
    cosine_similarity_a1,
    weighted_center_mean_2d,
)
from alphadia.scoring.utils import (
    correlation_coefficient,
    median_axis,
    normalize_profiles,
)
from alphadia.utils import USE_NUMBA_CACHING

float_array = nb.types.float32[:]  # TODO duplicated in plexscoring


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

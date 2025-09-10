"""Profile-based features for elution and mobility patterns."""

import numba as nb
import numpy as np

from alphadia.search.scoring.scoring_utils import (
    correlation_coefficient,
    median_axis,
    normalize_profiles,
)
from alphadia.search.scoring.utils import (
    fragment_correlation,
    fragment_correlation_different,
)
from alphadia.utils import USE_NUMBA_CACHING


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
        fragment_frame_correlation_masked = fragment_correlation(
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
    fragment_template_frame_correlation = fragment_correlation_different(
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

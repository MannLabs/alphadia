"""Feature extraction for precursor ions."""

import numba as nb
import numpy as np

from alphadia.search.scoring.features.features_utils import (
    weighted_center_mean_2d,
)
from alphadia.search.scoring.utils import save_corrcoeff, tile
from alphadia.utils import USE_NUMBA_CACHING


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

    expected_scan_center = tile(
        dense_precursors.shape[3], n_isotopes * n_observations
    ).reshape(n_isotopes, -1)
    expected_frame_center = tile(
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
    feature_array[15] = save_corrcoeff(
        isotope_intensity, np.sum(sum_precursor_intensity, axis=-1)
    )

    # isotope_height_correlation
    feature_array[16] = save_corrcoeff(isotope_intensity, observed_precursor_height)

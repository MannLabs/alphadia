"""Reference-based features comparing against library spectra."""

import numba as nb
import numpy as np

from alphadia.search.scoring.features.features_utils import (
    cosine_similarity_a1,
)
from alphadia.utils import USE_NUMBA_CACHING


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

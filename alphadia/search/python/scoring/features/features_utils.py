"""Utility functions for feature calculations."""

import numba as nb
import numpy as np

from alphadia.utils import USE_NUMBA_CACHING


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


@nb.njit(cache=USE_NUMBA_CACHING)
def cosine_similarity_a1(template_intensity, fragments_intensity):
    fragment_norm = np.sqrt(np.sum(np.power(fragments_intensity, 2), axis=-1))
    template_norm = np.sqrt(np.sum(np.power(template_intensity, 2), axis=-1))

    div = (fragment_norm * template_norm) + 0.0001

    return np.sum(fragments_intensity * template_intensity, axis=-1) / div

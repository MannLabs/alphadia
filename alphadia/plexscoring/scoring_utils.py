"""Utility functions for scoring calculations in AlphaDIA.

This module provides numba-accelerated utility functions for various scoring calculations,
including correlation coefficients, profile normalization, and statistical operations.
"""

import numba as nb
import numpy as np
from numba import types

from alphadia.utils import USE_NUMBA_CACHING


@nb.njit(
    types.Array(types.float32, 1, "A")(
        types.Array(types.float32, 1, "A"), types.Array(types.float32, 2, "A")
    ),
    cache=USE_NUMBA_CACHING,
)
def correlation_coefficient(x: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Calculate the correlation coefficient between x and each y in ys.

    Returns a numpy array of the same length as ys.

    Parameters
    ----------
    x : np.ndarray[float32, ndim=1]
        Base array of shape (n,)
    ys : np.ndarray[float32, ndim=2]
        Array of shape (m, n) containing arrays to correlate with x

    Returns
    -------
    np.ndarray[float32, ndim=1]
        Array of shape (m,) containing correlation coefficients.
        Returns 0 for cases where either x or y has zero variance.

    """
    n = len(x)
    # Calculate means
    mx = x.mean()
    # Calculate mean for each y array manually since axis parameter isn't supported
    m = len(ys)
    my = np.zeros(m, dtype=np.float32)
    for i in range(m):
        my[i] = np.sum(ys[i]) / n

    # Initialize array for results
    result = np.zeros(m, dtype=np.float32)

    x_minus_mx = x - mx

    var_x = np.sum(x_minus_mx * x_minus_mx) / n

    # Calculate correlation coefficient for each y in ys
    for i in range(m):
        # Calculate covariance and variances
        ys_minus_my = ys[i] - my[i]
        cov = np.sum(x_minus_mx * ys_minus_my) / n
        var_y = np.sum(ys_minus_my * ys_minus_my) / n

        var_xy = var_x * var_y

        # Handle zero variance cases
        if var_xy == 0:
            result[i] = 0
        else:
            result[i] = cov / np.sqrt(var_xy)

    return result


@nb.njit(
    types.Array(types.float32, 2, "A")(
        types.Array(types.float32, 2, "A"), types.Optional(types.int64)
    ),
    cache=USE_NUMBA_CACHING,
)
def normalize_profiles(
    intensity_slice: np.ndarray, center_dilations: int = 1
) -> np.ndarray:
    """Calculate normalized intensity profiles from dense array.

    Parameters
    ----------
    intensity_slice : np.ndarray[float32, ndim=2]
        Array where first dimension represents different measurements,
        and subsequent dimensions represent mz and rt
    center_dilations : int, optional
        Number of points to consider around center for normalization.
        Default is 1.

    Returns
    -------
    np.ndarray[float32, ndim=2]
        Array of normalized intensity profiles with same shape as input,
        where profiles with zero center intensity are set to zero

    """
    center_idx = intensity_slice.shape[1] // 2

    # Calculate mean manually instead of using axis parameter
    center_intensity = np.ones((intensity_slice.shape[0], 1))

    for i in range(intensity_slice.shape[0]):
        window = intensity_slice[
            i, center_idx - center_dilations : center_idx + center_dilations + 1
        ]
        center_intensity[i, 0] = np.sum(window) / window.shape[0]

    # Create normalized output array, initialized to zeros
    center_intensity_normalized = np.zeros_like(intensity_slice, dtype=np.float32)

    # Only normalize profiles where center intensity > 0
    for i in range(intensity_slice.shape[0]):
        if center_intensity[i, 0] > 0:
            center_intensity_normalized[i] = intensity_slice[i] / center_intensity[i, 0]
    return center_intensity_normalized


@nb.njit(
    types.Array(types.float32, 1, "A")(
        types.Array(types.float32, 2, "A"), types.Optional(types.int64)
    ),
    cache=USE_NUMBA_CACHING,
)
def median_axis(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Calculate the median along a specified axis.

    Parameters
    ----------
    array : np.ndarray[float32, ndim=2]
        Input array
    axis : int, optional
        Axis along which to calculate median. Default is 0.

    Returns
    -------
    np.ndarray[float32, ndim=1]
        Array of medians along the specified axis

    """
    if axis == 0:
        result = np.zeros(array.shape[1], dtype=np.float32)
        for i in range(array.shape[1]):
            result[i] = np.median(array[:, i])
    else:  # axis == 1
        result = np.zeros(array.shape[0], dtype=np.float32)
        for i in range(array.shape[0]):
            result[i] = np.median(array[i, :])

    return result

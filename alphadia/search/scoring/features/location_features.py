"""Location-based features for candidate scoring."""

import numba as nb

from alphadia.utils import USE_NUMBA_CACHING


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

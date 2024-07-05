# native imports

# alphadia imports

# alpha family imports

# third party imports
import numba as nb
import numpy as np


@nb.njit(parallel=False, fastmath=True)
def search_sorted_left(slice, value):
    left = 0
    right = len(slice)

    while left < right:
        mid = (left + right) >> 1
        if slice[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@nb.njit
def ceil_to_base_two(x):
    # borrowed from Bit Twiddling Hacks
    # https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16

    x += 1
    return x


@nb.njit
def wrap0(
    value,
    limit,
):
    if value < 0:
        return 0
    else:
        return min(value, limit)


@nb.njit
def wrap1(
    values,
    limit,
):
    for i in range(values.shape[0]):
        values[i] = wrap0(values[i], limit)
    return values


@nb.njit
def get_mean0(dense, scan, cycle):
    """create a fixed window around the peak and extract the mean value"""
    # window size around peak
    w = 4

    # extract mz
    mz_window = dense[
        max(scan - w, 0) : scan + w, max(cycle - w, 0) : cycle + w
    ].flatten()

    return np.mean(mz_window)


@nb.njit
def get_mean_sparse0(dense, scan, cycle, threshold):
    """create a fixed window around the peak and extract the mean value"""
    # window size around peak
    w = 4

    # extract mz
    mz_window = dense[
        max(scan - w, 0) : scan + w, max(cycle - w, 0) : cycle + w
    ].flatten()

    mask = mz_window < threshold
    fraction_nonzero = np.mean(mask.astype("int8"))

    values = np.mean(mz_window[mask]) if fraction_nonzero > 0 else threshold

    return values


@nb.njit
def symetric_limits_1d(
    array_1d,
    center,
    f=0.95,
    center_fraction=0.01,
    min_size=1,
    max_size=10,
):
    """
    Find the limits of a symetric peak in a 1D array.
    Allthough the edge is symetrically extended in both directions, the trailing edge will use the last valid value when it reaches the limits of the array.

    Parameters
    ----------

    array_1d : np.ndarray
        1D array of intensities

    center : int
        Index of the center of the peak

    f : float
        minimal required decrease in intensity relative to the trailing intensity

    center_fraction : float
        minimal required intensity relative to the center intensity

    min_size : int
        minimal distance of the trailing edge of the peak from the center

    max_size : int
        maximal distance of the trailing edge of the peak from the center

    Returns
    -------

    np.ndarray, dtype='int32', shape=(2,)
        Array of containing the start and stop index of the peak.
        The start index is inclusive, the stop index is exclusive.
        Providing an empty array will return np.array([center, center], dtype='int32').
        Providing a center outside the array will return np.array([center, center], dtype='int32').

    """

    if len(array_1d) == 0:
        return np.array([center, center], dtype="int32")

    if center < 0 or center >= array_1d.shape[0]:
        return np.array([center, center], dtype="int32")

    center_intensity = array_1d[center]
    trailing_intensity = center_intensity

    limit = min_size

    for s in range(min_size + 1, max_size):
        intensity = (
            array_1d[max(center - s, 0)] + array_1d[min(center + s, len(array_1d) - 1)]
        ) / 2
        if intensity < f * trailing_intensity:
            if intensity > center_intensity * center_fraction:
                limit = s
                trailing_intensity = intensity
            else:
                break

        else:
            break

    return np.array(
        [max(center - limit, 0), min(center + limit + 1, len(array_1d))], dtype="int32"
    )


@nb.njit
def symetric_limits_2d(
    a,
    scan_center,
    dia_cycle_center,
    f_mobility=0.95,
    f_rt=0.95,
    center_fraction=0.01,
    min_size_mobility=3,
    max_size_mobility=20,
    min_size_rt=1,
    max_size_rt=10,
):
    mobility_lower = max(0, scan_center - min_size_mobility)
    mobility_upper = min(a.shape[0], scan_center + min_size_mobility)
    dia_cycle_lower = max(0, dia_cycle_center - min_size_rt)
    dia_cycle_upper = min(a.shape[1], dia_cycle_center + min_size_rt)

    mobility_limits = symetric_limits_1d(
        a[:, dia_cycle_lower:dia_cycle_upper].sum(axis=1),
        scan_center,
        f=f_mobility,
        center_fraction=center_fraction,
        min_size=min_size_mobility,
        max_size=max_size_mobility,
    )

    dia_cycle_limits = symetric_limits_1d(
        a[mobility_lower:mobility_upper, :].sum(axis=0),
        dia_cycle_center,
        f=f_rt,
        center_fraction=center_fraction,
        min_size=min_size_rt,
        max_size=max_size_rt,
    )

    return mobility_limits, dia_cycle_limits


@nb.njit(inline="always")
def save_corrcoeff(x: np.array, y: np.array):
    """Save way to calculate the correlation coefficient between two one-dimensional arrays.

    Parameters
    ----------

    x : np.array
        One-dimensional array of shape (n,)

    y : np.array
        One-dimensional array of shape (n,)

    Returns
    -------
    float
        Correlation coefficient between x and y

    """
    assert len(x) > 0
    assert x.ndim == 1
    assert x.shape == y.shape

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    x_centered = x - x_bar
    y_centered = y - y_bar

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

    return numerator / (denominator + 1e-12)


@nb.njit()
def fragment_correlation(
    fragments_profile,
):
    """Calculates a save correlation matrix for a given fragment profile.

    Parameters
    ----------

    fragments_profile: np.ndarray
        array of shape (n_fragments, n_observations, n_data_points)

    Returns
    -------

    np.ndarray
        array of shape (n_observations, n_fragments, n_fragments)

    """

    assert fragments_profile.ndim == 3

    n_fragments = fragments_profile.shape[0]
    n_observations = fragments_profile.shape[1]
    n_data_points = fragments_profile.shape[2]
    assert n_data_points > 0

    # (n_observations, n_fragments, n_fragments)
    output = np.zeros((n_observations, n_fragments, n_fragments), dtype="float32")
    if n_data_points == 0:
        return output

    for i_observations in range(n_observations):
        # (n_fragments, 1)
        profile_mean = np.reshape(
            np.sum(fragments_profile[:, i_observations], axis=1) / n_data_points,
            (n_fragments, 1),
        )

        # (n_fragments, n_data_points)
        profile_centered = fragments_profile[:, i_observations] - profile_mean

        # (n_fragments, 1)
        profile_std = np.reshape(
            np.sqrt(np.sum(profile_centered**2, axis=1) / n_data_points),
            (n_fragments, 1),
        )

        # (n_fragments, n_fragments)
        covariance_matrix = np.dot(profile_centered, profile_centered.T) / n_data_points

        # (n_fragments, n_fragments)
        std_matrix = np.dot(profile_std, profile_std.T)

        # (n_fragments, n_fragments)
        correlation_matrix = covariance_matrix / (std_matrix + 1e-12)
        output[i_observations] = correlation_matrix

    return output


@nb.njit()
def fragment_correlation_different(x: np.ndarray, y: np.ndarray):
    """Calculates a save correlation matrix for a given fragment profile.

    Parameters
    ----------

    x : np.ndarray
        array of shape (n_fragments, n_observations, n_data_points)

    y : np.ndarray
        array of shape (n_fragments, n_observations, n_data_points)

    Returns
    -------

    output : np.ndarray
        array of shape (n_observations, n_fragments_x, n_fragments_y)

    """

    assert x.ndim == 3
    assert y.ndim == 3
    assert x.shape[1:] == y.shape[1:]

    n_fragments_x = x.shape[0]
    n_fragments_y = y.shape[0]
    n_observations = x.shape[1]
    n_data_points = x.shape[2]
    assert n_data_points > 0

    # (n_observations, n_fragments_x, n_fragments_y)
    output = np.zeros((n_observations, n_fragments_x, n_fragments_y), dtype=np.float32)
    if n_data_points == 0:
        return output

    for i_observations in range(n_observations):
        # (n_fragments_x, 1)
        x_mean = np.reshape(
            np.sum(x[:, i_observations], axis=1) / n_data_points, (n_fragments_x, 1)
        )

        # (n_fragments_y, 1)
        y_mean = np.reshape(
            np.sum(y[:, i_observations], axis=1) / n_data_points, (n_fragments_y, 1)
        )

        # (n_fragments_x, n_data_points)
        x_centered = x[:, i_observations] - x_mean

        # (n_fragments_y, n_data_points)
        y_centered = y[:, i_observations] - y_mean

        # (n_fragments_x, 1)
        x_std = np.reshape(
            np.sqrt(np.sum(x_centered**2, axis=1) / n_data_points), (n_fragments_x, 1)
        )

        # (n_fragments_y, 1)
        y_std = np.reshape(
            np.sqrt(np.sum(y_centered**2, axis=1) / n_data_points), (n_fragments_y, 1)
        )

        # (n_fragments_x, n_fragments_y)
        covariance_matrix = np.dot(x_centered, y_centered.T) / n_data_points

        # (n_fragments_x, n_fragments_y)
        std_matrix = np.dot(x_std, y_std.T)

        # (n_fragments_x, n_fragments_y)
        correlation_matrix = covariance_matrix / (std_matrix + 1e-12)
        output[i_observations] = correlation_matrix

    return output


@nb.njit(inline="always")
def amean(array, axis):
    return np.sum(array, axis=axis) / array.shape[axis]

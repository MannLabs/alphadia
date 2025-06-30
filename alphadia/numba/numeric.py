# native imports

# alphadia imports

# alpha family imports

# third party imports
import numba as nb
import numpy as np

from alphadia.utils import USE_NUMBA_CACHING


@nb.njit(inline="always", cache=USE_NUMBA_CACHING)
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


@nb.njit(cache=USE_NUMBA_CACHING)
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


@nb.njit(cache=USE_NUMBA_CACHING)
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

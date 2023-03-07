import numpy as np
import numba as nb

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

    x +=1
    return x

@nb.njit
def roll(a, delta0, delta1):

    b = np.zeros_like(a)
    b[delta0:, delta1:] = a[:-delta0, :-delta1]
    b[:delta0, delta1:] = a[-delta0:, :-delta1]
    b[delta0:, :delta1] = a[:-delta0, -delta1:]
    b[:delta0, :delta1] = a[-delta0:, -delta1:]

    return b

@nb.njit
def fourier_a0(dense, kernel):
    """
    Numba helper function to apply a gaussian filter to a dense stack.

    Parameters
    ----------

    dense : np.ndarray
        Array of shape (n_tofs, n_scans, n_frames)

    kernel : np.ndarray
        Array of shape (n_scans, n_frames)

    Returns
    -------

    np.ndarray
        Array of shape (n_tofs, n_scans, n_frames) containing the filtered dense stack.

    """

    k0, k1 = kernel.shape

    out = np.zeros_like(dense)

    fourier_filter = np.fft.rfft2(kernel, dense.shape[1:])
    for i in range(dense.shape[0]):
        out[i] = roll(np.fft.irfft2(np.fft.rfft2(dense[i]) * fourier_filter), -k0//2, -k1//2)

    return out

@nb.njit
def fourier_a1(dense, kernel):
    """
    Numba helper function to apply a gaussian filter to a dense stack.

    Parameters
    ----------

    dense : np.ndarray
        Array of shape (n_tofs, n_observations, n_scans, n_frames)

    kernel : np.ndarray
        Array of shape (n_scans, n_frames)

    Returns
    -------

    np.ndarray
        Array of shape (n_tofs, n_observations, n_scans, n_frames) containing the filtered dense stack.

    """

    k0, k1 = kernel.shape

    out = np.zeros_like(dense, dtype='float32')

    fourier_filter = np.fft.rfft2(kernel, dense.shape[2:])
    for i in range(dense.shape[0]):
        for j in range(dense.shape[1]):
            out[i, j] = roll(np.fft.irfft2(np.fft.rfft2(dense[i, j]) * fourier_filter), -k0//2, -k1//2)

    return out
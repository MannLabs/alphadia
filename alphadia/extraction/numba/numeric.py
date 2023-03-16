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

@nb.njit
def transpose(
        tof_indices,
        push_indptr,
        values
):
    
    """
    The default alphatims data format consists of a sparse matrix where pushes are the rows, tof indices (discrete mz values) the columns and intensities the values.
    A lookup starts with a given push index p which points to the row. The start and stop indices of the row are accessed from dia_data.push_indptr[p] and dia_data.push_indptr[p+1].
    The tof indices are then accessed from dia_data.tof_indices[start:stop] and the corresponding intensities from dia_data.intensity_values[start:stop].

    The function transposes the data such that the tof indices are the rows and the pushes are the columns. 
    This is usefull when accessing only a small number of tof indices (e.g. when extracting a single precursor) and the number of pushes is large (e.g. when extracting a whole run).

    Parameters
    ----------

    tof_indices : np.ndarray
        column indices (n_values)

    push_indptr : np.ndarray
        start stop values for each row (n_rows +1)

    values : np.ndarray
        values (n_values)

    Returns
    -------

    push_indices : np.ndarray
        row indices (n_values)

    tof_indptr : np.ndarray
        start stop values for each row (n_rows +1)

    new_values : np.ndarray
        values (n_values)

    """
    # this is one less than the old col count or the new row count
    max_tof_index = tof_indices.max()
    
    tof_indcount = np.zeros((max_tof_index+1), dtype=np.int64)

    # get new column counts
    for v in tof_indices:
        tof_indcount[v] += 1

    # get new indptr
    tof_indptr = np.zeros((max_tof_index + 1 + 1), dtype=np.int64)
    for i in range(max_tof_index+1):
        tof_indptr[i + 1] = tof_indptr[i] + tof_indcount[i]

    tof_indcount = np.zeros((max_tof_index+1), dtype=np.uint32)

    # get new values
    push_indices = np.zeros((len(tof_indices)), dtype=np.uint32)

    new_values = np.zeros_like(values)

    for push_idx in range(len(push_indptr)-1):
        start_push_indptr = push_indptr[push_idx]
        stop_push_indptr = push_indptr[push_idx + 1]

        for idx in range(start_push_indptr, stop_push_indptr):
 
            # new row
            tof_index = tof_indices[idx]

            push_indices[tof_indptr[tof_index] + tof_indcount[tof_index]] = push_idx
            new_values[tof_indptr[tof_index] + tof_indcount[tof_index]] = values[idx]
            tof_indcount[tof_index] += 1

    return push_indices, tof_indptr, new_values
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
def convolve_fourier(dense, kernel):
    """
    Numba helper function to apply a gaussian filter to a dense stack.

    Parameters
    ----------

    dense : np.ndarray
        Array of shape (n_scans, n_frames)

    kernel : np.ndarray
        Array of shape (n_frames)

    Returns
    -------

    np.ndarray
        Array of shape (n_tofs, n_scans, n_frames) containing the filtered dense stack.

    """

    k0, k1 = kernel.shape

    out = np.zeros_like(dense)

    fourier_filter = np.fft.rfft2(kernel, dense.shape)
    
    out = roll(np.fft.irfft2(np.fft.rfft2(dense) * fourier_filter), -k0//2, -k1//2)

    return out

@nb.njit
def convolve_fourier_a0(dense, kernel):
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

#@nb.njit
def convolve_fourier_a1_pyfunc(dense, kernel):
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
def convolve_fourier_a1(dense, kernel):
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
    """ create a fixed window around the peak and extract the mean value
    """
    # window size around peak
    w = 4

    # extract mz
    mz_window = dense[
        max(scan-w,0):scan+w,
        max(cycle-w,0):cycle+w
    ].flatten()

    return np.mean(mz_window)

@nb.njit
def get_mean_sparse0(dense, scan, cycle, threshold):
    """ create a fixed window around the peak and extract the mean value
    """
    # window size around peak
    w = 4

    # extract mz
    mz_window = dense[
        max(scan-w,0):scan+w,
        max(cycle-w,0):cycle+w
    ].flatten()

    mask = (mz_window < threshold)
    fraction_nonzero = np.mean(mask.astype('int8'))

    if fraction_nonzero > 0:
        values = np.mean(mz_window[mask])
    else:
        values = threshold

    return values


@nb.njit
def symetric_limits_1d(
        array_1d, 
        center, 
        f = 0.95,
        center_fraction = 0.01,
        min_size = 1, 
        max_size = 10,
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
        return np.array([center, center], dtype='int32')

    if center < 0 or center >= array_1d.shape[0]:
        return np.array([center, center], dtype='int32')

    center_intensity = array_1d[center]
    trailing_intensity = center_intensity

    limit = min_size

    for s in range(min_size+1, max_size):
        intensity = (array_1d[max(center-s,0)]+array_1d[min(center+s, len(array_1d)-1)])/2
        if intensity < f * trailing_intensity:
            if intensity > center_intensity * center_fraction:
                limit = s
                trailing_intensity = intensity
            else:
                break

        else: 
            break

    return np.array([max(center-limit, 0), min(center+limit+1, len(array_1d))], dtype='int32')

@nb.njit
def symetric_limits_2d(
        a, 
        scan_center, 
        dia_cycle_center,
        f_mobility = 0.95,
        f_rt = 0.95,
        center_fraction = 0.01,
        min_size_mobility = 3,
        max_size_mobility = 20,
        min_size_rt = 1,
        max_size_rt = 10
    ):

    mobility_lower = max(0, scan_center - min_size_mobility)
    mobility_upper = min(a.shape[0], scan_center + min_size_mobility)
    dia_cycle_lower = max(0, dia_cycle_center - min_size_rt)
    dia_cycle_upper = min(a.shape[1], dia_cycle_center + min_size_rt)

    mobility_limits = symetric_limits_1d(
        a[:,dia_cycle_lower:dia_cycle_upper].sum(axis=1),
        scan_center,
        f = f_mobility,
        center_fraction = center_fraction,
        min_size = min_size_mobility,
        max_size = max_size_mobility,

    )

    dia_cycle_limits = symetric_limits_1d(
        a[mobility_lower:mobility_upper,:].sum(axis=0),
        dia_cycle_center,
        f = f_rt,
        center_fraction = center_fraction,
        min_size = min_size_rt,
        max_size = max_size_rt
    )

    return mobility_limits, dia_cycle_limits
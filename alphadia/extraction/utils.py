# internal imports
import alphatims.bruker
import alphatims.utils
import alphabase.spectral_library.library_base

# external imports
import logging

from typing import Tuple, Union, List

import pandas as pd
import numpy as np

def rt_to_frame_index(limits: Tuple, dia_data: alphatims.bruker.TimsTOF):
    """converts retention time limits to frame limits while including full precursor cycles"""
    
    # the indices are defined in a way that all precursor cycles which have at least a frame within the limits are included

    # FUTURE, sloooooooow.... numba this
    # FUTURE, calculate precursor cycle ones
    cycle_length = dia_data.cycle.shape[1]

    frames = dia_data.frames.copy()
    frames['PrecursorCycle'] = (frames['Id']-dia_data.zeroth_frame)//cycle_length

    lowest_precursor_cycle = frames[frames['Time'] >= limits[0]]['PrecursorCycle'].min()
    lowest_frame_index = frames[frames['PrecursorCycle'] == lowest_precursor_cycle]['Id'].min()


    highest_precursor_cycle = frames[frames['Time'] <= limits[1]]['PrecursorCycle'].max()
    next_highest_precursor_cycle = highest_precursor_cycle + 1
    next_highest_frame_index = frames[frames['PrecursorCycle'] == next_highest_precursor_cycle]['Id'].min()

    return (int(lowest_frame_index), int(next_highest_frame_index))



def im_to_scan_index(limits: Tuple, dia_data: alphatims.bruker.TimsTOF):
    # the indices are defined in a way that all scan within the limits are included, second index is exclusive
    lowest_scan_index = np.argmin(dia_data.mobility_values >= limits[1])
    next_highest_scan_index = np.argmin(dia_data.mobility_values > limits[0])    

    return (int(lowest_scan_index), int(next_highest_scan_index))

def calculate_mass_slice(mz, ppm):
    diff = mz / (10**6) * ppm
    return mz-diff, mz+diff

def estimate_elution_limits(dia_data: alphatims.bruker.TimsTOF, 
                            t_start: float = 2*60, 
                            t_end: float = 2*60, 
                            sigma: float = 2):
    """Helper function to determine elution start and end for rt calibration.

    the first t seconds from the start and end of the TIC trace are considered. The elution start and end are defined as the first & last timepoint with more than x sigma intensity than the start & end.
    
    Parameters
    ----------
    dia_data : alphatims.bruker.TimsTOF
        An alphatims.bruker.TimsTOF data object.
    t_start : float
        seconds after the experiment start to consider for elution start calculation.
    t_end : float
        seconds before the experiment end to consider for elution end calculation.
    sigma : float
        multiplication of standard deviation to consider the change to be significant

    Returns
    -------
    float, float : elution start and end in seconds
    """

    data = dia_data.frames.query('MsMsType == 0')[[
        'Time', 'SummedIntensities']
    ]

    # Estimate elution start
    initial_data = data[data['Time'] <= t_start]
    initial_data_mean = initial_data['SummedIntensities'].mean()
    initial_data_std = initial_data['SummedIntensities'].std()

    treshold = initial_data_mean + sigma * initial_data_std
    elution_start = data[data['SummedIntensities'] >= treshold]['Time'].min()


    # Estimate elution end
    max_rt = data['Time'].max()
    final_data = data[data['Time'] > max_rt - t_end]
    final_data_mean = final_data['SummedIntensities'].mean()
    final_data_std = final_data['SummedIntensities'].std()

    treshold = final_data_mean + sigma * final_data_std
    elution_end = data[data['SummedIntensities'] >= treshold]['Time'].max()

    return elution_start, elution_end

@alphatims.utils.njit(nogil=True)
def indices_to_slices(elution_group_idxes):
    """Converts a monotonous array of indices to slices.

    """

    current_eg = elution_group_idxes[0]
    if current_eg != 0:
        raise ValueError('expecting sorted indices going continously from 0 to N')

    max_precursor = len(elution_group_idxes)
    max_eg = np.max(elution_group_idxes)

    precursor_start_end = np.zeros((max_eg+1,2), dtype=np.uint64)

    for i in range(max_precursor):
        eg = elution_group_idxes[i]

        precursor_start_end[eg,1] = i+1

        # precursor has the same elution group
        if current_eg == eg:
            pass
        else:
            precursor_start_end[eg,0] = i

        current_eg = eg
    return precursor_start_end

@alphatims.utils.njit(nogil=True)
def join_left(
    left: np.ndarray, 
    right: np.ndarray
    ):
    """joins all values in the left array to the values in the right array. 
    The index to the element in the right array is returned. 
    If the value wasn't found, -1 is returned. If the element appears more than once, the last appearance is used.

    Parameters
    ----------

    left: numpy.ndarray
        left array which should be matched

    right: numpy.ndarray
        right array which should be matched to

    Returns
    -------
    numpy.ndarray, dtype = int64
        array with length of the left array which indices pointing to the right array
        -1 is returned if values could not be found in the right array
    """
    left_indices = np.argsort(left)
    left_sorted = left[left_indices]

    right_indices = np.argsort(right)
    right_sorted = right[right_indices]

    joined_index = -np.ones(len(left), dtype='int64')
    
    # from hereon sorted arrays are expected
    lower_right = 0

    for i in range(len(joined_index)):

        for k in range(lower_right, len(right)):

            if left_sorted[i] >= right_sorted[k]:
                if left_sorted[i] == right_sorted[k]:
                    joined_index[i] = k
                    lower_right = k
            else:
                break

    # the joined_index_sorted connects indices from the sorted left array with the sorted right array
    # to get the original indices, the order of both sides needs to be restored
    # First, the indices pointing to the right side are restored by masking the array for hits and looking up the right side
    joined_index[joined_index >= 0] = right_indices[joined_index[joined_index >= 0]]

    # Next, the left side is restored by arranging the items
    joined_index[left_indices] =  joined_index

    return joined_index

def test_join_left():

    left = np.random.randint(0,10,20)
    right = np.arange(0,10)
    joined = join_left(left, right)

    assert all(left==joined)

test_join_left()

def reannotate_fragments(
    speclib: alphabase.spectral_library.library_base.SpecLibBase, 
    fragment_speclib: alphabase.spectral_library.library_base.SpecLibBase,
    verbose = True
):
    """Reannotate an alphabase SpecLibBase library with framents from a different SpecLibBase library

    Parameters
    ----------
    speclib: alphabase.spectral_library.library_base.SpecLibBase
        Spectral library which contains the precursors to be annotated. All fragments mz and fragment intensities will be removed.

    fragment_speclib: alphabase.spectral_library.library_base.SpecLibBase
        Spectral library which contains the donor precursors whose fragments should be used.

    Returns
    -------

    alphabase.spectral_library.library_base.SpecLibBase
        newly annotated spectral library
 
    """
    if verbose:
        num_precursor_left = len(speclib.precursor_df)
        num_precursor_right = len(fragment_speclib.precursor_df)
        num_fragments_right = len(fragment_speclib.fragment_mz_df) * len(fragment_speclib.fragment_mz_df.columns)
        logging.info(f'Speclib with {num_precursor_left:,} precursors will be reannotated with speclib with {num_precursor_right:,} precursors and {num_fragments_right:,} fragments')

    # reannotation is based on mod_seq_hash column
    hash_column_name = 'mod_seq_hash'

    # create hash columns if missing
    if hash_column_name not in speclib.precursor_df.columns:
        speclib.hash_precursor_df()

    if fragment_speclib not in fragment_speclib.precursor_df.columns:
        fragment_speclib.hash_precursor_df()

    speclib_hash = speclib.precursor_df[hash_column_name].values
    fragment_speclib_hash = fragment_speclib.precursor_df[hash_column_name].values

    speclib_indices = join_left(speclib_hash, fragment_speclib_hash)

    matched_mask = (speclib_indices >= 0)

    if verbose:
        matched_count = np.sum(matched_mask)
        not_matched_count = np.sum(~matched_mask)
    
        logging.info(f'A total of {matched_count:,} precursors were succesfully annotated, {not_matched_count:,} precursors were not matched')


    frag_start_idx = fragment_speclib.precursor_df['frag_start_idx'].values[speclib_indices]
    frag_end_idx = fragment_speclib.precursor_df['frag_end_idx'].values[speclib_indices]
    
    speclib._precursor_df = speclib._precursor_df[matched_mask]
    speclib._precursor_df['frag_start_idx'] = frag_start_idx[matched_mask]
    speclib._precursor_df['frag_end_idx'] = frag_end_idx[matched_mask]

    speclib._fragment_mz_df = fragment_speclib._fragment_mz_df.copy()
    speclib._fragment_intensity_df = fragment_speclib._fragment_intensity_df.copy()

    return speclib

alphatims.utils.njit()
def make_np_slice(a):
    """Creates a numpy style slice from ondimensional array.
        [start, stop] => [[start, stop, 1]]

        [[start, stop],    [[start, stop, 1],
         [start, stop]] =>  [start, stop, 1]]
    """

    if len(a.shape) == 1:
        return np.array([[a[...,0],a[...,1],1]], dtype='int64')
    else:
        return np.array([a[...,0],a[...,1],np.ones_like(a[...,1])], dtype='int64').T

from numba import stencil

def normal(x, mu, sigma):
    """
    
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-np.power((x-mu)/sigma, 2)/2)

def kernel_1d(
        size: int, 
        sigma: float, 
        norm: str = 'sum'
    ): 
    """Create a one dimensional numba stencil which can be used for smoothing

    Parameters
    ----------

    size : int
        number of elements left and right from the index of interest which should be included in the kernel

    sigma : int
        standard deviation which should be used for calculating the normal distribution. 
        The density of the normal distribution is calculated on the scale of indices [-2, -1, 0, 1 ...]
    
    norm : str (default sum)
        norm which is used for scaling the weights {sum, max}

    Returns
    -------

    function
        function decorated with a numba.stencil decorator

    """
    indices = np.arange(-size,size)

    weights = normal(indices,0,sigma)

    if norm == 'sum':
        weights = weights/np.sum(weights)
    elif norm == 'max':
        weights = weights/np.max(weights)
    else:
        raise ValueError(f'norm {norm} not known')
 
    def kernel_stencil(array, indices, weights):
        return np.dot(array[indices],weights)
    
    njitted = stencil(kernel_stencil, neighborhood = ((-size,size),))

    def stencil_caller(array):
        return njitted(array, indices, weights)
    
    return stencil_caller

import alphatims.utils

@alphatims.utils.njit()
def multivariate_normal(
        x: np.ndarray, 
        mu: np.ndarray, 
        sigma: np.ndarray
    ):
    """multivariate normal distribution, probability density function
    
    Parameters
    ----------

    x : np.ndarray
        `(N, D,)`

    mu : np.ndarray
        `(1, D,)`

    sigma : np.ndarray
        `(D, D,)`

    Returns
    -------

    np.ndarray, float32
        array of shape `(N,)` with the density at each point

    """

    k = mu.shape[0]
    dx = x - mu

    # implementation is not very efficient for large N as the N x N matrix will created only for storing the diagonal
    a = np.exp(-1/2*np.diag(dx @ np.linalg.inv(sigma) @ dx.T))
    b = (np.pi*2)**(-k/2)*np.linalg.det(sigma)**(-1/2)
    #print(a*b)
    return a * b
    
def kernel_2d(
        size: int, 
        sigma: float, 
        norm: str = 'sum'
    ): 
    """Create a one dimensional numba stencil which can be used for smoothing

    Parameters
    ----------

    size : int
        number of elements left and right from the index of interest which should be included in the kernel

    sigma : int
        standard deviation which should be used for calculating the normal distribution. 
        The density of the normal distribution is calculated on the scale of indices [-2, -1, 0, 1 ...]
    
    norm : str (default sum)
        norm which is used for scaling the weights {sum, max}

    Returns
    -------

    function
        function decorated with a numba.stencil decorator

    """
    # create indicies [-2, -1, 0, 1 ...]
    x, y = np.meshgrid(np.arange(-size,size+1),np.arange(-size,size+1))
    xy = np.column_stack((x.flatten(), y.flatten())).astype('float32')

    # mean is always zero
    mu = np.array([[0., 0.]])

    # sigma is set with no covariance
    sigma_mat = np.array([[sigma,0.],[0.,sigma]])

    weights = multivariate_normal(xy, mu, sigma_mat)

    norm = 'max'
    if norm == 'sum':
            weights = weights/np.sum(weights)
    elif norm == 'max':
        weights = weights/np.max(weights)
    elif norm == 'none':
        pass
    else:
        raise ValueError(f'norm {norm} not known')
 
    def kernel_stencil(array, ix, iy, weights):
        res = 0
        for x, y, w in zip(ix, iy, weights):
            res += array[x, y] * w
        return res

    njitted = stencil(kernel_stencil, neighborhood = ((-size,size),(-size,size),))

    def stencil_caller(array):
        return njitted(array, x.flatten(), y.flatten(), weights)
    
    return stencil_caller

def kernel_2d_fft(
        size: int, 
        sigma: float, 
        norm: str = 'sum'
    ): 
    """Create a one dimensional numba stencil which can be used for smoothing

    Parameters
    ----------

    size : int
        number of elements left and right from the index of interest which should be included in the kernel

    sigma : int
        standard deviation which should be used for calculating the normal distribution. 
        The density of the normal distribution is calculated on the scale of indices [-2, -1, 0, 1 ...]
    
    norm : str (default sum)
        norm which is used for scaling the weights {sum, max}

    Returns
    -------

    function
        function decorated with a numba.stencil decorator

    """
    # create indicies [-2, -1, 0, 1 ...]
    x, y = np.meshgrid(np.arange(-size,size+1),np.arange(-size,size+1))
    xy = np.column_stack((x.flatten(), y.flatten())).astype('float32')

    # mean is always zero
    mu = np.array([[0., 0.]])

    # sigma is set with no covariance
    sigma_mat = np.array([[sigma,0.],[0.,sigma]])

    weights = multivariate_normal(xy, mu, sigma_mat)
    weights = weights.reshape(size*2+1,size*2+1)

    norm = 'max'
    if norm == 'sum':
            weights = weights/np.sum(weights)
    elif norm == 'max':
        weights = weights/np.max(weights)
    elif norm == 'none':
        pass
    else:
        raise ValueError(f'norm {norm} not known')
 
    def conv2dfft(x,y):
        return np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(y, x.shape))


    def stencil_caller(array):
        return conv2dfft(array, weights)
    
    return stencil_caller

@alphatims.utils.njit()
def estimate_peak_boundaries(
        a, 
        scan_center, 
        dia_cycle_center,
        f = 0.95,
        b = 5
    ):
    
    base_intensity = a[scan_center,dia_cycle_center]
    # determine lower limit in the mobility dimension (top border)
    trailing_intensity = base_intensity
    lower_limit = scan_center-b
    for s in range(scan_center-b,0, -1):
        intensity = a[s,dia_cycle_center]

        if trailing_intensity *f >= intensity:
            lower_limit = s
            trailing_intensity = intensity
        else: break

    # determine lower limit in the mobility dimension (lower border)
    trailing_intensity = base_intensity
    upper_limit = scan_center+b
    for s in range(scan_center+b,a.shape[0]):
        intensity = a[s,dia_cycle_center]
   
        if trailing_intensity *f>= intensity:
            upper_limit = s
            trailing_intensity = intensity
        else: break

    # determine lower limit in the dia_cycle dimension (left border)
    trailing_intensity = base_intensity
    left_limit = dia_cycle_center-b
    for p in range(dia_cycle_center-b,0,-1):
        intensity = a[scan_center,p]
       
        if trailing_intensity *f>= intensity:
            left_limit = p
            trailing_intensity = intensity
        else: break

    # determine upper limit in the dia_cycle dimension (right border)
    trailing_intensity = base_intensity
    right_limit = dia_cycle_center+b
    for p in range(dia_cycle_center+b,a.shape[1]):
        intensity = a[scan_center,p]
       
        if trailing_intensity *f>= intensity:
            right_limit = p
            trailing_intensity = intensity
        else: break

    return np.array([lower_limit, upper_limit], dtype='int32'), np.array([left_limit, right_limit], dtype='int32')

@alphatims.utils.njit()
def find_peaks(a):
    """accepts a dense representation and returns the top three peaks
    
    """

    scan = []
    dia_cycle = []
    intensity = []

    for s in range(2,a.shape[0]-2):
        for p in range(2,a.shape[1]-2):
             
            isotope_is_peak = (a[s-2,p] < a[s-1,p] < a[s,p] > a[s+1,p] > a[s+2,p])
            isotope_is_peak &= (a[s,p-2] < a[s,p-1] < a[s,p] > a[s,p+1] > a[s,p+2])

            if isotope_is_peak:
                intensity.append(a[s,p])
                scan.append(s)
                dia_cycle.append(p)
    
    scan = np.array(scan)
    dia_cycle = np.array(dia_cycle)
    intensity = np.array(intensity)

    idx = np.argsort(intensity)[::-1][:3]

    scan = scan[idx]
    dia_cycle = dia_cycle[idx]
    intensity = intensity[idx]



    return scan, dia_cycle, intensity

@alphatims.utils.njit()
def get_precursor_mz(dense, scan, dia_cycle):
    """ create a fixed window around the peak and extract the mz as well as the intensity
    
    """
    # window size around peak
    w = 2

    # extract mz
    mz_window = dense[1,0,max(scan-w,0):scan+w,max(dia_cycle-w,0):dia_cycle+w].flatten()
    mask = (mz_window>0)
    fraction_nonzero = np.mean(mask.astype('int8'))

    if fraction_nonzero > 0:
        mz = np.mean(mz_window[mask])
    else:
        mz = -1
    #extract intensity
    intensity = np.sum(dense[0,0,max(scan-w,0):scan+w,max(dia_cycle-w,0):dia_cycle+w])

    return fraction_nonzero, mz, intensity
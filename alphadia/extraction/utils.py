# native imports
import logging
from ctypes import Structure, c_double
from typing import Tuple, Union, List

# alphadia imports

# alpha family imports
import alphatims.bruker
import alphatims.utils
from alphabase.spectral_library.base import SpecLibBase

# third party imports
import pandas as pd
import numpy as np
import numba as nb
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ISOTOPE_DIFF = 1.0032999999999674

def recursive_update(
            full_dict: dict, 
            update_dict: dict
        ):
        """recursively update a dict with a second dict. The dict is updated inplace.

        Parameters
        ----------
        full_dict : dict
            dict to be updated, is updated inplace.

        update_dict : dict
            dict with new values

        Returns
        -------
        None

        """
        for key, value in update_dict.items():
            if key in full_dict.keys():
                if isinstance(value, dict):
                    recursive_update(full_dict[key], update_dict[key])
                else:
                    full_dict[key] = value
            else:
                full_dict[key] = value

def density_scatter(x, y, axis=None, **kwargs):

    if axis is None:
        axis = plt.gca()

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    axis.scatter(x, y, c=z, **kwargs)


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
    speclib: SpecLibBase, 
    fragment_speclib: SpecLibBase,
    verbose = True
) -> SpecLibBase:
    """Reannotate an alphabase SpecLibBase library with framents from a different SpecLibBase library

    Parameters
    ----------
    speclib: SpecLibBase
        Spectral library which contains the precursors to be annotated. All fragments mz and fragment intensities will be removed.

    fragment_speclib: SpecLibBase
        Spectral library which contains the donor precursors whose fragments should be used.

    Returns
    -------

    SpecLibBase
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
    
    speclib._precursor_df = speclib._precursor_df[matched_mask].copy()
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

    return njitted

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
def estimate_peak_boundaries_symmetric(
        a, 
        scan_center, 
        dia_cycle_center,
        f_mobility = 0.95,
        f_rt = 0.95,
        min_size_mobility = 5,
        max_size_mobility = 20,
        min_size_rt = 5,
        max_size_rt = 10
    ):
    

    # determine limits in the mobility dimension
    trailing_intensity = a[scan_center,dia_cycle_center]

    # The number of steps into both directions is limited by:
    # 1. The closest border (top or bottom)
    # 2. The max_size defined
    mobility_max_len = min(a.shape[0], a.shape[0]-scan_center)
    mobility_max_len = int(min(mobility_max_len, max_size_mobility))
    
    mobility_limit = min_size_mobility


    for s in range(min_size_mobility,mobility_max_len):

        intensity = (a[scan_center-s,dia_cycle_center]+a[scan_center+s,dia_cycle_center])/2
        if trailing_intensity * f_mobility >= intensity:
            mobility_limit = s
            trailing_intensity = intensity
        else: break

    mobility_limits = np.array([scan_center-mobility_limit, scan_center+mobility_limit], dtype='int32')
    

    # determine limits in the precursor cycle dimension
    trailing_intensity = a[scan_center,dia_cycle_center]

    # The number of steps into both directions is limited by:
    # 1. The closest border (top or bottom)
    # 2. The max_size defined
    dia_cycle_max_len = min(a.shape[1], a.shape[1]-dia_cycle_center)
    dia_cycle_max_len = int(min(dia_cycle_max_len, max_size_rt))
    
    dia_cycle_limit = min_size_rt

    for s in range(min_size_rt, dia_cycle_max_len):

        intensity = (a[scan_center,dia_cycle_center-s]+a[scan_center,dia_cycle_center+s])/2
        if trailing_intensity * f_rt >= intensity:
            dia_cycle_limit = s
            trailing_intensity = intensity
        else: break

    dia_cycle_limits = np.array([dia_cycle_center-dia_cycle_limit, dia_cycle_center+dia_cycle_limit], dtype='int32')

    return mobility_limits, dia_cycle_limits

def plt_limits(mobility_limits, dia_cycle_limits):
    mobility_len = mobility_limits[1]-mobility_limits[0]
    dia_cycle_len = dia_cycle_limits[1]-dia_cycle_limits[0]

    rect = patches.Rectangle((dia_cycle_limits[0], mobility_limits[0] ), dia_cycle_len, mobility_len , linewidth=1, edgecolor='r', facecolor='none')

    return rect

@alphatims.utils.njit()
def find_peaks(a, top_n=3):
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

    idx = np.argsort(intensity)[::-1][:top_n]

    scan = scan[idx]
    dia_cycle = dia_cycle[idx]
    intensity = intensity[idx]

    return scan, dia_cycle, intensity

@alphatims.utils.njit()
def get_precursor_mz(dense_intensity, dense_mz, scan, dia_cycle):
    """ create a fixed window around the peak and extract the mz as well as the intensity
    
    """
    # window size around peak
    w = 4

    # extract mz
    mz_window = dense_mz[
        max(scan-w,0):scan+w,
        max(dia_cycle-w,0):dia_cycle+w
    ].flatten()

    mask = (mz_window > 0)
    fraction_nonzero = np.mean(mask.astype('int8'))

    if fraction_nonzero > 0:
        mz = np.mean(mz_window[mask])
    else:
        mz = -1

    return fraction_nonzero, mz



@alphatims.utils.njit()
def amean1(array):
    out = np.zeros(array.shape[0])
    for i in range(len(out)):
        out[i] = np.mean(array[i])
    return out

@alphatims.utils.njit()
def amean0(array):
    out = np.zeros(array.shape[1])
    for i in range(len(out)):
        out[i] = np.mean(array[:,i])
    return out

@alphatims.utils.njit()
def astd0(array):
    out = np.zeros(array.shape[1])
    for i in range(len(out)):
        out[i] = np.std(array[:,i])
    return out

@alphatims.utils.njit()
def astd1(array):
    out = np.zeros(array.shape[0])
    for i in range(len(out)):
        out[i] = np.std(array[i])
    return out

@alphatims.utils.njit()
def _and_envelope(input_profile, output_envelope):
    """
    Calculate the envelope of a profile spectrum.
    """
    
    for i in range(1, len(input_profile) - 2):
        if (input_profile[i] < input_profile[i-1]) & (input_profile[i] < input_profile[i+1]):
            output_envelope[i] = (input_profile[i-1] + input_profile[i+1]) / 2

@alphatims.utils.njit()
def _or_envelope(input_profile, output_envelope):
    """
    Calculate the envelope of a profile spectrum.
    """
    
    for i in range(1, len(input_profile) - 2):
        if (input_profile[i] < input_profile[i-1]) or (input_profile[i] < input_profile[i+1]):
            output_envelope[i] = (input_profile[i-1] + input_profile[i+1]) / 2
       

@alphatims.utils.njit()
def and_envelope(profile):
    envelope = profile.copy()

    for i in range(len(profile)):
        _and_envelope(profile[i],envelope[i])

    return envelope

@alphatims.utils.njit()
def or_envelope(profile):
    envelope = profile.copy()

    for i in range(len(profile)):
        _or_envelope(profile[i],envelope[i])

    return envelope

@alphatims.utils.njit()
def calculate_correlations(
        dense_template_profile, 
        dense_fragments_profile
    ):
    """Calculate correlation metrics between fragments and precursors

    Parameter
    ---------

    dense_precursor : np.ndarray
        (S, F) 

    dense_fragments : np.ndarray
        (N, S, F)

    Returns
    -------

    np.ndarray
        (N)

    """




    # F fragments and 1 precursors are concatenated, resulting in a (F+1, F+1) correlation matrix
    corr_frame = np.corrcoef(dense_fragments_profile, dense_template_profile)
    #corr_scan = np.corrcoef(fragment_scan_profile, precursor_scan_profile)
    # The first 
    
    mean_frame_corr = amean0(corr_frame[:-1,:-1])-1/len(corr_frame[:-1,:-1])
    #mean_scan_corr = amean0(corr_scan[:-1,:-1])-1/len(corr_scan[:-1,:-1])

    prec_frame_corr = corr_frame[-1,:-1]
    #prec_scan_corr = corr_scan[-1,:-1]
    
    return np.stack((mean_frame_corr, prec_frame_corr))

@alphatims.utils.njit()
def calculate_mass_deviation(
        dense_fragments_mz,
        fragments_mz,
        size = 4
    ):
    """Calculate the mass deviation between the observed fragment masses and the calculated fragment masses

    Parameters
    ----------
    dense_fragments_mz : np.ndarray
        (N, S, F)

    fragments_mz : np.ndarray
        (N)

    size : int, optional
        Size of the window around the center, by default 4

    Returns
    -------

    (np.ndarray, np.ndarray)
        (N), (N) mass deviation in ppm and number of observations
        
    
    """

    out_arr = np.zeros((fragments_mz.shape))
    num_observations = np.zeros((fragments_mz.shape))

    scan_center = dense_fragments_mz.shape[1] // 2
    frame_center = dense_fragments_mz.shape[2] // 2
    center_view = dense_fragments_mz#[:, scan_center - size:scan_center + size, frame_center - size:frame_center + size]

    idxs, scans, precs = np.nonzero(center_view > 0)

    for idx, scan, prec in zip(idxs, scans, precs):

        out_arr[idx] += (center_view[idx, scan, prec] - fragments_mz[idx]) / fragments_mz[idx] * 1e6
        num_observations[idx] += 1


    mass_error = out_arr / num_observations
    fraction_nonzero = num_observations / (center_view[0].shape[0]*center_view[0].shape[1])
    
    return mass_error, fraction_nonzero, num_observations

@alphatims.utils.njit()
def calc_isotopes_center(
        mz,
        charge,
        num_isotopes
    ):
    """Calculate the mass to charge ratios for a given number of isotopes, numba compatible
    
    Parameters
    ----------

    mz : float
        first monoisotopic mass to charge ratio of the precursor

    charge : int
        charge of the precursor, must be positive and larger than 1

    num_isotopes : int      
        number of isotopes to calculate

    Returns
    -------
    np.ndarray
        (num_isotopes) mass to charge ratios of the isotopes
        
    """

    out_mz = np.arange(0, num_isotopes)
    out_mz = out_mz * ISOTOPE_DIFF/max(charge, 1)
    out_mz += mz

    return out_mz

def get_isotope_columns(colnames):
    isotopes = []
    for col in colnames:
        if col[:2] == 'i_':
            try:
                isotopes.append(int(col[2:]))
            except:
                logging.warning(f'Column {col} does not seem to be a valid isotope column')
    
    isotopes = np.array(sorted(isotopes))

    if not np.all(np.diff(isotopes) == 1):
        logging.warning(f'Isotopes are not consecutive')

    return isotopes

@alphatims.utils.njit()
def mass_range(
        mz_list,
        ppm_tolerance
    ):

    out_mz = np.zeros((len(mz_list),2))
    out_mz[:,0] = mz_list - ppm_tolerance * mz_list/(10**6)
    out_mz[:,1] = mz_list + ppm_tolerance * mz_list/(10**6)
    return out_mz



def recalibrate_mz(df, calibration_estimator, plot=True, plot_title='', save_path=None):
    logging.info(f"Performing calibration with {len(df)} features")

    observed_mz = df['mz_observed'].values
    calculated_mz = df['mz_predicted'].values
    order = np.argsort(calculated_mz)

    observed_mz = observed_mz[order]
    calculated_mz = calculated_mz[order]

    observed_mass_error = (observed_mz - calculated_mz) / calculated_mz * 1e6

    calibration_estimator.fit(calculated_mz, observed_mz)

    calibrated_mz = calibration_estimator.predict(calculated_mz)

    calibration_curve = (calibrated_mz - calculated_mz) / calculated_mz * 1e6
    
    calibrated_mass_error = (observed_mz - calibrated_mz) / calibrated_mz * 1e6

    mass_error_95 = np.mean(np.abs(np.percentile(calibrated_mass_error, [2.5,97.5])))
    mass_error_99 = np.mean(np.abs(np.percentile(calibrated_mass_error, [0.5,99.5])))
    mass_error_70 = np.mean(np.abs(np.percentile(calibrated_mass_error, [15,85])))

    if plot:

        fig, ax = plt.subplots(ncols=2, figsize=(6.5,3.5))
        density_scatter(calculated_mz, observed_mass_error,ax[0], s=1)
        ax[0].plot(calculated_mz, calibration_curve, color='red')
        ax[0].set_ylim(-120, 120)

        ax[0].text(0.05, 0.95, f'{len(observed_mz):,} datapoints',
            horizontalalignment='left',
            verticalalignment='top',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='white', capstyle="round")],
            transform=ax[0].transAxes)
        
        ax[0].set_ylabel('Mass error (ppm)')
        ax[0].set_xlabel('mz')

        density_scatter(calculated_mz, calibrated_mass_error,ax[1], s=1)
        ax[1].plot([np.min(calculated_mz), np.max(calculated_mz)], [0,0], color='red')
        ax[1].set_ylim(-30, 30)
        ax[1].set_ylabel('Mass error (ppm)')
        ax[1].set_xlabel('mz')
        
        
        ax[1].text(0.05, 0.95, f'95% $\delta$ {mass_error_95:,.2f} ppm',
            horizontalalignment='left',
            verticalalignment='top',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='white', capstyle="round")],
            transform=ax[1].transAxes)
        
        ax[1].text(0.05, 0.88, f'99% $\delta$ {mass_error_99:,.2f} ppm',
            horizontalalignment='left',
            verticalalignment='top',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='white', capstyle="round")],
            transform=ax[1].transAxes)
        
        fig.suptitle(plot_title)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    return mass_error_70, mass_error_95, mass_error_99


def function_call(q):
    q.put('X' * 1000000)

def modify(n, x, s, A):
    n.value **= 2
    x.value **= 2
    s.value = s.value.upper()
    for a in A:
        a.x **= 2
        a.y **= 2

class Point(Structure):
    _fields_ = [('x', c_double), ('y', c_double)]


@alphatims.utils.njit()
def tile(a, n):
    return np.repeat(a, n).reshape(-1, n).T.flatten()

@alphatims.utils.njit
def make_slice_1d(
        start_stop
    ):
    """Numba helper function to create a 1D slice object from a start and stop value.

        e.g. make_slice_1d([0, 10]) -> np.array([[0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (2,) containing the start and stop value.

    Returns
    -------
    np.ndarray
        Array of shape (1,3) containing the start, stop and step value.

    """
    return np.array([[start_stop[0], start_stop[1],1]], dtype='uint64')

@alphatims.utils.njit
def make_slice_2d(
        start_stop
    ):
    """Numba helper function to create a 2D slice object from multiple start and stop value.

        e.g. make_slice_2d([[0, 10], [0, 10]]) -> np.array([[0, 10, 1], [0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (N, 2) containing the start and stop value for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing the start, stop and step value for each dimension.

    """

    out = np.ones((start_stop.shape[0], 3), dtype='uint64')
    out[:,0] = start_stop[:,0]
    out[:,1] = start_stop[:,1]
    return out

@alphatims.utils.njit
def expand_if_odd(
        limits
    ):
    """Numba helper function to expand a range if the difference between the start and stop value is odd.

        e.g. expand_if_odd([0, 11]) -> np.array([0, 12])

    Parameters
    ----------
    limits : np.ndarray
        Array of shape (2,) containing the start and stop value.

    Returns
    -------
    np.ndarray
        Array of shape (2,) containing the expanded start and stop value.

    """
    if (limits[1] - limits[0])%2 == 1:
        limits[1] += 1
    return limits

@alphatims.utils.njit
def fourier_filter(
        dense_stack, 
        kernel
    ):
    """Numba helper function to apply a gaussian filter to a dense stack. 
    The filter is applied as convolution wrapping around the edges, calculated in fourier space.

    As there seems to be no easy option to perform 2d fourier transforms in numba, the numpy fft is used in object mode.
    During multithreading the GIL has to be acquired to use the numpy fft and is realeased afterwards.

    Parameters
    ----------

    dense_stack : np.ndarray
        Array of shape (2, n_precursors, n_observations ,n_scans, n_cycles) containing the dense stack.

    kernel : np.ndarray
        Array of shape (k0, k1) containing the gaussian kernel.

    Returns
    -------
    smooth_output : np.ndarray
        Array of shape (n_precursors, n_observations, n_scans, n_cycles) containing the filtered dense stack.

    """

    k0 = kernel.shape[0]
    k1 = kernel.shape[1]

    # make sure both dimensions are even
    scan_mod = dense_stack.shape[3] % 2
    frame_mod = dense_stack.shape[4] % 2

    scan_size = dense_stack.shape[3] - scan_mod
    frame_size = dense_stack.shape[4] - frame_mod

    smooth_output = np.zeros((
        dense_stack.shape[1],
        dense_stack.shape[2], 
        scan_size,
        frame_size,
    ), dtype='float32')

    
    fourier_filter = np.fft.rfft2(kernel, smooth_output.shape[2:])

    for i in range(smooth_output.shape[0]):
        for j in range(smooth_output.shape[1]):
            layer = dense_stack[0,i,j,:scan_size,:frame_size]

            smooth_output[i,j] = np.fft.irfft2(np.fft.rfft2(layer) * fourier_filter)
                
    #with nb.objmode(smooth_output='float32[:,:,:,:]'):
    #    # roll back to original position
    #    smooth_output = np.roll(smooth_output, -k0//2, axis=2)
    #     smooth_output = np.roll(smooth_output, -k1//2, axis=3)

    return smooth_output


def calculate_score_groups(
        precursors_flat, 
        group_channels = False
    ):
    """
    Calculate score based on elution group and decoy status.

    Parameters
    ----------

    precursors_flat : pandas.DataFrame
        Precursor dataframe. Must contain columns 'elution_group_idx' and 'decoy'.

    group_channels : bool
        If True, all channels from a given precursor will be grouped together.

    Returns
    -------

    score_groups : pandas.DataFrame
        Updated precursor dataframe with score_group_idx column.
    """

    @nb.njit
    def channel_score_groups(
            elution_group_idx, 
            decoy
        ):
        score_groups = np.zeros(len(elution_group_idx), dtype=np.int32)
        current_group = 0
        current_eg = elution_group_idx[0]
        current_decoy = decoy[0]
        
        for i in range(len(elution_group_idx)):
            if (elution_group_idx[i] != current_eg) or (decoy[i] != current_decoy):
                current_group += 1
                current_eg = elution_group_idx[i]
                current_decoy = decoy[i]
            
            score_groups[i] = current_group
        return score_groups

    precursors_flat = precursors_flat.sort_values(by=['elution_group_idx', 'decoy'])

    if group_channels:
        precursors_flat['score_group_idx'] = channel_score_groups(precursors_flat['elution_group_idx'].values, precursors_flat['decoy'].values)
    else:
        precursors_flat['score_group_idx'] = np.arange(len(precursors_flat))

    return precursors_flat.sort_values(by=['score_group_idx']).reset_index(drop=True)
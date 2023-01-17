import alphabase.peptide.fragment
import numpy as np
from tqdm import tqdm

import pandas as pd
import logging
from .data import TimsTOFDIA
from . import utils
import alphatims.utils
from numba import njit, objmode, types
from numba.experimental import jitclass
import numba as nb

import matplotlib.pyplot as plt

from alphabase.constants.element import (
    CHEM_MONO_MASS
)

ISOTOPE_DIFF = CHEM_MONO_MASS['13C'] - CHEM_MONO_MASS['C']

class MS1CentricCandidateSelection(object):

    def __init__(self, 
            dia_data,
            precursors_flat, 
            rt_tolerance = 30,
            mobility_tolerance = 0.03,
            mz_tolerance = 120,
            num_isotopes = 2,
            num_candidates = 3,
            rt_column = 'rt_library',  
            precursor_mz_column = 'mz_library',
            mobility_column = 'mobility_library',
            thread_count = 1,
            debug = False
        ):
        """select candidates for MS2 extraction based on MS1 features

        Parameters
        ----------

        dia_data : alphadia.extraction.data.TimsTOFDIA
            dia data object

        precursors_flat : pandas.DataFrame
            flattened precursor dataframe

        rt_tolerance : float, optional
            rt tolerance in seconds, by default 30

        mobility_tolerance : float, optional
            mobility tolerance, by default 0.03

        mz_tolerance : float, optional
            mz tolerance in ppm, by default 120

        num_isotopes : int, optional
            number of isotopes to consider, by default 2

        num_candidates : int, optional
            number of candidates to select, by default 3

        Returns
        -------

        pandas.DataFrame
            dataframe containing the choosen candidates
            columns:
                - index: index of the precursor in the flattened precursor dataframe
                - fraction_nonzero: fraction of non-zero intensities around the candidate center

        
        """
        self.jit_data = dia_data.jitclass()
        self.precursors_flat = precursors_flat

        self.debug = debug
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.mobility_tolerance = mobility_tolerance
        self.num_isotopes = num_isotopes
        self.num_candidates = num_candidates

        self.thread_count = thread_count

        self.rt_column = rt_column
        self.precursor_mz_column = precursor_mz_column
        self.mobility_column = mobility_column

        k = 20
        self.kernel = gaussian_kernel_2d(k,5,9).astype(np.float32)
        

    def __call__(self):

        # initialize input container
        precursor_container = self.precursors_to_jit()

        # initialize output container
        candidate_container = CandidateContainer(
            precursor_container.n_precursors(), 
            self.num_candidates
        )

        pjit_fun = alphatims.utils.pjit(
            get_candidate_pjit,
            thread_count=self.thread_count
        )

        pjit_fun(
            range(precursor_container.n_elution_groups()),
            precursor_container,
            candidate_container,
            self.jit_data, 

            # single values
            self.debug,

            # single values
            self.rt_tolerance,
            self.mobility_tolerance,
            self.mz_tolerance,
            self.num_candidates,
            self.kernel
        )

        return self.candidates_to_df(candidate_container)


    def candidates_to_df(self, container):
        return container

    def precursors_to_jit(self):

        precursors_flat = self.precursors_flat.sort_values('elution_group_idx').reset_index(drop=True)
        eg_grouped = precursors_flat.groupby('elution_group_idx')

        n_elution_groups = len(eg_grouped)
        
        # has length of number of elution groups
        elution_group_idx = np.zeros(n_elution_groups, dtype=np.uint32)
        rt_values = np.zeros(n_elution_groups, dtype=np.float32)
        mobility_values = np.zeros(n_elution_groups, dtype=np.float32)
        charge_values = np.zeros(n_elution_groups, dtype=np.uint8)
        precursor_start_stop = np.zeros((n_elution_groups,2), dtype=np.uint32)
        
        precursor_count = 0
        for i, (name, grouped) in enumerate(eg_grouped):
            
            first_member = grouped.iloc[0]
            elution_group_idx[i] = first_member['elution_group_idx']
            rt_values[i] = first_member[self.rt_column]
            mobility_values[i] = first_member[self.mobility_column]
            charge_values[i] = first_member['charge']

            precursor_start_stop[i,0] = precursor_count
            precursor_start_stop[i,1] = precursor_count + len(grouped)
            precursor_count += len(grouped)

        # has length of number of precursors
        precursor_idx = precursors_flat['precursor_idx'].values.astype(np.uint32)
        precursor_mz = precursors_flat[self.precursor_mz_column].values.astype(np.float32)
        frag_start_stop = np.stack([
            precursors_flat['frag_start_idx'].values, 
            precursors_flat['frag_stop_idx'].values
        ], axis=1).astype(np.uint32)

        print(frag_start_stop.shape)

        decoy = precursors_flat['decoy'].values.astype(np.uint8)
        
        # create isotope apex offset
        if 'isotope_apex_offset' in precursors_flat.columns:
            isotope_apex_offset = precursors_flat['isotope_apex_offset'].values.astype(np.uint32)
        else:
            isotope_apex_offset = np.zeros_like(precursor_idx).astype(np.uint32)


        return ElutionGroupContainer(
            elution_group_idx,
            rt_values,
            mobility_values,
            charge_values,
            precursor_start_stop,
            precursor_idx,
            precursor_mz,
            frag_start_stop,
            decoy,
            isotope_apex_offset
        )



from matplotlib import patches

def visualize_candidates(profile, smooth, peak_scan, peak_cycle, limits_scan, limits_cycle):
    print(limits_scan, limits_cycle)
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(smooth, aspect='equal')
    ax[1].imshow(profile, aspect='equal')
    for i in range(len(peak_scan)):
        ax[0].scatter(peak_cycle[i], peak_scan[i], c='r')
        ax[0].text(peak_cycle[i]+1, peak_scan[i]+1, str(i), color='r')

        ax[1].scatter(peak_cycle[i], peak_scan[i], c='r')

        limit_scan = limits_scan[i]
        limit_cycle = limits_cycle[i]

        ax[0].add_patch(patches.Rectangle(
            (limit_cycle[0], limit_scan[0]),   # (x,y)
            limit_cycle[1]-limit_cycle[0],          # width
            limit_scan[1]-limit_scan[0],          # height
            fill=False,
            edgecolor='r'
        ))

        ax[0].add_patch(patches.Rectangle(
            (limit_cycle[0], limit_scan[0]),   # (x,y)
            limit_cycle[1]-limit_cycle[0],          # width
            limit_scan[1]-limit_scan[0],          # height
            fill=False,
            edgecolor='r'
        ))

        logging.info(f'peak {i}, scan: {peak_scan[i]}, cycle: {peak_cycle[i]}')
        # width and height
        logging.info(f'height: {limit_scan[1]-limit_scan[0]}, width: {limit_cycle[1]-limit_cycle[0]}')

    plt.show()

def gaussian_kernel_2d(
        size: int, 
        sigma_x: float, 
        sigma_y: float,
        norm: str = 'sum'
    ): 
    """

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
    x, y = np.meshgrid(np.arange(-size//2,size//2),np.arange(-size//2,size//2))
    xy = np.column_stack((x.flatten(), y.flatten())).astype('float32')

    # mean is always zero
    mu = np.array([[0., 0.]])

    # sigma is set with no covariance
    sigma_mat = np.array([[sigma_x,0.],[0.,sigma_y]])

    weights = utils.multivariate_normal(xy, mu, sigma_mat)
    return weights.reshape(size,size)

@alphatims.utils.njit
def make_slice_1d(start_stop):
    return np.array([[start_stop[0], start_stop[1],1]])

@alphatims.utils.njit
def make_slice_2d(start_stop):

    out = np.ones((start_stop.shape[0], 3), dtype='uint64')
    out[:,0] = start_stop[:,0]
    out[:,1] = start_stop[:,1]
    return out


def get_candidate_pjit(
        i, 
        precursor_container,
        candidate_container,
        jit_data, 
        debug,

        # single values
        rt_tolerance,
        mobility_tolerance,
        mz_tolerance,
        num_candidates,
        kernel
    ):
    """find candidates in a single elution group

    input arrays to this function can map to the whole elution group or to precursors in the elution group
    mapping is facilitated by the precursor_start and precursor_stop arrays

    """

    precursor_start = precursor_container.precursor_start_stop[i,0]
    precursor_stop = precursor_container.precursor_start_stop[i,1]
    
    n_precursors = precursor_stop -precursor_start

    precursor_slice = slice(precursor_start, precursor_stop)
    
    # shared values are the same for all precursors in the elution group
    shared_elution_group_idx = precursor_container.elution_group_idx[i]
    shared_rt = precursor_container.rt_values[i]
    shared_mobility = precursor_container.mobility_values[i]
    shared_charge = precursor_container.charge_values[i]


    # shared rt values
    shared_rt_limits = np.array([
        shared_rt-rt_tolerance, 
        shared_rt+rt_tolerance
        
    ])
    shared_frame_limits = make_slice_1d(
        expand_if_odd(
            jit_data.return_frame_indices(
                shared_rt_limits,
                True
            )
        )
    )

    # shared mobility values
    shared_mobility_limits = np.array([
        shared_mobility+mobility_tolerance,
        shared_mobility-mobility_tolerance
    ])
    shared_scan_limits = make_slice_1d(
        expand_if_odd(
            jit_data.return_scan_indices(
                shared_mobility_limits
            )
        )
    )
    
    local_isotope_apex_offset = precursor_container.isotope_apex_offset[precursor_slice]
    local_precursor_idx = precursor_container.precursor_idx[precursor_slice]
    local_precursor_mz = precursor_container.precursor_mz[precursor_slice]
    local_decoy = precursor_container.decoy[precursor_slice]
    local_top_isotope_mz = local_precursor_mz + local_isotope_apex_offset * ISOTOPE_DIFF / shared_charge
    

    # sort by precursor m/z !!! required for alphatims handling
    precursor_order = np.argsort(local_top_isotope_mz)
    local_isotope_apex_offset = local_isotope_apex_offset[precursor_order]
    local_precursor_idx = local_precursor_idx[precursor_order]
    local_precursor_mz = local_precursor_mz[precursor_order]
    local_decoy = local_decoy[precursor_order]
    local_top_isotope_mz = local_top_isotope_mz[precursor_order]

    # elution group size is then expanded to include the isotope apex

    

    local_mz_limits = utils.mass_range(local_top_isotope_mz, mz_tolerance)
    local_tof_limits = make_slice_2d(jit_data.return_tof_indices(
        local_mz_limits
    ))
    

    dense, precursor_index = jit_data.get_dense(
        shared_frame_limits,
        shared_scan_limits,
        local_tof_limits,
        np.array([[-1.,-1.]])
    )
    

    smooth_dense = fourier_filter(dense, kernel)

    if debug:
        with objmode():
        
            n_plots = smooth_dense.shape[0]
            fig, axs = plt.subplots(n_plots,2, figsize=(n_plots*4,10))

            for j in range(n_plots):
                axs[j,0].imshow(dense[0,j,0])
                axs[j,1].imshow(smooth_dense[j,0])
            plt.show()

    

    for precursor_in_batch, idx in enumerate(local_precursor_idx):
        # select precursor
        smooth_precursor = smooth_dense[precursor_in_batch]
        # sum different frames
        smooth_precursor = np.sum(smooth_precursor, axis=0)

        peak_scan_list, peak_cycle_list, peak_intensity_list = utils.find_peaks(
            smooth_precursor, 
            top_n=num_candidates
        )

        

        for candidate_in_precursor, (scan, cycle, intensity) in enumerate(
            zip(
                peak_scan_list, 
                peak_cycle_list, 
                peak_intensity_list
                )
            ):
            candidate_index = (precursor_start + precursor_in_batch) * num_candidates + candidate_in_precursor
  
            candidate_container.precursor_idx[candidate_index] = idx
            candidate_container.elution_group_idx[candidate_index] = shared_elution_group_idx
            

    pass


@alphatims.utils.njit
def expand_if_odd(limits):
    if (limits[1] - limits[0])%2 == 1:
        limits[1] += 1
    return limits


@alphatims.utils.njit
def fourier_filter(dense_stack, kernel):

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

    with objmode(smooth_output='float32[:,:,:,:]'):
        fourier_filter = np.fft.rfft2(kernel, smooth_output.shape[2:])

        for i in range(smooth_output.shape[0]):
            for j in range(smooth_output.shape[1]):
                layer = dense_stack[0,i,j,:scan_size,:frame_size]
    
                smooth_output[i,j] = np.fft.irfft2(np.fft.rfft2(layer) * fourier_filter)
                

        # roll back to original position
        smooth_output = np.roll(smooth_output, -k0//2, axis=2)
        smooth_output = np.roll(smooth_output, -k1//2, axis=3)

    return smooth_output

@jitclass()
class ElutionGroupContainer:

    elution_group_idx: nb.uint32[::1]
    elution_group_iterator: nb.uint32[::1]
    rt_values: nb.float32[::1]
    mobility_values: nb.float32[::1]
    charge_values: nb.uint8[::1]
    precursor_start_stop: nb.uint32[:,::1]

    precursor_idx: nb.uint32[::1]
    precursor_mz:nb.float32[::1]
    frag_start_stop: nb.uint32[:,::1]
    decoy: nb.uint8[::1]
    isotope_apex_offset: nb.uint32[::1]

    def __init__(
            self, 
            elution_group_idx,
            rt_values,
            mobility_values,
            charge_values,
            precursor_start_stop,
            precursor_idx,
            precursor_mz,
            frag_start_stop,
            decoy,
            isotope_apex_offset
        ) -> None:

        # have length of number of elution groups
        self.elution_group_idx = elution_group_idx
        self.elution_group_iterator = np.arange(len(elution_group_idx), dtype=np.uint32)
        self.rt_values = rt_values
        self.mobility_values = mobility_values
        self.charge_values = charge_values
        self.precursor_start_stop = precursor_start_stop

        # have length of number of precursors
        self.precursor_idx = precursor_idx
        self.precursor_mz = precursor_mz
        self.frag_start_stop = frag_start_stop
        self.decoy = decoy
        self.isotope_apex_offset = isotope_apex_offset

    def n_precursors(self):
        return len(self.precursor_idx)

    def n_elution_groups(self):
        return len(self.elution_group_idx)


       

@jitclass([
    ('n_precursors', nb.int64), 
    ('n_candidates', nb.int64),
    ('mz_observed', nb.float64[::1]),
    ('mass_error', nb.float64[::1]),
    ('fraction_nonzero', nb.float64[::1]),
    ('intensity', nb.float64[::1]),
    ('scan_limit', nb.int64[:, ::1]),
    ('scan_center', nb.int64[::1]),
    ('frame_limit', nb.int64[:, ::1]),
    ('frame_center', nb.int64[::1]),
    ('precursor_idx', nb.int64[::1]),
    ('elution_group_idx', nb.int64[::1]),
    ('decoy', nb.int64[::1])
])

class CandidateContainer:
    def __init__(
            self, 
            n_precursors, 
            n_candidates
        ) -> None:

        self.n_precursors = n_precursors
        self.n_candidates = n_candidates
        self.n_candidates = n_candidates * n_precursors

        self.mz_observed = np.zeros(self.n_candidates, dtype=np.float64)
        self.mass_error = np.zeros(self.n_candidates, dtype=np.float64)
        self.fraction_nonzero = np.zeros(self.n_candidates, dtype=np.float64)
        self.intensity = np.zeros(self.n_candidates, dtype=np.float64)

        self.scan_limit = np.zeros((self.n_precursors, 2), dtype=np.int64)
        self.scan_center = np.zeros(self.n_precursors, dtype=np.int64)

        self.frame_limit = np.zeros((self.n_precursors, 2), dtype=np.int64)
        self.frame_center = np.zeros(self.n_precursors, dtype=np.int64)

        self.precursor_idx = np.zeros(self.n_candidates, dtype=np.int64)
        self.elution_group_idx = np.zeros(self.n_candidates, dtype=np.int64)
        self.decoy = np.zeros(self.n_candidates, dtype=np.int64)

"""
    def get_candidates(self, i):

        rt_limits = np.array(
            [
                self.precursors_flat[self.rt_column].values[i]-self.rt_tolerance, 
                self.precursors_flat[self.rt_column].values[i]+self.rt_tolerance
            ]
        )
        frame_limits = utils.make_np_slice(
            self.dia_data.return_frame_indices(
                rt_limits,
                True
            )
        )

        mobility_limits = np.array(
            [
                self.precursors_flat[self.mobility_column].values[i]+self.mobility_tolerance,
                self.precursors_flat[self.mobility_column].values[i]-self.mobility_tolerance

            ]
        )
        scan_limits = utils.make_np_slice(
            self.dia_data.return_scan_indices(
                mobility_limits,
            )
        )

        precursor_mz = self.precursors_flat[self.precursor_mz_column].values[i]
        isotopes = utils.calc_isotopes_center(precursor_mz,self.precursors_flat.charge.values[i], self.num_isotopes)
        isotope_limits = utils.mass_range(isotopes, self.mz_tolerance)
        tof_limits = utils.make_np_slice(
            self.dia_data.return_tof_indices(
                isotope_limits,
            )
        )

        dense = self.dia_data.get_dense(
            frame_limits,
            scan_limits,
            tof_limits,
            np.array([[-1.,-1.]])
        )

        profile = np.sum(dense[0], axis=0)

        if profile.shape[0] <  6 or profile.shape[1] < 6:
            return []

        # smooth intensity channel
        new_height = profile.shape[0] - profile.shape[0]%2
        new_width = profile.shape[1] - profile.shape[1]%2
        smooth = self.kernel(profile)

        
        
        # cut first k elements from smooth representation with k being the kernel size
        # due to fft (?) smooth representation is shifted
        smooth = smooth[self.k:,self.k:]

        if self.debug:
            old_profile = profile.copy()
        profile[:smooth.shape[0], :smooth.shape[1]] = smooth
        

        out = []

        # get all peak candidates
        peak_scan, peak_cycle, intensity = utils.find_peaks(profile, top_n=self.num_candidates)

    
        limits_scan = []
        limits_cycle = []
        for j in range(len(peak_scan)):
            limit_scan, limit_cycle = utils.estimate_peak_boundaries_symmetric(profile, peak_scan[j], peak_cycle[j], f=0.99)
            limits_scan.append(limit_scan)
            limits_cycle.append(limit_cycle)

        for limit_scan, limit_cycle in zip(limits_scan, limits_cycle):
            fraction_nonzero, mz, intensity = utils.get_precursor_mz(dense, peak_scan[j], peak_cycle[j])
            mass_error = (mz - precursor_mz)/mz*10**6


            if np.abs(mass_error) < self.mz_tolerance:

                out_dict = {'index':i}
                
                # the mz column is choosen by the initial parameters and cannot be guaranteed to be present
                # the mz_observed column is the product of this step and will therefore always be present
                if self.has_mz_calibrated:
                    out_dict['mz_calibrated'] = self.precursors_flat.mz_calibrated.values[i]

                if self.has_mz_library:
                    out_dict['mz_library'] = self.precursors_flat.mz_library.values[i]

                frame_center = frame_limits[0,0]+peak_cycle[j]*self.dia_data.cycle.shape[1]
                scan_center = scan_limits[0,0]+peak_scan[j]

                out_dict.update({
                    'mz_observed':mz, 
                    'mass_error':(mz - precursor_mz)/mz*10**6,
                    'fraction_nonzero':fraction_nonzero,
                    'intensity':intensity, 
                    'scan_center':scan_center, 
                    'scan_start':scan_limits[0,0]+limit_scan[0], 
                    #'scan_stop':scan_limits[0,1],
                    'scan_stop':scan_limits[0,0]+limit_scan[1],
                    'frame_center':frame_center, 
                    'frame_start':frame_limits[0,0]+limit_cycle[0]*self.dia_data.cycle.shape[1],
                    'frame_stop':frame_limits[0,0]+limit_cycle[1]*self.dia_data.cycle.shape[1],
                    'rt_library': self.precursors_flat['rt_library'].values[i],
                    'rt_observed': self.dia_data.rt_values[frame_center],
                    'mobility_library': self.precursors_flat['mobility_library'].values[i],
                    'mobility_observed': self.dia_data.mobility_values[scan_center]
                })

                out.append(out_dict)

        if self.debug:
            visualize_candidates(old_profile, profile, peak_scan, peak_cycle, limits_scan, limits_cycle)
       
 
        return out
"""
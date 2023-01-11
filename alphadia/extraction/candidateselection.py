import alphabase.peptide.fragment
import numpy as np
from tqdm import tqdm

import pandas as pd
import logging
from .data import TimsTOFDIA
from . import utils

import matplotlib.pyplot as plt

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
        self.dia_data = dia_data
        self.precursors_flat = precursors_flat

        self.rt_tolerance = rt_tolerance
        self.mobility_tolerance = mobility_tolerance
        self.mz_tolerance = mz_tolerance
        self.num_isotopes = num_isotopes
        self.num_candidates = num_candidates

        self.rt_column = rt_column
        self.precursor_mz_column = precursor_mz_column
        self.mobility_column = mobility_column

        self.has_mz_library = 'mz_library' in self.precursors_flat.columns
        self.has_mz_calibrated = 'mz_calibrated' in self.precursors_flat.columns

        self.k = 6
        self.kernel = utils.kernel_2d_fft(self.k,4)

        self.debug = debug

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
                full_precursor_cycle=True
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

                frame_center = frame_limits[0,0]+peak_cycle[j]*self.dia_data._cycle.shape[1]
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
                    'frame_start':frame_limits[0,0]+limit_cycle[0]*self.dia_data._cycle.shape[1],
                    'frame_stop':frame_limits[0,0]+limit_cycle[1]*self.dia_data._cycle.shape[1],
                    'rt_library': self.precursors_flat['rt_library'].values[i],
                    'rt_observed': self.dia_data.rt_values[frame_center],
                    'mobility_library': self.precursors_flat['mobility_library'].values[i],
                    'mobility_observed': self.dia_data.mobility_values[scan_center]
                })

                out.append(out_dict)

        if self.debug:
            visualize_candidates(old_profile, profile, peak_scan, peak_cycle, limits_scan, limits_cycle)
       
 
        return out

    def __call__(self):
    
        candidates = []
        if not self.debug:
            for i in tqdm(range(len(self.precursors_flat))):
                candidates += self.get_candidates(i)
        else:
            candidates += self.get_candidates(0)
    
        return pd.DataFrame(candidates)

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
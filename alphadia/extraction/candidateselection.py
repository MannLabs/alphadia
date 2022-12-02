import alphabase.peptide.fragment
import numpy as np
from tqdm import tqdm

import pandas as pd

from .data import TimsTOFDIA
from . import utils

class MS1CentricCandidateSelection(object):

    def __init__(self, 
            dia_data,
            precursors_flat, 
            rt_tolerance = 30,
            mobility_tolerance = 0.03,
            mz_tolerance = 120,
            num_isotopes = 2,
            num_candidates = 3,
            rt_column = 'rt_predicted',  
            precursor_mz_column = 'mz_library',
            mobility_column = 'mobility_predicted',
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

        self.kernel = utils.kernel_2d_fft(4,3)

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

        # smooth intensity channel
        new_height = profile.shape[0] - profile.shape[0]%2
        new_width = profile.shape[1] - profile.shape[1]%2
        smooth = self.kernel(profile)
        profile[:smooth.shape[0], :smooth.shape[1]] = smooth

        out = []

        # get all peak candidates
        scan, dia_cycle, intensity = utils.find_peaks(profile, top_n=self.num_candidates)

        for j in range(len(scan)):
            
            mobility_limits, dia_cycle_limits = utils.estimate_peak_boundaries_symmetric(profile, scan[j], dia_cycle[j])
            fraction_nonzero, mz, intensity = utils.get_precursor_mz(dense, scan[j], dia_cycle[j])

            mass_error = (mz - precursor_mz)/mz*10**6


            if np.abs(mass_error) < self.mz_tolerance:

                out_dict = {'index':i}
                
                # the mz column is choosen by the initial parameters and cannot be guaranteed to be present
                # the mz_observed column is the product of this step and will therefore always be present
                if self.has_mz_calibrated:
                    out_dict['mz_calibrated'] = self.precursors_flat.mz_calibrated.values[i]

                if self.has_mz_library:
                    out_dict['mz_library'] = self.precursors_flat.mz_library.values[i]

                out_dict.update({
                    'mz_observed':mz, 
                    'mass_error':(mz - precursor_mz)/mz*10**6,
                    'fraction_nonzero':fraction_nonzero,
                    'intensity':intensity, 
                    'scan_center':scan_limits[0,0]+scan[j], 
                    'scan_start':scan_limits[0,0]+mobility_limits[0], 
                    'scan_stop':scan_limits[0,0]+mobility_limits[1],
                    'frame_center':frame_limits[0,0]+dia_cycle[j]*self.dia_data._cycle.shape[1], 
                    'frame_start':frame_limits[0,0]+dia_cycle_limits[0]*self.dia_data._cycle.shape[1],
                    'frame_stop':frame_limits[0,0]+dia_cycle_limits[1]*self.dia_data._cycle.shape[1],
                })

                out.append(out_dict)

        return out

    def __call__(self):
    
        candidates = []
        for i in tqdm(range(len(self.precursors_flat))):
            candidates += self.get_candidates(i)
        return pd.DataFrame(candidates)
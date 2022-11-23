import alphabase.peptide.fragment
import numpy as np
from tqdm import tqdm

import pandas as pd

from .data import TimsTOFDIA
from . import utils
import logging

class MS2ExtractionWorkflow():
    
    def __init__(self, 
            dia_data,
            precursors_flat, 
            candidates,
            fragments_flat,
            num_precursor_isotopes=3,
            precursor_mass_tolerance=20,
            fragment_mass_tolerance=100,
            include_fragment_info=True
                   
        ):

        self.dia_data = dia_data
        self.precursors_flat = precursors_flat
        self.candidates = candidates

        self.fragments_mz = fragments_flat['mz'].values.copy()
        self.fragments_intensity = fragments_flat['intensity'].values.copy()
        self.fragments_type = np.array(fragments_flat['type'].values.copy(), dtype='U20')

        self.num_precursor_isotopes = num_precursor_isotopes
        self.precursor_mass_tolerance = precursor_mass_tolerance
        self.fragment_mass_tolerance = fragment_mass_tolerance
        self.include_fragment_info = include_fragment_info

        # check if rough calibration is possible
        if 'mass_error' in self.candidates.columns:

            target_indices = np.nonzero(precursors_flat['decoy'].values == 0)[0]
            target_df = candidates[candidates['index'].isin(target_indices)]
            
            correction = np.mean(target_df['mass_error'])
            logging.info(f'rough calibration will be performed {correction:.2f} ppm')

            self.fragments_mz = fragments_flat['mz'].values + fragments_flat['mz'].values/(10**6)*correction


    def __call__(self):

        logging.info(f'performing MS2 extraction for {len(self.candidates):,} candidates')

        features = []
        candidate_iterator = self.candidates.to_dict(orient="records")
        for i, candidate_dict in tqdm(enumerate(candidate_iterator), total=len(candidate_iterator)):
            features += self.get_features(i, candidate_dict)

        features = pd.DataFrame(features)

        logging.info(f'MS2 extraction was able to extract {len(features):,} sets of features for {len(features)/len(self.candidates)*100:.2f}% of candidates')

        

        return features

    def get_features(self, i, candidate_dict):

        

        c_precursor_index = candidate_dict['index']

        c_charge = self.precursors_flat.charge.values[c_precursor_index]

        # observed mz
        c_mz = candidate_dict['mz']

        # theoretical mz
        c_theoretical_mz = candidate_dict['precursor_mz']

        c_frag_start_idx = self.precursors_flat.frag_start_idx.values[c_precursor_index]
        c_frag_end_idx = self.precursors_flat.frag_end_idx.values[c_precursor_index]

        c_fragments_mzs = self.fragments_mz[c_frag_start_idx:c_frag_end_idx]
        c_fragments_order = np.argsort(c_fragments_mzs)

        c_fragments_mzs = c_fragments_mzs[c_fragments_order]
        c_intensity = self.fragments_intensity[c_frag_start_idx:c_frag_end_idx][c_fragments_order]
        c_fragments_type = self.fragments_type[c_frag_start_idx:c_frag_end_idx][c_fragments_order]
        

        fragment_limits = utils.mass_range(c_fragments_mzs, self.fragment_mass_tolerance)
        fragment_tof_limits = utils.make_np_slice(
            self.dia_data.return_tof_indices(
                fragment_limits,
            )
        )

        scan_limits = np.array([[candidate_dict['scan_start'],candidate_dict['scan_stop'],1]])
        frame_limits = np.array([[candidate_dict['frame_start'],candidate_dict['frame_stop'],1]])
    
        quadrupole_limits = np.array([[c_mz,c_mz]])
        dense_fragments = self.dia_data.get_dense(
            frame_limits,
            scan_limits,
            fragment_tof_limits,
            quadrupole_limits
        )

        # calculate fragment values
        fragment_intensity = np.sum(dense_fragments[0], axis=(1,2))
        intensity_mask = np.nonzero(fragment_intensity > 100)[0]

        if len(intensity_mask) < 2:
            return []

        num_isotopes = 3
        
        # get dense precursor
        precursor_isotopes = utils.calc_isotopes_center(
            c_mz,
            c_charge,
            self.num_precursor_isotopes
        )
        precursor_isotope_limits = utils.mass_range(
                precursor_isotopes,
                self.precursor_mass_tolerance
        )
        precursor_tof_limits =utils.make_np_slice(
            self.dia_data.return_tof_indices(
                precursor_isotope_limits,
            )
        )
        dense_precursor = self.dia_data.get_dense(
            frame_limits,
            scan_limits,
            precursor_tof_limits,
            np.array([[-1.,-1.]])
        )


        # ========= assembling general features =========

        frame_center = candidate_dict['frame_center']
        rt_center = self.dia_data.rt_values[frame_center]
        rt_lib = self.precursors_flat.rt.values[c_precursor_index]

        candidate_dict['rt_diff'] = rt_center - rt_lib

        scan_center = candidate_dict['scan_center']
        mobility_center = self.dia_data.mobility_values[scan_center]
        mobility_lib = self.precursors_flat.mobility.values[c_precursor_index]

        candidate_dict['mobility_diff'] = mobility_center - mobility_lib

        # ========= assembling precursor features =========
        theoreticsl_precursor_isotopes = utils.calc_isotopes_center(
            c_theoretical_mz,
            c_charge,
            self.num_precursor_isotopes
        )

        precursor_mass_err, precursor_fraction, precursor_observations = utils.calculate_mass_deviation(
                dense_precursor[1], 
                theoreticsl_precursor_isotopes, 
                size=5
        )

        precursor_intensity = np.sum(dense_precursor[0], axis=(1,2))

        # monoisotopic precursor intensity
        candidate_dict['mono_precursor_intensity'] = precursor_intensity[0]

        # monoisotopic precursor mass error
        candidate_dict['mono_precursor_mass_error'] = precursor_mass_err[0]

        # monoisotopic precursor observations
        candidate_dict['mono_precursor_observations'] = precursor_observations[0]

        # monoisotopic precursor fraction
        candidate_dict['mono_precursor_fraction'] = precursor_fraction[0]

        # highest intensity isotope
        candidate_dict['top_precursor_isotope'] = np.argmax(precursor_intensity, axis=0)

        # highest intensity isotope mass error
        candidate_dict['top_precursor_intensity'] = precursor_intensity[candidate_dict['top_precursor_isotope']]

        # precursor mass error
        candidate_dict['top_precursor_mass_error'] = precursor_mass_err[candidate_dict['top_precursor_isotope']]

        # ========= assembling fragment features =========

        dense_fragments_filtered = dense_fragments[:,intensity_mask]
        fragment_intensity = fragment_intensity[intensity_mask]

        fragment_mass_err, fragment_fraction, fragment_observations = utils.calculate_mass_deviation(
                dense_fragments_filtered[1], 
                c_fragments_mzs[intensity_mask], 
                size=5
        )

        correlations = utils.calculate_correlations(
                np.sum(dense_precursor[0],axis=0), 
                dense_fragments_filtered[0]
        )

       
        precursor_correlations = np.mean(correlations[0:2], axis=0)
        fragment_correlations = np.mean(correlations[2:4], axis=0)

        corr_sum = np.mean(correlations,axis=0)

        # the fragment order is given by the sum of the correlations
        fragment_order = np.argsort(corr_sum)[::-1]

        # number of fragments above intensity threshold
        candidate_dict['num_fragments'] = len(fragment_order)

        # number of fragments with precursor correlation above 0.5
        candidate_dict['num_fragments_pcorr_5'] = np.sum(precursor_correlations[fragment_order] > 0.5)
        candidate_dict['num_fragments_pcorr_3'] = np.sum(precursor_correlations[fragment_order] > 0.3)

        # number of fragments with precursor correlation above 0.5
        candidate_dict['num_fragments_fcorr_3'] = np.sum(fragment_correlations[fragment_order] > 0.3)
        candidate_dict['num_fragments_fcorr_2'] = np.sum(fragment_correlations[fragment_order] > 0.2)
        candidate_dict['num_fragments_fcorr_1'] = np.sum(fragment_correlations[fragment_order] > 0.1)


        # mean precursor correlation for top n fragments
        candidate_dict['mean_pcorr_top_5'] = np.mean(precursor_correlations[fragment_order[0:5]])
        candidate_dict['mean_pcorr_top_10'] = np.mean(precursor_correlations[fragment_order[0:10]])
        candidate_dict['mean_pcorr_top_15'] = np.mean(precursor_correlations[fragment_order[0:15]])

        # mean correlation for top n fragments
        candidate_dict['mean_fcorr_top_5'] = np.mean(fragment_correlations[fragment_order[0:5]])
        candidate_dict['mean_fcorr_top_10'] = np.mean(fragment_correlations[fragment_order[0:10]])
        candidate_dict['mean_fcorr_top_15'] = np.mean(fragment_correlations[fragment_order[0:15]])

        # ========= assembling individual fragment information =========

        if self.include_fragment_info:
            candidate_dict['mass_error_list'] = ';'.join([ f'{el:.3f}' for el in fragment_mass_err[fragment_order][:10]])
            candidate_dict['mass_list'] = ';'.join([ f'{el:.3f}' for el in c_fragments_mzs[fragment_order][:10]])
            candidate_dict['intensity_list'] = ';'.join([ f'{el:.3f}' for el in fragment_intensity[fragment_order][:10]])
            candidate_dict['type_list'] = ';'.join(c_fragments_type[fragment_order][:10])

        return [candidate_dict]
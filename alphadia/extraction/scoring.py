import alphabase.peptide.fragment
import numpy as np
from tqdm import tqdm

import pandas as pd

from .data import TimsTOFDIA
from . import utils
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import numba as nb 


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from alphadia.library import fdr_to_q_values
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score

class MS2ExtractionWorkflow():
    
    def __init__(self, 
            dia_data,
            precursors_flat, 
            candidates,
            fragments_flat,
            num_precursor_isotopes=2,
            precursor_mass_tolerance=20,
            fragment_mass_tolerance=100,
            coarse_mz_calibration=False,
            include_fragment_info=True,
            rt_column = 'rt_library',
            mobility_column = 'mobility_library',
            precursor_mz_column = 'mz_library',
            fragment_mz_column = 'mz_library',
            debug=False
                   
        ):

        self.dia_data = dia_data
        self.precursors_flat = precursors_flat
        self.candidates = candidates


        self.fragments_mz_library = fragments_flat['mz_library'].values.copy()
        self.fragments_mz = fragments_flat[fragment_mz_column].values.copy()
        self.fragments_intensity = fragments_flat['intensity'].values.copy().astype('float64')
        self.fragments_type = fragments_flat['type'].values.copy().astype('int8')

        self.num_precursor_isotopes = num_precursor_isotopes
        self.precursor_mass_tolerance = precursor_mass_tolerance
        self.fragment_mass_tolerance = fragment_mass_tolerance
        self.include_fragment_info = include_fragment_info

        self.rt_column = rt_column
        self.mobility_column = mobility_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column

        self.debug = debug

        # check if rough calibration is possible
        if 'mass_error' in self.candidates.columns and coarse_mz_calibration:

            target_indices = np.nonzero(precursors_flat['decoy'].values == 0)[0]
            target_df = candidates[candidates['index'].isin(target_indices)]
            
            correction = np.mean(target_df['mass_error'])
            logging.info(f'rough calibration will be performed {correction:.2f} ppm')

            self.fragments_mz = fragments_flat[fragment_mz_column].values + fragments_flat[fragment_mz_column].values/(10**6)*correction


    def __call__(self):

        logging.info(f'performing MS2 extraction for {len(self.candidates):,} candidates')

        features = []
        candidate_iterator = self.candidates.to_dict(orient="records")
        if not self.debug:
            for i, candidate_dict in tqdm(enumerate(candidate_iterator), total=len(candidate_iterator)):
                features += self.get_features(i, candidate_dict)
        else:
            for i, candidate_dict in enumerate(candidate_iterator):
                logging.info(f'Only a single candidate is allowed in debugging mode')
                return self.get_features(i, candidate_dict)
      
            
        features = pd.DataFrame(features)
        if len(self.candidates) > 0:
            logging.info(f'MS2 extraction was able to extract {len(features):,} sets of features for {len(features)/len(self.candidates)*100:.2f}% of candidates')
        else:
            logging.warning(f'zero candidates were provided for MS2 extraction')
        

        return features

    def get_features(self, i, candidate_dict):

        c_precursor_index = candidate_dict['index']

        c_charge = self.precursors_flat.charge.values[c_precursor_index]

        c_mz_predicted = candidate_dict['mz_library']

        # observed mz
        c_mz = candidate_dict[self.precursor_mz_column]

        c_frag_start_idx = self.precursors_flat.frag_start_idx.values[c_precursor_index]
        c_frag_stop_idx = self.precursors_flat.frag_stop_idx.values[c_precursor_index]

        c_fragments_mzs = self.fragments_mz[c_frag_start_idx:c_frag_stop_idx]
        #print('fragment_mz', c_fragments_mzs)
        
        c_fragments_order = np.argsort(c_fragments_mzs)
        
        c_fragments_mzs = c_fragments_mzs[c_fragments_order]
        #print('fragment_mz ordered', c_fragments_mzs)
        
        c_intensity = self.fragments_intensity[c_frag_start_idx:c_frag_stop_idx][c_fragments_order]
        c_fragments_type = self.fragments_type[c_frag_start_idx:c_frag_stop_idx][c_fragments_order]

        fragment_limits = utils.mass_range(c_fragments_mzs, self.fragment_mass_tolerance)
        fragment_tof_limits = utils.make_np_slice(
            self.dia_data.return_tof_indices(
                fragment_limits,
            )
        )

        scan_limits = np.array([[candidate_dict['scan_start'],candidate_dict['scan_stop'],1]])
        frame_limits = np.array([[candidate_dict['frame_start'],candidate_dict['frame_stop'],1]])
    
        quadrupole_limits = np.array([[c_mz_predicted,c_mz_predicted]])



        dense_fragments = self.dia_data.get_dense(
            frame_limits,
            scan_limits,
            fragment_tof_limits,
            quadrupole_limits
        )

        #return dense_fragments, (c_fragments_mzs, c_intensity, c_fragments_type)

        # calculate fragment values
        fragment_intensity = np.sum(dense_fragments[0], axis=(1,2))
        intensity_mask = np.nonzero(fragment_intensity > 10)[0]

        dense_fragments = dense_fragments[:,intensity_mask]
        c_fragments_mzs = c_fragments_mzs[intensity_mask]
    
        c_intensity = c_intensity[intensity_mask]
        c_fragments_type = c_fragments_type[intensity_mask]
        fragment_intensity = fragment_intensity[intensity_mask]

        if len(intensity_mask) < 3:
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

        if self.debug:
            
            visualize_dense_fragments(dense_precursor)
            visualize_dense_fragments(dense_fragments)
  
        #print(np.sum(dense_fragments[0],axis=(1,2)))
        #print(np.sum(dense_precursor[0],axis=(1,2)))

        #return {}

        # ========= assembling general features =========

        frame_center = candidate_dict['frame_center']
        rt_center = self.dia_data.rt_values[frame_center]
        rt_lib = self.precursors_flat[self.rt_column].values[c_precursor_index]

        candidate_dict['rt_diff'] = rt_center - rt_lib

        scan_center = candidate_dict['scan_center']
        mobility_center = self.dia_data.mobility_values[scan_center]
        mobility_lib = self.precursors_flat[self.mobility_column].values[c_precursor_index]

        candidate_dict['mobility_diff'] = mobility_center - mobility_lib

        # ========= assembling precursor features =========
        theoreticsl_precursor_isotopes = utils.calc_isotopes_center(
            c_mz,
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


        fragment_mass_err, fragment_fraction, fragment_observations = utils.calculate_mass_deviation(
                dense_fragments[1], 
                c_fragments_mzs, 
                size=5
        )

        precursor_sum = np.sum(dense_precursor[0],axis=0)

        
        correlations = utils.calculate_correlations(
                np.sum(dense_precursor[0],axis=0), 
                dense_fragments[0]
        )

        precursor_correlations = np.mean(correlations[0:2], axis=0)
        fragment_correlations = np.mean(correlations[2:4], axis=0)

        corr_sum = np.mean(correlations,axis=0)

        # the fragment order is given by the sum of the correlations
        fragment_order = np.argsort(corr_sum)[::-1]

        # number of fragments above intensity threshold
        candidate_dict['num_fragments'] = len(fragment_order)

        # number of fragments with precursor correlation above 0.5
        candidate_dict['num_fragments_pcorr_7'] = np.sum(precursor_correlations[fragment_order] > 0.7)
        candidate_dict['num_fragments_pcorr_5'] = np.sum(precursor_correlations[fragment_order] > 0.5)
        candidate_dict['num_fragments_pcorr_3'] = np.sum(precursor_correlations[fragment_order] > 0.3)
        candidate_dict['num_fragments_pcorr_2'] = np.sum(precursor_correlations[fragment_order] > 0.2)
        candidate_dict['num_fragments_pcorr_1'] = np.sum(precursor_correlations[fragment_order] > 0.1)

        # number of fragments with precursor correlation above 0.5
        candidate_dict['num_fragments_fcorr_7'] = np.sum(precursor_correlations[fragment_order] > 0.7)
        candidate_dict['num_fragments_fcorr_5'] = np.sum(precursor_correlations[fragment_order] > 0.5)
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

        #  ======== assembling fragment intensity features ========

        candidate_dict['fragment_intensity_top_5'] = np.mean(fragment_intensity[fragment_order[0:5]])
        candidate_dict['fragment_intensity_top_10'] = np.mean(fragment_intensity[fragment_order[0:10]])
        candidate_dict['fragment_intensity'] = np.mean(fragment_intensity)

        candidate_dict['fragment_rank_score_1d'] = local_rank_score_1d(fragment_intensity, c_intensity)
        candidate_dict['fragment_rank_score_2d'] = local_rank_score_2d(fragment_intensity, c_intensity)

        candidate_dict['fragment_dot_product'] = np.dot(fragment_intensity, c_intensity)
        candidate_dict['fragment_similarity'] = cosine_similarity_float(fragment_intensity, c_intensity)

        fragment_intensity_weighted = calc_fragment_shape(dense_fragments[0, fragment_order])
        candidate_dict['intensity_weighted_top_5'] = np.mean(fragment_intensity_weighted[0:5])
        candidate_dict['intensity_weighted_top_10'] = np.mean(fragment_intensity_weighted[0:10])
        candidate_dict['intensity_weighted'] = np.mean(fragment_intensity_weighted)

        fragment_shape = fragment_intensity_weighted / np.sum(dense_fragments[0], axis = (1,2))
        candidate_dict['fragment_shape_top_5'] = np.mean(fragment_shape[0:5])
        candidate_dict['fragment_shape_top_10'] = np.mean(fragment_shape[0:10])
        candidate_dict['fragment_shape'] = np.mean(fragment_shape)

        # ========= assembling individual fragment information =========

        if self.include_fragment_info:
            mz_library = self.fragments_mz_library[c_frag_start_idx:c_frag_stop_idx]      
            #print('fragment_mz ordered by correlation',c_fragments_mzs[fragment_order])
            #print(mz_library)
            mz_library = mz_library[c_fragments_order]
            #print(mz_library)
            mz_library = mz_library[intensity_mask]
            #print(mz_library)
            mz_library = mz_library[fragment_order]
            #print(mz_library)

            # this will be mz_predicted in the first round and mz_calibrated as soon as calibration has been locked in
            mz_used = c_fragments_mzs[fragment_order]
            mass_error = fragment_mass_err[fragment_order]
    
            mz_observed = mz_used + mass_error * 1e-6 * mz_used

            candidate_dict['fragment_mz_library_list'] = ';'.join([ f'{el:.4f}' for el in mz_library[:15]])
            candidate_dict['fragment_mz_observed_list'] = ';'.join([ f'{el:.4f}' for el in mz_observed[:15]])
            candidate_dict['mass_error_list'] = ';'.join([ f'{el:.3f}' for el in mass_error[:15]])
            #candidate_dict['mass_list'] = ';'.join([ f'{el:.3f}' for el in c_fragments_mzs[fragment_order][:10]])
            candidate_dict['intensity_list'] = ';'.join([ f'{el:.3f}' for el in fragment_intensity[fragment_order][:15]])
            candidate_dict['type_list'] = ';'.join([ f'{el:.3f}' for el in c_fragments_type[fragment_order][:15]])

        return [candidate_dict]
    
def unpack_fragment_info(candidate_scoring_df):

    all_precursor_indices = []
    all_fragment_mz_library = []
    all_fragment_mz_observed = []
    all_mass_errors = []
    all_intensities = []

    for precursor_index, fragment_mz_library_list, fragment_mz_observed_list, mass_error_list, intensity_list in zip(
        candidate_scoring_df['index'].values,
        candidate_scoring_df.fragment_mz_library_list.values, 
        candidate_scoring_df.fragment_mz_observed_list.values, 
        candidate_scoring_df.mass_error_list.values,
        candidate_scoring_df.intensity_list.values
    ):
        fragment_masses = [float(i) for i in fragment_mz_library_list.split(';')]
        all_fragment_mz_library += fragment_masses

        all_fragment_mz_observed += [float(i) for i in fragment_mz_observed_list.split(';')]
        
        all_mass_errors += [float(i) for i in mass_error_list.split(';')]
        all_intensities += [float(i) for i in intensity_list.split(';')]
        all_precursor_indices += [precursor_index] * len(fragment_masses)

    fragment_calibration_df = pd.DataFrame({
        'precursor_index': all_precursor_indices,
        'mz_library': all_fragment_mz_library,
        'mz_observed': all_fragment_mz_observed,
        'mass_error': all_mass_errors,
        'intensity': all_intensities
    })

    return fragment_calibration_df.dropna().reset_index(drop=True)

@nb.njit()
def assign_best_candidate(sorted_index_values):
    best_candidate = -np.ones(np.max(sorted_index_values), dtype=np.int64)
    for i, idx in enumerate(sorted_index_values):
        if best_candidate[idx] == -1:
            best_candidate[idx] = i

    best_candidate = best_candidate[best_candidate >= 0]
    return best_candidate




def fdr_correction(features, 
    feature_columns = ['mz_library', 'mass_error',
       'fraction_nonzero', 'log_intensity', 'rt_diff',
       'mobility_diff', 'log_mono_precursor_intensity',
       'mono_precursor_mass_error', 'mono_precursor_observations',
       'mono_precursor_fraction', 'top_precursor_isotope',
       'log_top_precursor_intensity', 'top_precursor_mass_error', 'num_fragments',
       'num_fragments_pcorr_7',
       'num_fragments_pcorr_5',
       'num_fragments_pcorr_3',
       'num_fragments_pcorr_2',
       'num_fragments_pcorr_1',
       'num_fragments_fcorr_7',
       'num_fragments_fcorr_5',
       'num_fragments_fcorr_3',
       'num_fragments_fcorr_2',
       'num_fragments_fcorr_1', 
       'mean_pcorr_top_5',
       'mean_pcorr_top_10',
       'mean_pcorr_top_15', 'mean_fcorr_top_5', 'mean_fcorr_top_10',
       'mean_fcorr_top_15','fragment_intensity_top_5',
       'fragment_intensity_top_10', 'fragment_intensity',
       'fragment_rank_score_1d', 'fragment_rank_score_2d',
       'fragment_dot_product', 'fragment_similarity',
       'intensity_weighted_top_5', 'intensity_weighted_top_10',
       'intensity_weighted', 'fragment_shape_top_5', 'fragment_shape_top_10',
       'fragment_shape', 'charge', 'nAA'],
       neptune_run=None
    ):
    features = features.dropna().reset_index(drop=True).copy()

    features['log_intensity'] = np.log10(features['intensity'])
    features['log_mono_precursor_intensity'] = np.log10(features['mono_precursor_intensity'])
    features['log_top_precursor_intensity'] = np.log10(features['top_precursor_intensity'])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('GBC', MLPClassifier(hidden_layer_sizes=(50, 25, 5), max_iter=400, alpha=1, learning_rate='adaptive', learning_rate_init=0.001, early_stopping=True, tol=1e-6))
    ])

    X = features[feature_columns].values
    y = features['decoy'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipeline.fit(X_train, y_train)
    

    y_test_proba = pipeline.predict_proba(X_test)[:,1]
    y_test_pred = np.round(y_test_proba)

    y_train_proba = pipeline.predict_proba(X_train)[:,1]
    y_train_pred = np.round(y_train_proba)

    features['proba'] = pipeline.predict_proba(X)[:,1]
    # subset to the best candidate for every precursor
    features = features.sort_values(by=['proba'], ascending=True)
    best_candidates = assign_best_candidate(features['index'].values)
    features_best_df = features.iloc[best_candidates].copy()


    # ROC curve
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    roc_auc_test = auc(fpr_test, tpr_test)

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    roc_auc_train = auc(fpr_train, tpr_train)


    # plotting

    fig, axs = plt.subplots(ncols=3, figsize=(12,3.5))

    axs[0].plot(fpr_test, tpr_test,label="ROC test (area = %0.2f)" % roc_auc_test)
    axs[0].plot(fpr_train, tpr_train,label="ROC train (area = %0.2f)" % roc_auc_train)

    axs[0].plot([0, 1], [0, 1], color="k", linestyle="--")
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel("false positive rate")
    axs[0].set_ylabel("true positive rate")
    axs[0].set_title("ROC Curve")
    axs[0].legend(loc="lower right")
    
    sns.histplot(data=features_best_df, x='proba', hue='decoy', bins=30, element="step", fill=False, ax=axs[1])
    axs[1].set_xlabel('score')
    axs[1].set_ylabel('number of precursors')
    axs[1].set_title("Score Distribution")

    features_best_df = features_best_df.sort_values(['proba'], ascending=True)
    target_values = 1-features_best_df['decoy'].values
    decoy_cumsum = np.cumsum(features_best_df['decoy'].values)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum/target_cumsum
    features_best_df['qval'] = fdr_to_q_values(fdr_values)
    q_val = features_best_df[features_best_df['qval'] <0.05 ]['qval'].values

    ids = np.arange(0, len(q_val), 1)
    axs[2].plot(q_val, ids)
    axs[2].set_xlim(-0.001, 0.05)
    axs[2].set_xlabel('q-value')
    axs[2].set_ylabel('number of precursors')
    axs[2].set_title("Identifications")

    fig.tight_layout()

    if neptune_run is not None:
        neptune_run[f'eval/fdr'].log(fig)
        plt.close()
    else:
        plt.show()

    return features_best_df

@nb.njit
def cosine_similarity_int(a, b):
    div = np.sqrt(np.sum(a))*np.sqrt(np.sum(b))
    if div == 0:
        return 0
    return np.sum((a*b))/div

@nb.njit
def cosine_similarity_float(a, b):
    div = np.linalg.norm(a)*np.linalg.norm(b)
    if div == 0:
        return 0
    return np.dot(a, b)/div

@nb.njit
def local_rank_score_1d(intensity_library, intensity_observed):

    lower_order_library = (intensity_library[:-1] < intensity_library[1:])*1
    upper_order_library = (intensity_library[:-1] > intensity_library[1:])*1

    lower_order_observed = (intensity_observed[:-1] < intensity_observed[1:])*1
    upper_order_observed = (intensity_observed[:-1] > intensity_observed[1:])*1

    lower_score = cosine_similarity_int(lower_order_library, lower_order_observed)
    upper_score = cosine_similarity_int(upper_order_library, upper_order_observed)
    return (lower_score + upper_score)/2

@nb.njit
def local_rank_score_2d(intensity_library, intensity_observed):
    lower_order_library = (np.expand_dims(intensity_library, -1) < np.expand_dims(intensity_library, 0))*1
    upper_order_library = (np.expand_dims(intensity_library, -1) > np.expand_dims(intensity_library, 0))*1

    lower_order_observed = (np.expand_dims(intensity_observed, -1) < np.expand_dims(intensity_observed, 0))*1
    upper_order_observed = (np.expand_dims(intensity_observed, -1) > np.expand_dims(intensity_observed, 0))*1

    lower_score = cosine_similarity_int(lower_order_library.flatten(), lower_order_observed.flatten())
    upper_score = cosine_similarity_int(upper_order_library.flatten(), upper_order_observed.flatten())
    return (lower_score + upper_score)/2

@nb.njit
def _fragment_shape(dense_fragment):

    mean_scan = dense_fragment.shape[0]/2
    mean_cycle = dense_fragment.shape[1]/2

    stack = 0

    for scan in range(dense_fragment.shape[0]):
        for cycle in range(dense_fragment.shape[1]):
            if dense_fragment[scan, cycle] > 0:
                stack += dense_fragment[scan, cycle] / ((scan - mean_scan)**2 + (cycle - mean_cycle)**2 + 1)

    return stack

@nb.njit
def calc_fragment_shape(dense):
    out = np.zeros(dense.shape[0])
    for frag_idx in nb.prange(dense.shape[0]):
        out[frag_idx] = _fragment_shape(dense[frag_idx])

    return out

@nb.njit
def _fragment_center_intensity(dense_fragment):

    mean_scan = dense_fragment.shape[0]/2
    mean_cycle = dense_fragment.shape[1]/2

    lower_scan = np.floor(mean_scan)-2
    upper_scan = np.ceil(mean_scan)+2

    lower_cycle = np.floor(mean_cycle)-2
    upper_cycle = np.ceil(mean_cycle)+2

    return np.sum(dense_fragment[lower_scan:upper_scan, lower_cycle:upper_cycle])


@nb.njit
def fragment_center_intensity(dense):
    out = np.zeros(dense.shape[0])
    for frag_idx in nb.prange(dense.shape[0]):
        out[frag_idx] = _fragment_center_intensity(dense[frag_idx])

    return out



def visualize_dense_fragments(dense):
    intensities = dense[0]
    mass_errors = dense[1]

    n_frags = intensities.shape[0]

    rt_profile = utils.or_envelope(np.sum(intensities,axis=1))
    mobility_profile = utils.or_envelope(np.sum(mass_errors,axis=2))

    ipp = 2
    fig, axs = plt.subplots(3, n_frags, figsize=(n_frags*ipp, 3*ipp))

    for i, ax in enumerate(axs[0]):
        ax.imshow(intensities[i], aspect='equal', cmap='viridis')

    for i, ax in enumerate(axs[1]):
        ax.plot(rt_profile[i])

    for i, ax in enumerate(axs[2]):
        ax.plot(mobility_profile[i])

    fig.tight_layout()
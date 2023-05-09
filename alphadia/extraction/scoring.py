# native imports
import logging
import os

# alphadia imports
from alphadia.extraction import utils
from alphadia.extraction import features, plotting
from alphadia.extraction import validate
from alphadia.library import fdr_to_q_values
from alphadia.extraction.numba import fragments

# alpha family imports
import alphatims

# third party imports
import numpy as np
import pandas as pd
import numba as nb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression




@nb.njit()
def assign_best_candidate(sorted_index_values):
    best_candidate = -np.ones(np.max(sorted_index_values), dtype=np.int64)
    for i, idx in enumerate(sorted_index_values):
        if best_candidate[idx] == -1:
            best_candidate[idx] = i

    best_candidate = best_candidate[best_candidate >= 0]
    return best_candidate

def fdr_correction(features, 
        feature_columns = 
            ['precursor_mass_error',
             'mz_observed',
            'precursor_isotope_correlation', 
            'fraction_fragments', 
            'intensity_correlation',
            'sum_precursor_intensity',
            'sum_fragment_intensity',
            'mean_fragment_intensity',
            'mean_fragment_nonzero',
            'rt_error',
            'rt_observed',
            'mobility_error',
            'mobility_observed',
            'mean_observation_score',
            'var_observation_score',
            'fragment_frame_correlation', 
            'fragment_scan_correlation', 
            'template_frame_correlation', 
            'template_scan_correlation',
            'fwhm_rt',
            'fwhm_mobility',
            'sum_b_ion_intensity',
            'sum_y_ion_intensity',
            'observed_difference_b_y',
            'aggreement_b_y'
            ],
        figure_path = None,
        neptune_run = None,
        index_group = 'elution_group_idx'
    ):
    features = features.dropna().reset_index(drop=True).copy()

    if 'intensity' in features.columns:
        features['log_intensity'] = np.log10(features['intensity'])

    if 'mono_precursor_intensity' in features.columns:
        features['log_mono_precursor_intensity'] = np.log10(features['mono_precursor_intensity'])

    if 'top_precursor_intensity' in features.columns:
        features['log_top_precursor_intensity'] = np.log10(features['top_precursor_intensity'])


    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('GBC', MLPClassifier(hidden_layer_sizes=(50, 25, 5), max_iter=1000, alpha=0.1, learning_rate='adaptive', learning_rate_init=0.001, early_stopping=True, tol=1e-6))
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
    best_candidates = assign_best_candidate(features[index_group].values)
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

    # log figure to neptune ai
    if neptune_run is not None:
        neptune_run[f'eval/fdr'].log(fig)

    if figure_path is not None:
        
        i = 0
        file_name = os.path.join(figure_path, f'fdr_{i}.png')
        while os.path.exists(file_name):
            file_name = os.path.join(figure_path, f'fdr_{i}.png')
            i += 1

        fig.savefig(file_name)
        
    
    else:
        plt.show()  

    plt.close()

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

import numba as nb
from numba.extending import overload_method, overload


@nb.njit()
def concat(array_list):
    length = sum([len(a) for a in array_list])
    out = np.empty(length, dtype=np.float32)
    start = 0
    for a in array_list:
        out[start:start+len(a)] = a
        start += len(a)
    return out

@nb.guvectorize([
    (nb.float64[:], nb.float64[:]),
    (nb.float32[:], nb.float32[:]),
    ], '(n)->(n)')
def or_envelope(x, res):
    res[:] = x
    for i in range(1, len(x) - 1):
        if (x[i] < x[i-1]) or (x[i] < x[i+1]):
            res[i] = (x[i-1] + x[i+1]) / 2


from alphadia.extraction import features
from alphadia.extraction import candidateselection
from alphadia.extraction.plotting import plot_dia_cycle
import numba as nb
from tqdm import tqdm
from matplotlib import patches

from alphadia.extraction.quadrupole import quadrupole_transfer_function, calculate_template, calculate_observation_importance
from alphadia.extraction.plotting import plot_all_precursors

@nb.experimental.jitclass()
class Candidate:

    elution_group_idx: nb.uint32
    precursor_idx: nb.uint32[::1]
    
    scan_start: nb.int64
    scan_stop: nb.int64
    scan_center: nb.int64

    frame_start: nb.int64
    frame_stop: nb.int64
    frame_center: nb.int64

    charge: nb.uint8
    decoy: nb.uint8
    rank: nb.uint8

    frag_start_idx: nb.uint32[::1]
    frag_stop_idx: nb.uint32[::1]
    precursor_mz: nb.float32[::1]

    isotope_intensity: nb.float32[:, ::1]
    isotope_mz: nb.float32[:, ::1]

    scan_limit: nb.uint64[:, ::1]
    frame_limit: nb.uint64[:, ::1]
    fragment_tof_limit : nb.uint64[:, ::1]
    precursor_tof_limit : nb.uint64[:, ::1]
    fragment_quadrupole_limit : nb.float32[:, ::1]

    dense_fragments : nb.float32[:, :, :, :, ::1]
    dense_precursors : nb.float32[:, :, :, :, ::1]
    observation_importance : nb.float64[:, ::1]
    template : nb.float64[:, :, :, ::1]


    fragments: fragments.FragmentContainer.class_type.instance_type

    features: nb.types.DictType(nb.types.unicode_type, nb.float32)
    fragment_features: nb.types.DictType(nb.types.unicode_type, nb.float32[:])

    def __init__(
            self, 
            elution_group_idx,
            precursor_idx,

            scan_start,
            scan_stop,
            scan_center,
            frame_start,
            frame_stop,
            frame_center,

            charge,
            decoy,
            rank,

            frag_start_idx,
            frag_stop_idx,
            precursor_mz,

            isotope_intensity,
        ) -> None:
        
        self.elution_group_idx = elution_group_idx
        self.precursor_idx = precursor_idx
        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.scan_center = scan_center
        self.frame_start = frame_start
        self.frame_stop = frame_stop
        self.frame_center = frame_center

        self.charge = charge
        self.decoy = decoy
        self.rank = rank

        self.frag_start_idx = frag_start_idx
        self.frag_stop_idx = frag_stop_idx
        self.precursor_mz = precursor_mz

        self.isotope_intensity = isotope_intensity

        self.sort_precursor_by_mz()
        self.trim_isotopes()
        self.assemble_isotope_mz()

    def assemble_isotope_mz(self):
        """
        Assemble the isotope m/z values from the precursor m/z and the isotope
        offsets.
        """
        offset = np.arange(self.isotope_intensity.shape[1]) * 1.0033548350700006 / self.charge
        self.isotope_mz = np.expand_dims(self.precursor_mz, 1).astype(np.float32) + np.expand_dims(offset,0).astype(np.float32)

    def sort_precursor_by_mz(self):
        """
        Sort all arrays by m/z
        
        """
        mz_order = np.argsort(self.precursor_mz)
        self.precursor_idx = self.precursor_idx[mz_order]
        self.precursor_mz = self.precursor_mz[mz_order]
        self.frag_start_idx = self.frag_start_idx[mz_order]
        self.frag_stop_idx = self.frag_stop_idx[mz_order]
        self.isotope_intensity = self.isotope_intensity[mz_order, :]

    def trim_isotopes(self):

        elution_group_isotopes = np.sum(self.isotope_intensity, axis=0)
        self.isotope_intensity = self.isotope_intensity[:,elution_group_isotopes>0.01]

    def determine_frame_limits(
            self, 
        ):

        self.frame_limit = np.array(
            [[
                self.frame_start,
                self.frame_stop,
                1
            ]], dtype=np.uint64
        )

    def determine_scan_limits(
            self,
        ):

        self.scan_limit = np.array(
            [[
                self.scan_start,
                self.scan_stop,
                1
            ]],dtype=np.uint64
        )

    def determine_fragment_tof_limit(
            self,
            jit_data,
            fragment_mz_tolerance,
        ):

        mz_limits = utils.mass_range(self.fragments.mz, fragment_mz_tolerance)
        self.fragment_tof_limit = utils.make_slice_2d(
            jit_data.get_tof_indices(
                mz_limits
            )
        )

    def determine_precursor_tof_limit(
            self,
            jit_data,
            precursor_mz_tolerance,
        ):

        
        mz_limits = utils.mass_range(self.isotope_mz.flatten(), precursor_mz_tolerance)
        self.precursor_tof_limit = utils.make_slice_2d(
            jit_data.get_tof_indices(
                mz_limits
            )
        )

    def determine_fragment_quadrupole_limit(
            self,
        ):

        self.fragment_quadrupole_limit = np.array(
            [[
                np.min(self.isotope_mz)-0.5,
                np.max(self.isotope_mz)+0.5
            ]], dtype=np.float32
        )

    def visualize_window(
            self,
            cycle
        ):
        with nb.objmode:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            plot_dia_cycle(cycle, axs[0])
            plot_dia_cycle(cycle, axs[1])

            scan_width = self.scan_stop - self.scan_start
            quad_width = self.fragment_quadrupole_limit[0,1]-self.fragment_quadrupole_limit[0,0]

            axs[0].add_patch(
                patches.Rectangle(
                    (self.fragment_quadrupole_limit[0,0], self.scan_start),
                    quad_width,
                    scan_width,
                    color='blue',
                    alpha=0.5
                )
            )
            axs[1].add_patch(
                patches.Rectangle(
                    (self.fragment_quadrupole_limit[0,0], self.scan_start),
                    quad_width,
                    scan_width,
                    color='blue',
                    alpha=0.5
                )
            )

            axs[1].set_xlim(
                self.fragment_quadrupole_limit[0,0]-quad_width,
                self.fragment_quadrupole_limit[0,1]+quad_width
            )

            axs[1].set_ylim(
                self.scan_stop+scan_width,
                self.scan_start-scan_width
            )
            #ax.set_title(f"Cycle {cycle}")
            plt.show()

    def visualize_fragments(
        self,
        dense_fragments, 
        precursor_index
        ):

        with nb.objmode:

            #v_min = np.min(dense_fragments)
            #v_max = np.max(dense_fragments)

            dpi = 20

            px_width_dense = max(dense_fragments.shape[4],20)
            px_height_dense = dense_fragments.shape[3]

            n_fragments = dense_fragments.shape[1]
            n_cols = n_fragments

            n_observations = dense_fragments.shape[2]
            n_rows = n_observations * 2

            px_width_figure = px_width_dense * n_cols / dpi
            px_height_figure = px_height_dense * n_rows / dpi+1

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(px_width_figure, px_height_figure), sharex=True, sharey=True)

            if len(axs.shape) == 1:
                axs = axs.reshape(1,-1)

            for obs in range(n_observations):
                dense_index = obs * 2
                mass_index = obs * 2+1
                for frag in range(n_fragments):

                    frag_type = chr(self.fragments.type[frag])
                    frag_charge = self.fragments.charge[frag]
                    frag_number = self.fragments.number[frag]

                    axs[dense_index, frag].set_title(f"{frag_type}{frag_number} z{frag_charge}")
                    axs[dense_index, frag].imshow(dense_fragments[0, frag, obs])#, vmin=v_min, vmax=v_max)

                    masked = np.ma.masked_where(dense_fragments[1, frag, obs] == 0, dense_fragments[1, frag, obs])
                    axs[mass_index, frag].imshow(masked, cmap='RdBu')
                    
                    axs[mass_index, frag].set_xlabel(f"frame")
                axs[mass_index, 0].set_ylabel(f"observation {obs}\n scan")
                axs[dense_index, 0].set_ylabel(f"observation {obs}\n scan")

            
            fig.tight_layout()
            plt.show()

    def visualize_precursors(
        self,
        dense_precursors,
        precursor_index
        ):

        with nb.objmode:
            v_min = np.min(dense_precursors)
            v_max = np.max(dense_precursors)

            dpi = 20
            px_width_dense = max(dense_precursors.shape[4],20)
            px_height_dense = dense_precursors.shape[3]

            n_precursors = dense_precursors.shape[1]
            n_cols = n_precursors

            n_observations = dense_precursors.shape[2]
            n_rows = n_observations * 2

            px_width_figure = px_width_dense * n_cols / dpi
            px_height_figure = px_height_dense * n_rows / dpi

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(px_width_figure, px_height_figure))

            if len(axs.shape) == 1:
                axs = axs.reshape(axs.shape[0],1)

            for obs in range(n_observations):
                dense_index = obs * 2
                mass_index = obs * 2+1

                for frag in range(n_precursors):
                    axs[dense_index, frag].imshow(dense_precursors[0, frag, obs], vmin=v_min, vmax=v_max)
                    masked = np.ma.masked_where(dense_precursors[1, frag, obs] == 0, dense_precursors[1, frag, obs])
                    axs[mass_index, frag].imshow(masked, cmap='RdBu')
            
            fig.tight_layout()
            plt.show()


    def visualize_template(
        self,
        dense_precursors,
        qtf,
        template
    ):
        with nb.objmode:
            plot_all_precursors(
                dense_precursors,
                qtf,
                template,
                self.isotope_intensity
            )
        pass

    def add_fixed_features(
        self,
        jit_data
    ):

        self.features['base_width_mobility'] = jit_data.mobility_values[self.scan_start] - jit_data.mobility_values[self.scan_stop-1 ]
        self.features['base_width_rt'] = jit_data.rt_values[self.frame_stop-1] - jit_data.rt_values[self.frame_start]
        self.features['rt_observed'] = jit_data.rt_values[self.frame_center]
        self.features['mobility_observed'] = jit_data.mobility_values[self.scan_center]

    def build_correlation_features(
        self,
        jit_data,
        debug
    ):

        total_fragment_intensity = np.sum(np.sum(self.dense_fragments[0], axis=-1), axis=-1).astype(np.float32)
        total_template_intensity = np.sum(np.sum(self.template, axis=-1), axis=-1)

        fragment_mask_2d = (total_fragment_intensity > 0).astype(np.int8)
        fragment_mask_1d = np.sum(fragment_mask_2d, axis=-1) > 0
        fragment_mask_2d = fragment_mask_2d * np.expand_dims(self.fragments.intensity, axis=-1)

        # (n_fragments, n_observations, n_frames)
        fragments_frame_profile = features.or_envelope_2d(features.frame_profile_2d(self.dense_fragments[0]))
        template_frame_profile = features.or_envelope_2d(features.frame_profile_2d(self.template))

        # (n_fragments, n_observations, n_scans)
        fragments_scan_profile = features.or_envelope_2d(features.scan_profile_2d(self.dense_fragments[0]))
        template_scan_profile = features.or_envelope_2d(features.scan_profile_2d(self.template))

        
        if debug:
            with nb.objmode:
                plotting.plot_fragment_profile(
                    self.template,
                    fragments_scan_profile,
                    fragments_frame_profile,
                    template_frame_profile,
                    template_scan_profile,
                )
        

        # (n_fragments, n_observations)
        fragment_scan_correlation, template_scan_correlation = features.weighted_correlation(
            fragments_scan_profile,
            template_scan_profile,
            fragment_mask_2d,
        )

        
        # (n_fragments, n_observations)
        fragment_frame_correlation, template_frame_correlation = features.weighted_correlation(
            fragments_frame_profile,
            template_frame_profile,
            fragment_mask_2d,
        )

        # calculate retention time FWHM

        # (n_fragments, n_observations)
        cycle_fwhm = np.zeros((
            fragments_frame_profile.shape[0], 
            fragments_frame_profile.shape[1], ),
            dtype=np.float32
        )

        rt_width = jit_data.rt_values[self.frame_stop-1] - jit_data.rt_values[self.frame_start]

        for i_fragment in range(fragments_frame_profile.shape[0]):
            for i_observation in range(fragments_frame_profile.shape[1]):
                max_intensity = np.max(fragments_frame_profile[i_fragment, i_observation])
                half_max = max_intensity / 2
                n_values_above = np.sum(fragments_frame_profile[i_fragment, i_observation] > half_max)
                fraction_above = n_values_above / len(fragments_frame_profile[i_fragment, i_observation])

                cycle_fwhm[i_fragment, i_observation] = fraction_above * rt_width

        
        # calculate mobility FWHM

        # (n_fragments, n_observations)
        mobility_fwhm = np.zeros((
            fragments_scan_profile.shape[0], 
            fragments_scan_profile.shape[1], ),
            dtype=np.float32
        )

        mobility_width = jit_data.mobility_values[self.scan_start] - jit_data.mobility_values[self.scan_stop-1]

        for i_fragment in range(fragments_scan_profile.shape[0]):
            for i_observation in range(fragments_scan_profile.shape[1]):
                max_intensity = np.max(fragments_scan_profile[i_fragment, i_observation])
                half_max = max_intensity / 2
                n_values_above = np.sum(fragments_scan_profile[i_fragment, i_observation] > half_max)
                fraction_above = n_values_above / len(fragments_scan_profile[i_fragment, i_observation])

                mobility_fwhm[i_fragment, i_observation] = fraction_above * mobility_width

        weights = self.fragments.intensity / np.sum(self.fragments.intensity)

        fragment_scan_mean_list = np.sum(fragment_scan_correlation * self.observation_importance, axis = -1)
        fragment_scan_mean_agg = np.sum(fragment_scan_mean_list * weights)
        self.features['fragment_scan_correlation'] = fragment_scan_mean_agg

        fragment_frame_mean_list = np.sum(fragment_frame_correlation * self.observation_importance, axis = -1)
        fragment_frame_mean_agg = np.sum(fragment_frame_mean_list * weights)
        self.features['fragment_frame_correlation'] = fragment_frame_mean_agg

        template_scan_mean_list = np.sum(template_scan_correlation * self.observation_importance, axis = -1)
        template_scan_mean_agg = np.sum(template_scan_mean_list * weights)
        self.features['template_scan_correlation'] = template_scan_mean_agg

        template_frame_mean_list = np.sum(template_frame_correlation * self.observation_importance, axis = -1)
        template_frame_mean_agg = np.sum(template_frame_mean_list * weights)
        self.features['template_frame_correlation'] = template_frame_mean_agg

        

        fragment_cycle_fwhm_mean_list = np.sum(cycle_fwhm * self.observation_importance, axis = -1)
        fragment_cycle_fwhm_mean_agg = np.sum(fragment_cycle_fwhm_mean_list * weights)
        self.features['fwhm_rt'] = fragment_cycle_fwhm_mean_agg

        fragment_scan_fwhm_mean_list = np.sum(mobility_fwhm * self.observation_importance, axis = -1)
        fragment_scan_fwhm_mean_agg = np.sum(fragment_scan_fwhm_mean_list * weights)
        self.features['fwhm_mobility'] = fragment_scan_fwhm_mean_agg

        # calculate features based on b and y ions

        # b = 98
        # y = 121

        b_ion_mask = self.fragments.type == 98
        y_ion_mask = self.fragments.type == 121

        intensity = self.fragments.intensity

        weighted_b_ion_intensity = total_fragment_intensity[b_ion_mask] * self.observation_importance
        if len(weighted_b_ion_intensity) > 0:
            log10_b_ion_intensity = np.log10(np.dot(intensity[b_ion_mask], np.sum(weighted_b_ion_intensity, axis = -1).astype(np.float32)) +0.001)
            expected_b_ion_intensity = np.log10(np.sum(intensity[b_ion_mask]) + 0.001)
        else:
            log10_b_ion_intensity = 0.001
            expected_b_ion_intensity = 0.001

        
        weighted_y_ion_intensity = total_fragment_intensity[y_ion_mask] * self.observation_importance
        if len(weighted_y_ion_intensity) > 0:
            log10_y_ion_intensity = np.log10(np.dot(intensity[y_ion_mask], np.sum(weighted_y_ion_intensity, axis = -1).astype(np.float32)) +0.001)
            expected_y_ion_intensity = np.log10(np.sum(intensity[y_ion_mask]) + 0.001)
        else:
            log10_y_ion_intensity = 0.001
            expected_y_ion_intensity = 0.001

        self.features['sum_b_ion_intensity'] = log10_b_ion_intensity
        self.features['sum_y_ion_intensity'] = log10_y_ion_intensity
        self.features['observed_difference_b_y'] = log10_b_ion_intensity - log10_y_ion_intensity
        self.features['expected_difference_b_y'] = expected_b_ion_intensity - expected_y_ion_intensity
        self.features['aggreement_b_y'] = np.abs(self.features['observed_difference_b_y'] - self.features['expected_difference_b_y'])

        #print(self.features['sum_b_ion_intensity'], self.features['sum_y_ion_intensity'], self.features['observed_difference_b_y'], self.features['expected_difference_b_y'])
  
    def process(
            self, 
            jit_data,
            fragment_container,
            quadrupole_calibration,
            precursor_mz_tolerance,
            fragment_mz_tolerance,
            debug
        ):

        self.features = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.types.float32,
        )

        self.fragment_features = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.float32,
        )

        self.determine_frame_limits()
        self.determine_scan_limits()

        frag_slice = np.stack(
            (
                self.frag_start_idx,
                self.frag_stop_idx,
                np.ones_like(self.frag_start_idx, dtype=np.uint8)
            ),
            axis=1
        )
        self.fragments = fragment_container.slice(frag_slice)
        self.fragments.sort_by_mz()

        #self.assemble_fragment_information(fragment_container)
        #self.determine_fragment_tof_limit(jit_data, fragment_mz_tolerance)
        #self.determine_precursor_tof_limit(jit_data, precursor_mz_tolerance)
        self.determine_fragment_quadrupole_limit()

        
        if debug:
            self.visualize_window(quadrupole_calibration.cycle_calibrated)

        dense_fragments, frag_precursor_index = jit_data.get_dense(
            self.frame_limit,
            self.scan_limit,
            self.fragments.mz,
            fragment_mz_tolerance,
            self.fragment_quadrupole_limit,
            absolute_masses = True
        )

        self.dense_fragments = dense_fragments

        # check if an empty array is returned
        # scan and quadrupole limits of the fragments candidte are outside the acquisition range
        if dense_fragments.shape[-1] == 0:
            return
        
        # only one fragment is found
        if dense_fragments.shape[1] <= 1:
            return
        
        # total intensity of all fragments is too low
        if np.sum(dense_fragments[0]) < 100:
            #print("No fragments found")
            return

        dense_precursors, prec_precursor_index = jit_data.get_dense(
            self.frame_limit,
            self.scan_limit,
            self.isotope_mz.flatten(),
            precursor_mz_tolerance,
            np.array([[-1.,-1.]]),
            absolute_masses = True
        )

        self.dense_precursors = dense_precursors

        

        qtf = quadrupole_transfer_function(
            quadrupole_calibration,
            frag_precursor_index,
            np.arange(self.scan_start, self.scan_stop),
            self.isotope_mz
        )

        template = calculate_template(
            qtf,
            dense_precursors,
            self.isotope_intensity
        )

        if debug:
            self.visualize_template(
                dense_precursors,
                qtf,
                template
            )

        observation_importance = calculate_observation_importance(
            template,

        )
        
        self.observation_importance = observation_importance
        self.template = template

        self.features, self.fragment_features = features.build_features(
            dense_fragments, 
            dense_precursors, 
            template, 
            self.isotope_intensity, 
            self.isotope_mz, 
            self.fragments
        )
        
        coverage = np.sum(qtf, axis = 2)
        self.features['fragment_coverage'] = np.mean(coverage) 
        
        self.build_correlation_features(jit_data, debug)
        
        self.add_fixed_features(jit_data)
        
        if debug:
            
            self.visualize_fragments(dense_fragments, frag_precursor_index)
            
            

            self.visualize_precursors(dense_precursors, prec_precursor_index)

@nb.experimental.jitclass()
class CandidateContainer:
    
        candidates: nb.types.ListType(Candidate.class_type.instance_type)
    
        def __init__(
                self, 
                candidates,
            ) -> None:

            self.candidates = candidates

        def __getitem__(self, idx):
            return self.candidates[idx]

        def __len__(self):
            return len(self.candidates)
        
import alphatims.utils


class MS2ExtractionWorkflow():
    
    def __init__(self, 
            dia_data,
            precursors_flat,
            fragments_flat,
            candidates,
            quadrupole_calibration, 
            num_precursor_isotopes=2,
            precursor_mz_tolerance=20,
            fragment_mz_tolerance=100,
            include_fragment_info=True,
            rt_column = 'rt_library',
            mobility_column = 'mobility_library',
            precursor_mz_column = 'mz_library',
            fragment_mz_column = 'mz_library',
            thread_count=10,
            debug=False
                   
        ):

        self.dia_data = dia_data
        self.precursors_flat = precursors_flat.sort_values(by='precursor_idx')
        self.fragments_flat = fragments_flat
        self.candidates = candidates

        self.num_precursor_isotopes = num_precursor_isotopes
        self.precursor_mz_tolerance = precursor_mz_tolerance
        self.fragment_mz_tolerance = fragment_mz_tolerance
        self.include_fragment_info = include_fragment_info

        self.rt_column = rt_column
        self.mobility_column = mobility_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column

        self.quadrupole_calibration = quadrupole_calibration

        self.debug = debug
        self.thread_count = thread_count

        self.available_isotopes = utils.get_isotope_columns(self.precursors_flat.columns)
        self.available_isotope_columns = [f'i_{i}' for i in self.available_isotopes]

    def __call__(self):

        # if debug mode, only iterate over 10 elution groups
        iterator_len = min(10,len(self.candidates)) if self.debug else len(self.candidates)
        thread_count = 1 if self.debug else self.thread_count

        candidate_container = self.assemble_candidates(iterator_len)
        fragment_container = self.assemble_fragments()

        alphatims.utils.set_threads(thread_count)

        _executor(
            range(iterator_len),
            candidate_container,
            self.dia_data,
            fragment_container,
            self.quadrupole_calibration.jit,
            self.precursor_mz_tolerance,
            self.fragment_mz_tolerance,
            self.debug
        )

        
        feature_df, fragment_df = self.collect_dataframes(candidate_container)
        
        
        feature_df = self.append_precursor_information(feature_df)
        self.log_stats(feature_df)

        if self.debug:
            return candidate_container, feature_df, fragment_df
        
        return feature_df.dropna(), fragment_df.dropna()

    def assemble_candidates(self, n):

        candidates = self.candidates.copy()

        precursor_pidx = self.precursors_flat['precursor_idx'].values
        candidate_pidx = candidates['precursor_idx'].values
        precursor_flat_lookup = np.searchsorted(precursor_pidx, candidate_pidx, side='left')

        if 'flat_frag_start_idx' not in candidates.columns:
            candidates['flat_frag_start_idx'] = self.precursors_flat['flat_frag_start_idx'].values[precursor_flat_lookup]

        if 'flat_frag_stop_idx' not in candidates.columns:
            candidates['flat_frag_stop_idx'] = self.precursors_flat['flat_frag_stop_idx'].values[precursor_flat_lookup]

        candidates['mz'] = self.precursors_flat[self.precursor_mz_column].values[precursor_flat_lookup]

        validate.candidates(candidates)

        for isotope_column in self.available_isotope_columns:
            candidates[isotope_column] = self.precursors_flat[isotope_column].values[precursor_flat_lookup]

        candidate_list = []

        for i in tqdm(range(n)):
            candidate_list.append(self._assemble_candidate(candidates.iloc[[i]]))

        candidate_list = nb.typed.List(
            candidate_list
        )

        return CandidateContainer(candidate_list)


    def _assemble_candidate(self, c):

        # in preparation for multiplexing, multiple precursor are allowed per candidate
        return Candidate(
            c['elution_group_idx'].values[0],
            c['precursor_idx'].values,
            c['scan_start'].values[0],
            c['scan_stop'].values[0],
            c['scan_center'].values[0],
            c['frame_start'].values[0],
            c['frame_stop'].values[0],
            c['frame_center'].values[0],
            c['charge'].values[0],
            c['decoy'].values[0],
            c['rank'].values[0],
            c['flat_frag_start_idx'].values.astype(np.uint32),
            c['flat_frag_stop_idx'].values.astype(np.uint32),
            c['mz'].values.astype(np.float32),
            c[self.available_isotope_columns].values.astype(np.float32),
        )

    def assemble_fragments(self):
            
        # set cardinality to 1 if not present
        if 'cardinality' in self.fragments_flat.columns:
            self.fragments_flat['cardinality'] = self.fragments_flat['cardinality'].values
        
        else:
            logging.warning('Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1.')
            self.fragments_flat['cardinality'] = np.ones(len(self.fragments_flat), dtype=np.uint8)
        
        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.fragments_flat(self.fragments_flat)

        return fragments.FragmentContainer(
            self.fragments_flat['mz_library'].values,
            self.fragments_flat[self.fragment_mz_column].values,
            self.fragments_flat['intensity'].values,
            self.fragments_flat['type'].values,
            self.fragments_flat['loss_type'].values,
            self.fragments_flat['charge'].values,
            self.fragments_flat['number'].values,
            self.fragments_flat['position'].values,
            self.fragments_flat['cardinality'].values
        )

    def _collect_candidate(self, candidate):

        out_dict = {}
        out_dict['precursor_idx'] = candidate.precursor_idx[0]
        out_dict['elution_group_idx'] = candidate.elution_group_idx
        #out_dict['decoy'] = candidate.decoy
        out_dict['charge'] = candidate.charge
        out_dict['rank'] = candidate.rank
        
        out_dict['scan_start'] = candidate.scan_start
        out_dict['scan_stop'] = candidate.scan_stop
        out_dict['scan_center'] = candidate.scan_center
        out_dict['frame_start'] = candidate.frame_start
        out_dict['frame_stop'] = candidate.frame_stop
        out_dict['frame_center'] = candidate.frame_center

        out_dict.update(
            candidate.features
        )
        
        return out_dict

    def collect_dataframes(self, candidate_container):

        fragment_collection = {
            'precursor_idx': [],
            'precursor_idx':[],	
            'mz_library':[],
            'mz_observed':[],
            'mass_error':[],
            'intensity':[],
            'type':[]
        }
        feature_collection = []

        for c in tqdm(candidate_container):

            n = 0
            for key, item in c.fragment_features.items():
                
                fragment_collection[key].append(item)
                n = len(item)
                
            fragment_collection['precursor_idx'].append(np.repeat(c.precursor_idx[0], n))

            feature_collection.append(self._collect_candidate(c))

        for key, item in fragment_collection.items():
            fragment_collection[key] = np.concatenate(item)

        return pd.DataFrame(feature_collection), pd.DataFrame(fragment_collection)

    def append_precursor_information(
        self, 
        df
        ):
        """
        Append relevant precursor information to the candidates dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe containing the extracted candidates

        Returns
        -------
        pandas.DataFrame
            dataframe containing the extracted candidates with precursor information appended
        """

        # precursor_flat_lookup has an element for every candidate and contains the index of the respective precursor
        precursor_pidx = self.precursors_flat['precursor_idx'].values
        candidate_pidx = df['precursor_idx'].values
        precursor_flat_lookup = np.searchsorted(precursor_pidx, candidate_pidx, side='left')

        df['decoy'] = self.precursors_flat['decoy'].values[precursor_flat_lookup]

        df['mz_library'] = self.precursors_flat['mz_library'].values[precursor_flat_lookup]
        if self.precursor_mz_column == 'mz_calibrated':
            df['mz_calibrated'] = self.precursors_flat['mz_calibrated'].values[precursor_flat_lookup]

        df['rt_library'] = self.precursors_flat['rt_library'].values[precursor_flat_lookup]
        if self.rt_column == 'rt_calibrated':
            df['rt_calibrated'] = self.precursors_flat['rt_calibrated'].values[precursor_flat_lookup]
        df['rt_error'] = df['rt_observed'] - df[self.rt_column]

        df['mobility_library'] = self.precursors_flat['mobility_library'].values[precursor_flat_lookup]
        if self.mobility_column == 'mobility_calibrated':
            df['mobility_calibrated'] = self.precursors_flat['mobility_calibrated'].values[precursor_flat_lookup]
        df['mobility_error'] = df['mobility_observed'] - df[self.mobility_column]

        if 'proteins' in self.precursors_flat.columns:
            df['proteins'] = self.precursors_flat['proteins'].values[precursor_flat_lookup]

        if 'channel' in self.precursors_flat.columns:
            df['channel'] = self.precursors_flat['channel'].values[precursor_flat_lookup]

        return df
   

    def log_stats(self, df):
        # log the number of nan rows

        total_rows = df.shape[0]
        logging.info(
            f'Scored {total_rows} candidates'
        )

        decoy_count = np.sum(df['decoy'] == 1)
        decoy_percentage = 100 * df[df['decoy'] == 1].isna().any(axis=1).sum() / decoy_count

        target_count = np.sum(df['decoy'] == 0)
        target_percentage = 100 * df[df['decoy'] == 0].isna().any(axis=1).sum() / target_count

        logging.info(f'{target_percentage:.2f}% of targets failed, {decoy_percentage:.2f}% of decoys failed')

@alphatims.utils.pjit()
def _executor(
        i,
        candidate_container,
        jit_data, 
        fragment_container,
        quadrupole_calibration,
        precursor_mz_tolerance,
        fragment_mz_tolerance,
        debug
    ):
    
    candidate_container[i].process(
        jit_data, 
        fragment_container,
        quadrupole_calibration,
        precursor_mz_tolerance,
        fragment_mz_tolerance,
        debug
    )
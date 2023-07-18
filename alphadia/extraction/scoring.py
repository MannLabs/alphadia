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
        ['base_width_mobility', 'base_width_rt', 'cycle_fwhm',
        'diff_b_y_ion_intensity', 'fragment_frame_correlation',
        'fragment_scan_correlation', 'height_correlation', 'height_fraction',
        'height_fraction_weighted', 'intensity_correlation',
        'intensity_fraction', 'intensity_fraction_weighted',
        'isotope_height_correlation', 'isotope_intensity_correlation',
        'mean_observation_score', 'mobility_fwhm', 'mobility_observed',
        'mono_ms1_height', 'mono_ms1_intensity', 'mz_library', 'mz_observed',
        'n_observations', 'rt_observed', 'sum_b_ion_intensity',
        'sum_ms1_height', 'sum_ms1_intensity', 'sum_y_ion_intensity',
        'template_frame_correlation', 'template_scan_correlation',
        'top3_frame_correlation',
        'top3_scan_correlation', 'top_ms1_height',
        'top_ms1_intensity', 'weighted_mass_deviation', 'weighted_mass_error',
        'weighted_ms1_height', 'weighted_ms1_intensity'],
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

    print(X.shape)
    print(y.shape)

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

def channel_fdr_correction(
    features,
    feature_columns = [ 'reference_intensity_correlation',
                    'mean_reference_scan_cosine',
                    'top3_reference_scan_cosine',
                    'mean_reference_frame_cosine',
                    'top3_reference_frame_cosine',
                    'mean_reference_template_scan_cosine',
                    'mean_reference_template_frame_cosine',
                    'mean_reference_template_frame_cosine_rank',
                    'mean_reference_template_scan_cosine_rank',
                    'mean_reference_frame_cosine_rank',
                    'mean_reference_scan_cosine_rank',
                    'reference_intensity_correlation_rank',
                    'top3_b_ion_correlation_rank',
                    'top3_y_ion_correlation_rank',
                    'top3_frame_correlation_rank',
                    'fragment_frame_correlation_rank',
                    'weighted_ms1_intensity_rank',
                    'isotope_intensity_correlation_rank',
                    'isotope_pattern_correlation_rank',
                    'mono_ms1_intensity_rank',
                    'weighted_mass_error_rank',
                    'base_width_mobility',
                    'base_width_rt',
                    'rt_observed',
                    'mobility_observed',
                    'mono_ms1_intensity',
                    'top_ms1_intensity',
                    'sum_ms1_intensity',
                    'weighted_ms1_intensity',
                    'weighted_mass_deviation',
                    'weighted_mass_error',
                    'mz_library',
                    'mz_observed',
                    'mono_ms1_height',
                    'top_ms1_height',
                    'sum_ms1_height',
                    'weighted_ms1_height',
                    'isotope_intensity_correlation',
                    'isotope_height_correlation',
                    'n_observations',
                    'intensity_correlation',
                    'height_correlation',
                    'intensity_fraction',
                    'height_fraction',
                    'intensity_fraction_weighted',
                    'height_fraction_weighted',
                    'mean_observation_score',
                    'sum_b_ion_intensity',
                    'sum_y_ion_intensity',
                    'diff_b_y_ion_intensity',
                    'fragment_scan_correlation',
                    'top3_scan_correlation',
                    'fragment_frame_correlation',
                    'top3_frame_correlation',
                    'template_scan_correlation',
                    'template_frame_correlation',
                    'cycle_fwhm',
                    'mobility_fwhm'],
    decoy_channel = 12,
    reference_channel = 0,
    target_channels = [4,8],
    ):
    features = features.dropna().reset_index(drop=True).copy()
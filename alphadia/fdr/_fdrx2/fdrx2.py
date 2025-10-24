from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphadia.fdr._fdrx2.lda import train_lda
from alphadia.fdr._fdrx2.utils import bad_features, zero_impact_features

# ruff: noqa


def get_optimal_training_data(
    df_decoy: pd.DataFrame, df_target: pd.DataFrame, available_columns: list[str]
) -> tuple[Any, Any]:
    good_features = [
        f for f in available_columns if f not in zero_impact_features + bad_features
    ]
    full_df = pd.concat([df_target, df_decoy])
    target_features, weights = train_lda(full_df, good_features)

    if "delta_rt" in full_df.columns:
        full_df["delta_rt"] = full_df["delta_rt"] * full_df["delta_rt"]

    (
        best_targets_all,
        decoys_nn,
        passing_decoys,
        passing_elution_groups,
        passing_targets,
        targets_nn,
    ) = _score_and_filter_candidates(full_df, good_features, weights)

    df_target = targets_nn
    df_decoy = decoys_nn
    return df_decoy, df_target


def _score_and_filter_candidates(
    full_df, good_features: list[Any], weights
) -> tuple[Any, Any, set[Any], Any, Any, Any]:
    # Score all
    full_df["lda_score"] = full_df[good_features].values @ weights

    # Get best TARGET and best DECOY per elution group separately using pure numpy
    # Extract numpy arrays for faster operations
    lda_scores = full_df["lda_score"].values
    decoy_flags = full_df["decoy"].values.astype(bool)
    elution_groups = full_df["elution_group_idx"].values

    # Process targets
    target_mask = ~decoy_flags
    target_indices = np.where(target_mask)[0]
    target_scores = lda_scores[target_mask]
    target_eg = elution_groups[target_mask]

    # Sort by elution group for efficient grouping
    sort_idx_t = np.argsort(target_eg)
    sorted_eg_t = target_eg[sort_idx_t]
    sorted_scores_t = target_scores[sort_idx_t]
    sorted_indices_t = target_indices[sort_idx_t]

    # Find best (minimum score) within each group
    _, group_starts_t, group_counts_t = np.unique(
        sorted_eg_t, return_index=True, return_counts=True
    )
    best_within_group_t = np.array(
        [
            start + np.argmin(sorted_scores_t[start : start + count])
            for start, count in zip(group_starts_t, group_counts_t)
        ]
    )
    best_target_indices = sorted_indices_t[best_within_group_t]

    # Process decoys
    decoy_indices = np.where(decoy_flags)[0]
    decoy_scores = lda_scores[decoy_flags]
    decoy_eg = elution_groups[decoy_flags]

    sort_idx_d = np.argsort(decoy_eg)
    sorted_eg_d = decoy_eg[sort_idx_d]
    sorted_scores_d = decoy_scores[sort_idx_d]
    sorted_indices_d = decoy_indices[sort_idx_d]

    _, group_starts_d, group_counts_d = np.unique(
        sorted_eg_d, return_index=True, return_counts=True
    )
    best_within_group_d = np.array(
        [
            start + np.argmin(sorted_scores_d[start : start + count])
            for start, count in zip(group_starts_d, group_counts_d)
        ]
    )
    best_decoy_indices = sorted_indices_d[best_within_group_d]

    # Extract dataframes using iloc for integer indexing
    best_targets_all = full_df.iloc[best_target_indices].reset_index(drop=True)
    best_decoys_all = full_df.iloc[best_decoy_indices].reset_index(drop=True)

    # Calculate LDA q-values (lower LDA score is better!)
    qvalues_lda_targets, qvalues_lda_decoys = _calculate_qvalues_simple(
        best_targets_all["lda_score"].values,
        best_decoys_all["lda_score"].values,
        lower_is_better=True,  # Lower LDA scores are better (targets have low, decoys have high)
    )
    best_targets_all["qvalue"] = qvalues_lda_targets
    best_decoys_all["qvalue"] = qvalues_lda_decoys

    # Filter to 50% FDR
    passing_targets = best_targets_all[best_targets_all["qvalue"] <= 0.5]
    passing_decoys = best_decoys_all[best_decoys_all["qvalue"] <= 0.5]

    # Get the elution groups that have passing targets (for NN training)

    passing_elution_groups = set(passing_targets["elution_group_idx"].values) | set(
        passing_decoys["elution_group_idx"].values
    )

    # Get best targets AND best decoys from passing elution groups
    targets_nn = best_targets_all[
        best_targets_all["elution_group_idx"].isin(passing_elution_groups)
    ].copy()
    decoys_nn = best_decoys_all[
        best_decoys_all["elution_group_idx"].isin(passing_elution_groups)
    ].copy()
    return (
        best_targets_all,
        decoys_nn,
        passing_decoys,
        passing_elution_groups,
        passing_targets,
        targets_nn,
    )


def _calculate_qvalues_simple(target_scores, decoy_scores, lower_is_better=False):
    """Calculate q-values using target-decoy competition for both targets and decoys."""
    all_scores = np.concatenate([target_scores, decoy_scores])
    is_target = np.concatenate(
        [np.ones_like(target_scores), np.zeros_like(decoy_scores)]
    )

    if lower_is_better:
        sorted_idx = np.argsort(all_scores)
    else:
        sorted_idx = np.argsort(-all_scores)

    all_scores = all_scores[sorted_idx]
    is_target = is_target[sorted_idx]

    cum_targets = np.cumsum(is_target)
    cum_decoys = np.cumsum(1 - is_target)
    qvalues_all = cum_decoys / np.maximum(cum_targets, 1)
    qvalues_all = np.minimum.accumulate(qvalues_all[::-1])[::-1]

    # Get q-values for targets
    target_positions = np.where(is_target)[0]
    original_target_idx = sorted_idx[target_positions]
    qvalues_targets = np.zeros(len(target_scores))
    qvalues_targets[original_target_idx] = qvalues_all[target_positions]

    # Get q-values for decoys
    decoy_positions = np.where(is_target == 0)[0]
    original_decoy_idx = sorted_idx[decoy_positions] - len(
        target_scores
    )  # Adjust index
    qvalues_decoys = np.zeros(len(decoy_scores))
    qvalues_decoys[original_decoy_idx] = qvalues_all[decoy_positions]

    return qvalues_targets, qvalues_decoys

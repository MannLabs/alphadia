from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# ruff: noqa


def train_lda(full_df, good_features: list[Any]) -> tuple[Any, Any]:
    # Filter to rank 1 for training (transient)
    rank1_df = full_df[full_df["rank"] == 0].copy()
    rank1_targets = rank1_df[rank1_df["decoy"] == 0]
    rank1_decoys = rank1_df[rank1_df["decoy"] == 1]

    print(
        f"  Rank 1 candidates: {len(rank1_df):,} ({len(rank1_targets):,} targets, {len(rank1_decoys):,} decoys)"
    )

    # Extract features for LDA training
    n_pairs = min(len(rank1_targets), len(rank1_decoys))
    target_features = rank1_targets[good_features].values.astype(np.float64)[:n_pairs]
    decoy_features = rank1_decoys[good_features].values.astype(np.float64)[:n_pairs]

    # Train LDA
    weights, _ = _fit_weights_lda(
        target_features, decoy_features, use_lda=True, verbose=0
    )
    weights = _check_weights(weights, good_features, verbose=0)
    return target_features, weights


def _fit_weights_lda(
    target_features: np.ndarray,
    decoy_features: np.ndarray,
    par_learn: np.ndarray | None = None,
    use_lda: bool = True,
    regularization: float = 1e-9,
    verbose: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit LDA weights to separate targets from decoys.

    This implements the algorithm from diann.cpp fit_weights() function (lines 9281-9423).

    Args:
        target_features: Feature matrix for targets (N_pairs x N_features)
        decoy_features: Feature matrix for decoys (N_pairs x N_features)
        par_learn: Boolean mask indicating which features to use (default: all True)
        use_lda: If True, use LDA formulation; if False, use simpler covariance-based approach
        regularization: Small value added to diagonal for numerical stability
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)

    Returns:
        weights: Learned weight vector (N_features,)
        stats: Dictionary containing training statistics

    Algorithm (from diann.cpp):
    1. Compute mean difference vector: av[i] = mean(target[i] - decoy[i])
    2. Compute class means: mt[i] = mean(target[i]), md[i] = mean(decoy[i])
    3. Build covariance matrix A:
       - LDA: A = 0.5 * (Cov(targets) + Cov(decoys))
       - Non-LDA: A = Cov(target - decoy)
    4. Solve: A * weights = av
    5. Add regularization to diagonal: A[i,i] += epsilon

    """
    n_pairs, n_features = target_features.shape
    assert (
        decoy_features.shape == target_features.shape
    ), "Target and decoy feature matrices must have same shape"

    if par_learn is None:
        par_learn = np.ones(n_features, dtype=bool)

    if verbose >= 1:
        print("\nOptimizing weights...")
        print(f"  Training pairs: {n_pairs}")
        print(f"  Features: {n_features}")
        print(f"  Active features: {par_learn.sum()}")

    # Check minimum training data
    if n_pairs < 20:
        warnings.warn("Too few training precursors (< 20), returning default weights")
        weights = np.zeros(n_features)
        weights[0] = 1.0  # Default: only use first feature
        return weights, {"n_pairs": n_pairs, "converged": False}

    # Step 1 & 2: Compute mean differences and class means (VECTORIZED)
    # We want DECOYS to have HIGHER scores (they are the negative class)
    # So we compute av[i] = mean(decoy[i] - target[i])
    av = np.where(par_learn, np.mean(decoy_features - target_features, axis=0), 0.0)

    if use_lda:
        mt = np.where(par_learn, np.mean(target_features, axis=0), 0.0)
        md = np.where(par_learn, np.mean(decoy_features, axis=0), 0.0)
    else:
        mt = np.zeros(n_features)
        md = np.zeros(n_features)

    if verbose >= 2:
        print(f"\nMean differences (top 10): {av[:10]}")

    # Step 3: Build covariance matrix (VECTORIZED for speed)
    # Lines 9387-9400 in diann.cpp
    if use_lda:
        # LDA formulation: A = 0.5 * (Cov_targets + Cov_decoys)
        # Vectorized: (X - mean).T @ (X - mean)
        target_centered = target_features - mt[np.newaxis, :]
        decoy_centered = decoy_features - md[np.newaxis, :]

        A = 0.5 * (
            target_centered.T @ target_centered + decoy_centered.T @ decoy_centered
        )
    else:
        # Non-LDA: A = Cov(decoy - target)
        diff = decoy_features - target_features
        diff_centered = diff - av[np.newaxis, :]

        A = diff_centered.T @ diff_centered

    # Normalize by n-1 (sample covariance)
    # Lines 9402-9403
    if n_pairs > 1:
        A /= n_pairs - 1

    # Add regularization to diagonal and apply feature mask
    # Line 9404-9406
    np.fill_diagonal(A, A.diagonal() + regularization)

    # Zero out rows/columns for features not in par_learn
    mask = ~par_learn
    A[mask, :] = 0.0
    A[:, mask] = 0.0

    # Step 4: Solve A * weights = av
    # Lines 9415-9420
    try:
        weights = np.linalg.solve(A, av)
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix, using pseudo-inverse")
        weights = np.linalg.lstsq(A, av, rcond=None)[0]

    # Apply mask
    weights = weights * par_learn

    if verbose >= 1:
        print(f"  Weight norm: {np.linalg.norm(weights):.6f}")
        print(f"  Max weight: {np.max(np.abs(weights)):.6f}")
        print(f"  Non-zero weights: {np.sum(np.abs(weights) > 1e-10)}")

    if verbose >= 2:
        print("\nLearned weights:")
        for i, w in enumerate(weights):
            if np.abs(w) > 1e-6:
                print(f"  Feature {i}: {w:.6f}")

    # Compute statistics
    target_scores = target_features @ weights
    decoy_scores = decoy_features @ weights

    stats = {
        "n_pairs": n_pairs,
        "n_features": n_features,
        "n_active_features": par_learn.sum(),
        "weight_norm": np.linalg.norm(weights),
        "target_score_mean": np.mean(target_scores),
        "target_score_std": np.std(target_scores),
        "decoy_score_mean": np.mean(decoy_scores),
        "decoy_score_std": np.std(decoy_scores),
        "separation": np.mean(decoy_scores)
        - np.mean(target_scores),  # Decoy - Target (should be positive)
        "converged": True,
    }

    if verbose >= 1:
        print("\nScoring statistics:")
        print(
            f"  Target score: {stats['target_score_mean']:.6f} ± {stats['target_score_std']:.6f}"
        )
        print(
            f"  Decoy score:  {stats['decoy_score_mean']:.6f} ± {stats['decoy_score_std']:.6f}"
        )
        print(f"  Separation (decoy - target):   {stats['separation']:.6f}")

    return weights, stats


def _check_weights(
    weights: np.ndarray, feature_names: list, verbose: int = 1
) -> np.ndarray:
    """Apply weight constraints (from diann.cpp lines 6593-6601).

    In DIA-NN, certain features like RT deviation and mass accuracy are constrained
    to be non-positive (they act as penalties).

    Args:
        weights: Weight vector
        feature_names: List of feature names
        verbose: Verbosity level

    Returns:
        weights: Constrained weight vector

    """
    weights_original = weights.copy()

    # Identify penalty features that should be non-positive
    penalty_features = ["delta_rt", "weighted_mass_error", "correlation_std"]

    for i, name in enumerate(feature_names):
        if any(penalty in name for penalty in penalty_features):
            if weights[i] > 0.0:
                if verbose >= 2:
                    print(f"  Constraining {name}: {weights[i]:.6f} -> 0.0")
                weights[i] = 0.0

    if verbose >= 1 and not np.allclose(weights, weights_original):
        print(
            f"Applied weight constraints to {np.sum(weights != weights_original)} features"
        )

    return weights

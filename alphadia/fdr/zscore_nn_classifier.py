"""Two-stage classifier: z-score pre-filter on rank 0, then NN on survivors.

Drop-in replacement for BinaryClassifierLegacyNewBatching.
Requires 'rank' to be included in the available_columns passed to perform_fdr.
The rank column is used for pre-filtering but excluded from NN training.
"""

import logging

import numpy as np

from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching, Classifier

try:
    from alphadia_search_rs import zscore_filter_mask as _zscore_filter_mask_rs

    _HAS_RUST_ZSCORE = True
except ImportError:
    _HAS_RUST_ZSCORE = False

logger = logging.getLogger()

ZSCORE_FEATURES = [
    "num_over_0",
    "delta_rt",
    "idf_corr_mass_gaussian",
    "intensity_correlation",
    "idf_hyperscore",
]

ZSCORE_FDR_THRESHOLD = 0.50
_MIN_STD = 1e-10


def _find_score_threshold(
    target_scores: np.ndarray, decoy_scores: np.ndarray, fdr_threshold: float
) -> float:
    """Find lowest score where q-value <= fdr_threshold."""
    all_scores = np.concatenate([target_scores, decoy_scores])
    is_target = np.concatenate(
        [np.ones(len(target_scores)), np.zeros(len(decoy_scores))]
    )

    order = np.argsort(-all_scores)
    is_target_sorted = is_target[order]
    scores_sorted = all_scores[order]

    cum_t = np.cumsum(is_target_sorted)
    cum_d = np.cumsum(1 - is_target_sorted)
    qvals = cum_d / np.maximum(cum_t, 1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]

    passing = np.where(qvals <= fdr_threshold)[0]
    if len(passing) == 0:
        return np.inf
    return scores_sorted[passing[-1]]


class ZScoreNNClassifier(Classifier):
    """Two-stage classifier: z-score pre-filter → NN.

    Stage 1: Z-score 5 features on rank 0 to find 50% FDR threshold.
             Apply threshold to all candidates — survivors proceed to NN.
    Stage 2: NN trained on survivors only (all features except rank).

    The 'rank' column must be included in available_columns so it reaches
    fit() and predict_proba(). It is stripped before NN training/prediction.

    Parameters
    ----------
    zscore_features : list[str]
        Feature names for z-score pre-filter.
    available_columns : list[str]
        All feature column names including 'rank'. Set by perform_fdr.
    zscore_fdr_threshold : float
        FDR threshold for z-score filter.
    **nn_kwargs
        Keyword arguments forwarded to BinaryClassifierLegacyNewBatching.

    """

    def __init__(
        self,
        zscore_features: list[str] | None = None,
        available_columns: list[str] | None = None,
        zscore_fdr_threshold: float = ZSCORE_FDR_THRESHOLD,
        **nn_kwargs,
    ):
        """Initialize the two-stage classifier.

        Parameters
        ----------
        zscore_features : list[str] | None
            Feature names for z-score pre-filter. Defaults to ZSCORE_FEATURES.
        available_columns : list[str] | None
            All feature column names including 'rank'.
        zscore_fdr_threshold : float
            FDR threshold for z-score filter.
        **nn_kwargs
            Keyword arguments forwarded to BinaryClassifierLegacyNewBatching.

        """
        self._zscore_features = zscore_features or ZSCORE_FEATURES
        self._available_columns = available_columns
        self._zscore_fdr_threshold = zscore_fdr_threshold
        self._nn_kwargs = nn_kwargs
        self._nn: BinaryClassifierLegacyNewBatching | None = None
        self._zscore_params: dict | None = None
        self._threshold: float = -np.inf

    @property
    def fitted(self) -> bool:
        """Return whether the classifier has been fitted."""
        return self._nn is not None and self._nn.fitted

    def _resolve_columns(self) -> tuple[int, list[int], list[int]]:
        """Resolve column indices from available_columns."""
        if self._available_columns is None:
            raise ValueError(
                "available_columns must be set before fit/predict. "
                "Pass it via constructor or set_available_columns()."
            )
        col_idx = {c: i for i, c in enumerate(self._available_columns)}
        rank_col = col_idx["rank"]
        zscore_cols = [col_idx[f] for f in self._zscore_features]
        nn_cols = [i for i, c in enumerate(self._available_columns) if c != "rank"]
        return rank_col, zscore_cols, nn_cols

    def set_available_columns(self, columns: list[str]) -> None:
        """Set the available columns (called before fit if not passed to constructor)."""
        self._available_columns = columns

    def _zscore_survivors(self, x: np.ndarray, zscore_cols: list[int]) -> np.ndarray:
        """Compute z-score filter mask using Rust if available, else numpy.

        Parameters
        ----------
        x : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        zscore_cols : list[int]
            Column indices for z-score features.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (n_samples,). True = passes z-score filter.

        """
        p = self._zscore_params
        if _HAS_RUST_ZSCORE:
            return _zscore_filter_mask_rs(
                np.ascontiguousarray(x, dtype=np.float64),
                zscore_cols,
                p["means"].tolist(),
                p["stds"].tolist(),
                p["signs"].tolist(),
                self._threshold,
            )

        feat = np.nan_to_num(
            x[:, zscore_cols].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        scores = np.sum((feat - p["means"]) / p["stds"] * p["signs"], axis=1)
        return scores >= self._threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit z-score threshold on rank 0, then train NN on survivors.

        Parameters
        ----------
        x : np.ndarray
            Training data of shape (n_samples, n_features). Includes rank column.
        y : np.ndarray
            Labels: 0 = target, 1 = decoy.

        """
        rank_col, zscore_cols, nn_cols = self._resolve_columns()

        ranks = x[:, rank_col]
        r0_mask = ranks == 0
        r0_target = r0_mask & (y == 0)
        r0_decoy = r0_mask & (y == 1)

        # Z-score parameters from rank 0
        r0_t_feat = x[r0_target][:, zscore_cols].astype(np.float64)
        r0_d_feat = x[r0_decoy][:, zscore_cols].astype(np.float64)
        r0_t_feat = np.nan_to_num(r0_t_feat, nan=0.0, posinf=0.0, neginf=0.0)
        r0_d_feat = np.nan_to_num(r0_d_feat, nan=0.0, posinf=0.0, neginf=0.0)

        all_r0 = np.vstack([r0_t_feat, r0_d_feat])
        means = np.mean(all_r0, axis=0)
        stds = np.std(all_r0, axis=0)
        stds[stds < _MIN_STD] = 1.0
        signs = np.sign(np.mean(r0_t_feat, axis=0) - np.mean(r0_d_feat, axis=0))
        signs[signs == 0] = 1.0

        self._zscore_params = {"means": means, "stds": stds, "signs": signs}

        # Score rank 0 and find threshold
        r0_t_scores = np.sum((r0_t_feat - means) / stds * signs, axis=1)
        r0_d_scores = np.sum((r0_d_feat - means) / stds * signs, axis=1)
        self._threshold = _find_score_threshold(
            r0_t_scores, r0_d_scores, self._zscore_fdr_threshold
        )

        # Score all candidates and filter (uses Rust if available)
        survivors = self._zscore_survivors(x, zscore_cols)

        logger.info(
            f"Z-score pre-filter: {survivors.sum():,} / {len(x):,} survivors "
            f"(threshold={self._threshold:.4f})"
        )

        # Train NN on survivors only (without rank column)
        x_nn = x[survivors][:, nn_cols]
        y_nn = y[survivors]

        n_nn_features = len(nn_cols)
        self._nn = BinaryClassifierLegacyNewBatching(
            input_dim=n_nn_features,
            **self._nn_kwargs,
        )
        self._nn.fit(x_nn, y_nn)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels. Non-survivors get label 1 (decoy)."""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities. Non-survivors get proba [0, 1] (certain decoy).

        Parameters
        ----------
        x : np.ndarray
            Data of shape (n_samples, n_features). Includes rank column.

        Returns
        -------
        np.ndarray
            Probabilities of shape (n_samples, 2). Column 0 = target, column 1 = decoy.

        """
        rank_col, zscore_cols, nn_cols = self._resolve_columns()

        # Z-score filter (uses Rust if available)
        survivors = self._zscore_survivors(x, zscore_cols)

        # Default: all are decoys
        proba = np.zeros((len(x), 2))
        proba[:, 1] = 1.0

        # NN prediction for survivors
        if survivors.any():
            x_nn = x[survivors][:, nn_cols]
            proba[survivors] = self._nn.predict_proba(x_nn)

        return proba

    def to_state_dict(self) -> dict:
        """Return a state dict of the classifier."""
        return {
            "zscore_params": self._zscore_params,
            "threshold": self._threshold,
            "zscore_features": self._zscore_features,
            "available_columns": self._available_columns,
            "zscore_fdr_threshold": self._zscore_fdr_threshold,
            "nn_kwargs": self._nn_kwargs,
            "nn_state": self._nn.to_state_dict() if self._nn else None,
        }

    def from_state_dict(self, state_dict: dict) -> None:
        """Load a state dict of the classifier."""
        self._zscore_params = state_dict["zscore_params"]
        self._threshold = state_dict["threshold"]
        self._zscore_features = state_dict["zscore_features"]
        self._available_columns = state_dict["available_columns"]
        self._zscore_fdr_threshold = state_dict["zscore_fdr_threshold"]
        self._nn_kwargs = state_dict["nn_kwargs"]
        if state_dict["nn_state"] is not None:
            nn_cols = [c for c in self._available_columns if c != "rank"]
            self._nn = BinaryClassifierLegacyNewBatching(
                input_dim=len(nn_cols),
                **self._nn_kwargs,
            )
            self._nn.from_state_dict(state_dict["nn_state"])

"""Stage-1 prefilter that removes obviously false candidates before the FDR classifier."""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from alphadia.exceptions import TooFewPSMError
from alphadia.fdr.classifiers import LightGBMClassifier
from alphadia.fdr.fdr import get_q_values

logger = logging.getLogger()

# Below this many PSMs the classifier is cheap anyway, and a cross-fitted gate would be
# trained on too few rows to be trusted with the decision which candidates it never sees.
_MIN_PSMS = 100_000


class CascadePrefilter:
    """Gate candidates on a small cross-fitted LightGBM model before the classifier is fitted.

    Most candidates are far from the decision boundary and cost the classifier time
    without changing which PSMs pass the FDR threshold. A small model on a feature subset
    ranks every candidate first; only those below its q-value threshold are passed on.

    The stage-1 scores are produced out-of-fold: every candidate is scored by a model that
    has not seen its label. A model that has seen the labels memorizes false targets as
    targets and decoys as decoys, so it would pass false targets preferentially and break
    the target-decoy symmetry the downstream FDR estimate relies on.
    """

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        feature_columns: list[str],
        classifier: LightGBMClassifier,
        q_value_threshold: float,
        n_folds: int = 2,
        min_psms: int = _MIN_PSMS,
        random_state: int | None = None,
    ):
        """Gate candidates on a small cross-fitted LightGBM model.

        Parameters
        ----------
        feature_columns : list[str]
            Feature columns the stage-1 model is fitted on.

        classifier : LightGBMClassifier
            Unfitted stage-1 model; a copy is fitted per fold.

        q_value_threshold : float
            Candidates whose stage-1 q-value exceeds this are not passed to the classifier.

        n_folds : int, default=2
            Number of cross-fitting folds.

        min_psms : int, default=100000
            Below this many PSMs every candidate is passed on unfiltered.

        random_state : int, optional
            Seed of the fold assignment.

        """
        self.feature_columns = feature_columns
        self.q_value_threshold = q_value_threshold
        self.n_folds = n_folds
        self.min_psms = min_psms
        self._classifier = classifier
        self._np_rng = np.random.default_rng(seed=random_state)

    def select(
        self, psm_df: pd.DataFrame, y: np.ndarray, *, is_final: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decide which candidates are passed on to the classifier.

        Parameters
        ----------
        psm_df : pd.DataFrame
            Candidates, holding `feature_columns` and `precursor_idx`.

        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

        is_final : bool, default=False
            Whether this is the FDR round whose scores are reported.

        Returns
        -------
        keep : np.ndarray, dtype=bool
            True for candidates the classifier should be fitted on and score.

        stage1_proba : np.ndarray, dtype=float
            Out-of-fold stage-1 decoy probability of every candidate; zeros when the
            prefilter did not run.

        """
        n_psms = len(psm_df)
        keep_all = np.ones(n_psms, dtype=bool), np.zeros(n_psms)

        if n_psms < self.min_psms:
            return keep_all

        x = psm_df[self.feature_columns].to_numpy()
        fold = self._np_rng.permutation(n_psms) % self.n_folds
        stage1_proba = np.empty(n_psms)

        try:
            for fold_idx in range(self.n_folds):
                in_fold = fold == fold_idx
                classifier = deepcopy(self._classifier)
                classifier.fit(x[~in_fold], y[~in_fold], is_final=is_final)
                stage1_proba[in_fold] = classifier.predict_proba(x[in_fold])[:, 1]
        except TooFewPSMError:
            logger.warning(
                "Too few PSMs to cross-fit the prefilter, passing all PSMs on"
            )
            return keep_all

        q_values = get_q_values(
            pd.DataFrame(
                {
                    "proba": stage1_proba,
                    "_decoy": y,
                    "precursor_idx": psm_df["precursor_idx"].to_numpy(),
                }
            )
        )["qval"].sort_index()
        keep = (q_values <= self.q_value_threshold).to_numpy()

        logger.info(
            f"Prefilter kept {keep.sum():,} of {n_psms:,} PSMs "
            f"({100 * keep.mean():.1f}%) at stage-1 q-value <= {self.q_value_threshold}: "
            f"{int(((y == 0) & keep).sum()):,} targets, {int(((y == 1) & keep).sum()):,} decoys"
        )

        return keep, stage1_proba

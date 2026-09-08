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

# The gate never passes on fewer PSMs than this. On a low-identification sample the
# stage-1 q-values leave almost everything behind and the classifier would be fitted on a
# few hundred rows (LightGBM aborts outright, finding no feature it can bin). The q-value
# set is a prefix of the stage-1 ranking, so the cut is simply extended down that ranking
# until the floor is reached, which is the same as relaxing the q-value threshold.
_MIN_KEPT_PSMS = 2_000


class CascadePrefilter:
    """Gate candidates on a small cross-fitted LightGBM model before the classifier is fitted.

    Most candidates are far from the decision boundary and cost the classifier time
    without changing which PSMs pass the FDR threshold. A small model on a feature subset
    ranks every candidate first; only those below its q-value threshold are passed on.

    The stage-1 scores are produced out-of-fold: every candidate is scored by a model that
    has not seen its label. A model that has seen the labels memorizes false targets as
    targets and decoys as decoys, so it would pass false targets preferentially and break
    the target-decoy symmetry the downstream FDR estimate relies on.

    Even out-of-fold, a target/decoy model passes false targets more readily than decoys
    (decoys are not perfect copies of false targets), and on a low-input sample the kept
    decoys then count only a fraction of the kept junk: the classifier accepts several false
    targets per accepted decoy however fair it is. The gate therefore drops whole elution
    groups, never single candidates: a target enters stage 2 together with its own decoy.
    Under the null the two are exchangeable and group-level selection is symmetric in them,
    so the kept junk stays balanced whatever the stage-1 model prefers.
    """

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        feature_columns: list[str],
        classifier: LightGBMClassifier,
        q_value_threshold: float,
        n_folds: int = 2,
        min_psms: int = _MIN_PSMS,
        min_kept_psms: int = _MIN_KEPT_PSMS,
        max_train_psms: int | None = None,
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
            The final FDR round widens the cut once when the identifications crowd it.

        n_folds : int, default=2
            Number of cross-fitting folds.

        min_psms : int, default=100000
            Below this many PSMs every candidate is passed on unfiltered.

        min_kept_psms : int, default=2000
            The gate keeps at least this many candidates, extending the cut down the
            stage-1 ranking when the q-value threshold alone would keep fewer.

        max_train_psms : int, optional
            Fit each fold's model on at most this many randomly drawn PSMs of the other
            folds. None fits on all of them.

        random_state : int, optional
            Seed of the fold assignment and the training subsample.

        """
        self.feature_columns = feature_columns
        self.q_value_threshold = q_value_threshold
        self.n_folds = n_folds
        self.min_psms = min_psms
        self.min_kept_psms = min_kept_psms
        self.max_train_psms = max_train_psms
        self._classifier = classifier
        self._np_rng = np.random.default_rng(seed=random_state)

    def select(
        self, psm_df: pd.DataFrame, y: np.ndarray, *, is_final: bool = False
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Decide which candidates are passed on to the classifier.

        Parameters
        ----------
        psm_df : pd.DataFrame
            Candidates, holding `feature_columns`, `precursor_idx` and `elution_group_idx`.

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

        n_passed : int
            Number of candidates that passed the stage-1 cut themselves; they are the
            best-ranked `n_passed` candidates by stage-1 score, the rest of `keep` are
            the other candidates of their elution groups.

        """
        n_psms = len(psm_df)
        keep_all = np.ones(n_psms, dtype=bool), np.zeros(n_psms), n_psms

        if n_psms < self.min_psms:
            return keep_all

        x = psm_df[self.feature_columns].to_numpy()
        fold = self._np_rng.permutation(n_psms) % self.n_folds
        stage1_proba = np.empty(n_psms)

        try:
            for fold_idx in range(self.n_folds):
                in_fold = fold == fold_idx
                train_idx = self._training_rows(np.flatnonzero(~in_fold))
                classifier = deepcopy(self._classifier)
                classifier.fit(x[train_idx], y[train_idx], is_final=is_final)
                stage1_proba[in_fold] = classifier.predict_proba(x[in_fold])[:, 1]
        except TooFewPSMError:
            logger.warning(
                "Too few PSMs to cross-fit the prefilter, passing all PSMs on"
            )
            return keep_all

        keep, n_passed = self.keep_from_scores(
            stage1_proba,
            y,
            psm_df["precursor_idx"].to_numpy(),
            psm_df["elution_group_idx"].to_numpy(),
            self.q_value_threshold,
        )
        return keep, stage1_proba, n_passed

    def keep_from_scores(
        self,
        stage1_proba: np.ndarray,
        y: np.ndarray,
        precursor_idx: np.ndarray,
        elution_group_idx: np.ndarray,
        q_value_threshold: float,
    ) -> tuple[np.ndarray, int]:
        """Cut the stage-1 ranking at a q-value threshold, never below the floor, and keep
        every candidate of the elution groups that reach the cut.

        Parameters
        ----------
        stage1_proba : np.ndarray, dtype=float
            Out-of-fold stage-1 decoy probability of every candidate.

        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

        precursor_idx : np.ndarray
            Precursor index of every candidate, used to break score ties.

        elution_group_idx : np.ndarray
            Elution group of every candidate; a group is kept whole or not at all.

        q_value_threshold : float
            Candidates whose stage-1 q-value exceeds this do not reach the cut.

        Returns
        -------
        keep : np.ndarray, dtype=bool
            True for candidates the classifier should be fitted on and score.

        n_passed : int
            Number of candidates that reached the cut themselves.

        """
        n_psms = len(y)
        q_values = (
            get_q_values(
                pd.DataFrame(
                    {"proba": stage1_proba, "_decoy": y, "precursor_idx": precursor_idx}
                )
            )["qval"]
            .sort_index()
            .to_numpy()
        )
        n_below_threshold = int((q_values <= q_value_threshold).sum())
        n_keep = min(max(n_below_threshold, self.min_kept_psms), n_psms)

        # the same order get_q_values ranks by, so the q-value set is a prefix of it
        order = np.lexsort((precursor_idx, y, stage1_proba))
        passed = np.zeros(n_psms, dtype=bool)
        passed[order[:n_keep]] = True
        keep = np.isin(elution_group_idx, elution_group_idx[passed])

        floor_note = (
            f", raised from {n_below_threshold:,} to the floor of {self.min_kept_psms:,}"
            if n_keep > n_below_threshold
            else ""
        )
        logger.info(
            f"Prefilter passed {n_keep:,} of {n_psms:,} PSMs ({100 * n_keep / n_psms:.1f}%) "
            f"at stage-1 q-value <= {q_value_threshold}{floor_note}, kept their elution "
            f"groups: {int(((y == 0) & keep).sum()):,} targets, "
            f"{int(((y == 1) & keep).sum()):,} decoys"
        )
        return keep, n_keep

    def _training_rows(self, candidates: np.ndarray) -> np.ndarray:
        """Rows one fold's model is fitted on.

        A coarse ranking does not need every row, and the fit cost is linear in them.
        """
        if self.max_train_psms is None or len(candidates) <= self.max_train_psms:
            return candidates
        return self._np_rng.choice(candidates, self.max_train_psms, replace=False)

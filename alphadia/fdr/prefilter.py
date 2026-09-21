"""Stage-1 prefilter that removes obviously false candidates before the FDR classifier."""

import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pandas as pd

from alphadia.exceptions import TooFewPSMError
from alphadia.fdr.classifiers import Classifier
from alphadia.fdr.fdr import get_q_values

logger = logging.getLogger()

# Below this many PSMs the classifier is cheap anyway, and a cross-fitted gate would be
# trained on too few rows to be trusted with the decision which candidates it never sees.
_MIN_PSMS = 100_000

# Rows one scoring task takes. Bounds the feature slice a task copies, which is several
# GB for a whole fold of a large run, while staying far above the batch size where the
# fixed cost per task would start to show.
_SCORE_CHUNK_PSMS = 1_000_000

# Targets minus decoys below this stage-1 q-value estimate how many true precursors the
# gate sees; logged against the kept set as a readout of how hard the gated problem is.
_PURITY_Q_VALUE = 0.1


class CascadePrefilter:
    """Gate candidates on a small cross-fitted model before the classifier is fitted.

    Most candidates are far from the decision boundary and cost the classifier time
    without changing which PSMs pass the FDR threshold. A small model on a feature subset
    ranks every candidate first; those below its q-value threshold are passed on, together
    with a random sample of the rest.

    The stage-1 scores are produced out-of-fold: every candidate is scored by a model that
    has not seen its label. A model that has seen the labels memorizes false targets as
    targets and decoys as decoys, so it would pass false targets preferentially and break
    the target-decoy symmetry the downstream FDR estimate relies on.
    """

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        feature_columns: list[str],
        classifier: Classifier,
        q_value_threshold: float,
        n_folds: int = 2,
        min_psms: int = _MIN_PSMS,
        far_psms: int = 0,
        max_train_psms: int | None = None,
        random_state: int | None = None,
    ):
        """Gate candidates on a small cross-fitted model.

        Parameters
        ----------
        feature_columns : list[str]
            Feature columns the stage-1 model is fitted on.

        classifier : Classifier
            Unfitted stage-1 model; a copy is fitted per fold.

        q_value_threshold : float
            Candidates whose stage-1 q-value exceeds this are not passed to the classifier.

        n_folds : int, default=2
            Number of cross-fitting folds.

        min_psms : int, default=100000
            Below this many PSMs every candidate is passed on unfiltered.

        far_psms : int, default=0
            Number of randomly drawn candidates above the q-value threshold that are
            passed on together with the kept set.

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
        self.far_psms = far_psms
        self.max_train_psms = max_train_psms
        self._classifier = classifier
        self._np_rng = np.random.default_rng(seed=random_state)

    def select(
        self, psm_df: pd.DataFrame, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decide which candidates are passed on to the classifier.

        Parameters
        ----------
        psm_df : pd.DataFrame
            Candidates, holding `feature_columns` and `precursor_idx`.

        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

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
        fold_rows = [
            np.flatnonzero(fold == fold_idx) for fold_idx in range(self.n_folds)
        ]

        # The folds are fitted one after another because their networks draw the initial
        # weights and the dropout masks from the process-wide torch generator: fitting them
        # at the same time would make the gate depend on thread timing instead of the seed.
        classifiers = []
        try:
            for fold_idx in range(self.n_folds):
                train_idx = self._training_rows(np.flatnonzero(fold != fold_idx))
                classifier = deepcopy(self._classifier)
                classifier.fit(x[train_idx], y[train_idx])
                classifiers.append(classifier)
        except TooFewPSMError:
            logger.warning(
                "Too few PSMs to cross-fit the prefilter, passing all PSMs on"
            )
            return keep_all

        # Scoring a fitted model draws no random numbers and the folds share nothing, so the
        # folds are scored at the same time. Torch releases the GIL while it computes and
        # stops scaling at the two intra-op threads the FDR task is capped to, which leaves
        # the cores this needs.
        def score_chunk(chunk: tuple[int, slice]) -> np.ndarray:
            fold_idx, rows = chunk
            return classifiers[fold_idx].predict_proba(x[fold_rows[fold_idx][rows]])[
                :, 1
            ]

        chunks = [
            (fold_idx, slice(start, start + _SCORE_CHUNK_PSMS))
            for fold_idx in range(self.n_folds)
            for start in range(0, len(fold_rows[fold_idx]), _SCORE_CHUNK_PSMS)
        ]
        with ThreadPoolExecutor(max_workers=self.n_folds) as pool:
            for (fold_idx, rows), proba in zip(
                chunks, pool.map(score_chunk, chunks), strict=True
            ):
                stage1_proba[fold_rows[fold_idx][rows]] = proba

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

        confident = q_values.to_numpy() <= _PURITY_Q_VALUE
        estimated_true = int(
            ((y == 0) & confident).sum() - ((y == 1) & confident).sum()
        )
        hard_negatives_per_true = (int(keep.sum()) - estimated_true) / max(
            estimated_true, 1
        )
        logger.info(
            f"Prefilter kept {keep.sum():,} of {n_psms:,} PSMs "
            f"({100 * keep.mean():.1f}%) at stage-1 q-value <= {self.q_value_threshold}: "
            f"{int(((y == 0) & keep).sum()):,} targets, {int(((y == 1) & keep).sum()):,} decoys; "
            f"{estimated_true:,} estimated true precursors, "
            f"{hard_negatives_per_true:.2f} hard negatives per true"
        )

        # The gate keeps the candidates where targets and decoys are hardest to tell apart,
        # and among those false targets are not exchangeable with decoys. A classifier fitted
        # on the kept set alone learns that asymmetry instead of the true-versus-false
        # boundary when the true precursors are few, as in plasma. A random draw from the
        # dropped bulk, where targets and decoys are exchangeable, keeps it anchored there.
        dropped = np.flatnonzero(~keep)
        far = self._np_rng.choice(
            dropped, min(self.far_psms, len(dropped)), replace=False
        )
        keep[far] = True
        logger.info(f"Prefilter added {len(far):,} random dropped PSMs to the kept set")

        return keep, stage1_proba

    def _training_rows(self, candidates: np.ndarray) -> np.ndarray:
        """Rows one fold's model is fitted on.

        A coarse ranking does not need every row, and the fit cost is linear in them.
        """
        if self.max_train_psms is None or len(candidates) <= self.max_train_psms:
            return candidates
        return self._np_rng.choice(candidates, self.max_train_psms, replace=False)

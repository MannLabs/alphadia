"""Cross-fitted self-training of the FDR classifier.

Every target PSM is labelled a positive example although most of them are false matches,
so the classifier learns from labels that are mostly wrong. Percolator-style self-training
(Käll et al. 2007) corrects for that: after a first fit, only the targets that pass a
strict q-value threshold are kept as positives and the model is refitted on them against
the decoys, for a few rounds.

Refitting on PSMs picked by the model's own scores, and scoring the PSMs the model was
fitted on, both bias the decoy count the FDR estimate rests on. Cross-fitting (Granholm et
al. 2012) removes the bias: every PSM is scored by a model that was fitted on the other
folds only.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphadia.fdr.classifiers import Classifier
from alphadia.fdr.fdr import get_q_values, keep_best

logger = logging.getLogger()

# Below this many positives a refit would fit noise; the previous model is kept instead.
_MIN_POSITIVES = 1_000

_MIN_FOLDS = 2

_COMPETITION_GROUP_COLUMN = "_competition_group"
_ROW_COLUMN = "_row"


@dataclass
class TrainingResult:
    """Scores of every PSM and the training set of the last fit."""

    proba: np.ndarray
    train_idx: np.ndarray
    y_train: np.ndarray


class CrossFittedTrainer:
    """Self-training per fold; every PSM is scored by a model that never saw it.

    Folds are drawn by competition group, so a target and its decoy always share a fold.
    The first fit of a fold takes every target of the other folds as a positive example
    and their decoys as negatives. Each refit keeps only the targets below `train_fdr`,
    estimated against the decoys after target-decoy competition, and fits on them against
    all decoys. The classifier is warm-started from one refit to the next.
    """

    def __init__(
        self,
        n_folds: int = 3,
        train_fdr: float = 0.01,
        n_refits: int = 0,
        min_positives: int = _MIN_POSITIVES,
        random_state: int | None = None,
    ):
        """Self-training per fold; every PSM is scored by a model that never saw it.

        Parameters
        ----------
        n_folds : int, default=3
            Number of folds.

        train_fdr : float, default=0.01
            q-value below which a target is kept as a positive example in a refit.

        n_refits : int, default=0
            Number of refits after the first fit on every target.

        min_positives : int, default=1000
            Below this many positives no refit is made and the previous model is kept.

        random_state : int, optional
            Seed of the fold assignment.

        """
        if n_folds < _MIN_FOLDS:
            raise ValueError(f"n_folds must be at least {_MIN_FOLDS}")

        self.n_folds = n_folds
        self.train_fdr = train_fdr
        self.n_refits = n_refits
        self.min_positives = min_positives
        self._np_rng = np.random.default_rng(seed=random_state)

    def fit_predict(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        y: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        *,
        is_final: bool = False,
    ) -> TrainingResult:
        """Fit one classifier per fold and score each fold with the others' model.

        Parameters
        ----------
        classifier : Classifier
            Unfitted classifier; it ends up fitted on the training rows of the last fold.

        x : np.ndarray, dtype=float
            Features of shape (n_samples, n_features).

        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

        competition_group : np.ndarray
            Group of every PSM in the target-decoy competition.

        precursor_idx : np.ndarray
            Precursor index of every PSM, used to break score ties.

        is_final : bool, default=False
            Passed on to the classifier's fit.

        Returns
        -------
        TrainingResult
            Decoy probability of every PSM, plus the rows and labels of the last fit.

        """
        unique_groups, inverse = np.unique(competition_group, return_inverse=True)
        fold = self._np_rng.permutation(len(unique_groups))[inverse] % self.n_folds
        is_target = y == 0

        logger.info(
            f"Cross-fitted fit on {len(y):,} PSMs in {self.n_folds} folds "
            f"({int(is_target.sum()):,} targets, {int((~is_target).sum()):,} decoys)"
        )

        proba = np.empty(len(y))
        for fold_idx in range(self.n_folds):
            in_fold = fold == fold_idx
            train_idx = np.flatnonzero(~in_fold)
            fold_classifier = (
                classifier if fold_idx == self.n_folds - 1 else deepcopy(classifier)
            )
            result = self._self_train(
                fold_classifier,
                x[train_idx],
                is_target[train_idx],
                competition_group[train_idx],
                precursor_idx[train_idx],
                is_final=is_final,
            )
            proba[in_fold] = fold_classifier.predict_proba(x[in_fold])[:, 1]

        return TrainingResult(
            proba=proba,
            train_idx=train_idx[result.train_idx],
            y_train=result.y_train,
        )

    def _self_train(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        is_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        *,
        is_final: bool,
    ) -> TrainingResult:
        """Fit by self-training on the confident targets against all decoys."""
        y_fit = (~is_target).astype(float)
        positives = is_target

        for refit in range(self.n_refits + 1):
            train_idx = np.flatnonzero(positives | ~is_target)
            classifier.fit(x[train_idx], y_fit[train_idx], is_final=is_final)
            proba = classifier.predict_proba(x)[:, 1]

            if refit == self.n_refits:
                break

            new_positives = self._select_positives(
                proba, is_target, competition_group, precursor_idx
            )
            n_positives = int(new_positives.sum())
            if n_positives < self.min_positives:
                logger.warning(
                    f"Only {n_positives:,} targets below train_fdr {self.train_fdr}; "
                    f"keeping the model after {refit} refit(s)"
                )
                break

            n_changed = int((new_positives != positives).sum())
            logger.info(
                f"Refit {refit + 1}: {n_positives:,} positives ({n_changed:,} changed)"
            )
            positives = new_positives

        return TrainingResult(
            proba=proba, train_idx=train_idx, y_train=y_fit[train_idx]
        )

    def _select_positives(
        self,
        proba: np.ndarray,
        is_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
    ) -> np.ndarray:
        """Targets that win their competition group below `train_fdr`."""
        df = pd.DataFrame(
            {
                "proba": proba,
                "_decoy": (~is_target).astype(int),
                _COMPETITION_GROUP_COLUMN: competition_group,
                "precursor_idx": precursor_idx,
                _ROW_COLUMN: np.arange(len(proba)),
            }
        )
        df = keep_best(df, group_columns=[_COMPETITION_GROUP_COLUMN])
        df = get_q_values(df)

        positives = np.zeros(len(proba), dtype=bool)
        winners = df[(df["_decoy"] == 0) & (df["qval"] <= self.train_fdr)]
        positives[winners[_ROW_COLUMN].to_numpy()] = True
        return positives

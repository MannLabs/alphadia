"""Classifier training that keeps the decoy count of the FDR estimate honest.

Every target PSM is labelled a positive example although most of them are false
matches, so the classifier learns from labels that are mostly wrong. Percolator-style
self-training fixes that: after a first fit, only the targets that pass a strict
q-value threshold are kept as positives and the model is refitted on them against the
decoys, for a few rounds.

Refitting on PSMs picked by the model's own scores, and scoring the PSMs the model was
fitted on, both bias the decoy count the FDR estimate rests on. Two remedies are
implemented. Cross-fitting (Percolator, Granholm et al. 2012) scores every PSM with a
model that was fitted on the other folds. RESET (Freestone, Noble and Keich, 2024)
splits the decoys instead: one share trains the classifier, the other is hidden among
the targets during training and is the only share counted when the FDR is estimated, so
a false target and a hidden decoy are treated identically at every step whatever the
classifier did to the PSMs it was fitted on.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphadia.fdr.classifiers import Classifier
from alphadia.fdr.fdr import get_q_values, keep_best

logger = logging.getLogger()

# Below this many positives a refit would fit noise; the previous model is kept instead.
_MIN_POSITIVES = 1_000

_COMPETITION_GROUP_COLUMN = "_competition_group"
_ROW_COLUMN = "_row"
_DECOY_WEIGHT_COLUMN = "_decoy_weight"


@dataclass
class TrainingResult:
    """Scores of every PSM and the training set of the last fit."""

    proba: np.ndarray
    train_idx: np.ndarray
    y_train: np.ndarray


class SelfTrainer(ABC):
    """Self-training on confident targets.

    The first fit takes every pseudo-target as a positive example and the training
    decoys as negatives. Each further iteration keeps only the pseudo-targets below
    `train_fdr`, estimated against the training decoys after target-decoy competition,
    and refits on them. The classifier is warm-started from one iteration to the next.
    """

    def __init__(
        self,
        train_fdr: float = 0.01,
        n_iterations: int = 0,
        max_negative_ratio: float | None = None,
        min_positives: int = _MIN_POSITIVES,
        random_state: int | None = None,
    ):
        """Self-training on confident targets.

        Parameters
        ----------
        train_fdr : float, default=0.01
            q-value below which a pseudo-target is kept as a positive example in a refit.

        n_iterations : int, default=0
            Number of refits after the first fit on every pseudo-target.

        max_negative_ratio : float, optional
            Cap on the training decoys per positive in a refit, drawn at random anew
            for every refit. None fits every refit on all training decoys.

        min_positives : int, default=1000
            Below this many positives no refit is made and the previous model is kept.

        random_state : int, optional
            Seed of the decoy split, the fold assignment and the training subsamples.

        """
        self.train_fdr = train_fdr
        self.n_iterations = n_iterations
        self.max_negative_ratio = max_negative_ratio
        self.min_positives = min_positives
        self._np_rng = np.random.default_rng(seed=random_state)

    @abstractmethod
    def prepare(self, y: np.ndarray, precursor_idx: np.ndarray) -> np.ndarray:
        """Weight every PSM adds to the decoy count of the FDR estimate, zero for targets.

        Parameters
        ----------
        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

        precursor_idx : np.ndarray
            Precursor index of every PSM.

        Returns
        -------
        decoy_weight : np.ndarray, dtype=float
            Weight of every PSM in the decoy count.

        """

    @abstractmethod
    def fit_predict(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        y: np.ndarray,
        decoy_weight: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        *,
        is_final: bool = False,
    ) -> TrainingResult:
        """Fit the classifier and score every PSM.

        Parameters
        ----------
        classifier : Classifier
            Unfitted classifier; it ends up fitted.

        x : np.ndarray, dtype=float
            Features of shape (n_samples, n_features).

        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

        decoy_weight : np.ndarray, dtype=float
            The weights `prepare` assigned to these PSMs.

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

    def _self_train(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        pseudo_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        training_decoy_weight: float,
        *,
        is_final: bool,
    ) -> TrainingResult:
        """Fit by self-training on pseudo-targets against the remaining PSMs."""
        y_fit = (~pseudo_target).astype(float)
        positives = pseudo_target

        for iteration in range(self.n_iterations + 1):
            train_idx = self._training_rows(positives, pseudo_target, iteration)
            classifier.fit(x[train_idx], y_fit[train_idx], is_final=is_final)
            proba = classifier.predict_proba(x)[:, 1]

            if iteration == self.n_iterations:
                break

            new_positives = self._select_positives(
                proba,
                pseudo_target,
                competition_group,
                precursor_idx,
                training_decoy_weight,
            )
            n_positives = int(new_positives.sum())
            if n_positives < self.min_positives:
                logger.warning(
                    f"Only {n_positives:,} pseudo-targets below train_fdr "
                    f"{self.train_fdr}; keeping the model of iteration {iteration}"
                )
                break

            n_changed = int((new_positives != positives).sum())
            logger.info(
                f"Iteration {iteration + 1}: {n_positives:,} positives "
                f"({n_changed:,} changed)"
            )
            positives = new_positives

        return TrainingResult(
            proba=proba, train_idx=train_idx, y_train=y_fit[train_idx]
        )

    def _training_rows(
        self, positives: np.ndarray, pseudo_target: np.ndarray, iteration: int
    ) -> np.ndarray:
        """Rows of one fit: the positives and the training decoys, capped in a refit.

        A refit's positives are few against the decoys, which drowns them for a network
        trained on a fixed number of epochs; the first fit sees every PSM regardless.
        """
        negatives = np.flatnonzero(~pseudo_target)
        if iteration > 0 and self.max_negative_ratio is not None:
            n_max = int(self.max_negative_ratio * positives.sum())
            if len(negatives) > n_max:
                negatives = self._np_rng.choice(negatives, n_max, replace=False)
        return np.sort(np.concatenate([np.flatnonzero(positives), negatives]))

    def _select_positives(  # noqa: PLR0913 # Too many arguments
        self,
        proba: np.ndarray,
        pseudo_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        training_decoy_weight: float,
    ) -> np.ndarray:
        """Pseudo-targets that win their competition group below `train_fdr`."""
        df = pd.DataFrame(
            {
                "proba": proba,
                "_decoy": (~pseudo_target).astype(int),
                _DECOY_WEIGHT_COLUMN: np.where(
                    pseudo_target, 0.0, training_decoy_weight
                ),
                _COMPETITION_GROUP_COLUMN: competition_group,
                "precursor_idx": precursor_idx,
                _ROW_COLUMN: np.arange(len(proba)),
            }
        )
        df = keep_best(df, group_columns=[_COMPETITION_GROUP_COLUMN])
        df = get_q_values(df, decoy_weight_column=_DECOY_WEIGHT_COLUMN)

        positives = np.zeros(len(proba), dtype=bool)
        winners = df[(df["_decoy"] == 0) & (df["qval"] <= self.train_fdr)]
        positives[winners[_ROW_COLUMN].to_numpy()] = True
        return positives


class HiddenDecoyTrainer(SelfTrainer):
    """Self-training with a hidden decoy share that alone feeds the FDR estimate.

    The hidden decoys are labelled target during training; the training decoys are the
    negative class and add nothing to the final decoy count.
    """

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        hidden_decoy_fraction: float = 0.5,
        train_fdr: float = 0.01,
        n_iterations: int = 0,
        max_negative_ratio: float | None = None,
        min_positives: int = _MIN_POSITIVES,
        random_state: int | None = None,
    ):
        """Self-training with a hidden decoy share that alone feeds the FDR estimate.

        Parameters
        ----------
        hidden_decoy_fraction : float, default=0.5
            Share of the decoy precursors hidden from the classifier and counted in the
            FDR estimate.

        train_fdr, n_iterations, max_negative_ratio, min_positives, random_state
            See `SelfTrainer`.

        """
        if not 0 < hidden_decoy_fraction < 1:
            raise ValueError("hidden_decoy_fraction must be strictly between 0 and 1")

        super().__init__(
            train_fdr=train_fdr,
            n_iterations=n_iterations,
            max_negative_ratio=max_negative_ratio,
            min_positives=min_positives,
            random_state=random_state,
        )
        self.hidden_decoy_fraction = hidden_decoy_fraction

    @property
    def decoy_weight(self) -> float:
        """Weight of a hidden decoy in the FDR estimate.

        The false targets are estimated by all decoys, of which only the hidden share is
        counted.
        """
        return 1 / self.hidden_decoy_fraction

    @property
    def _training_decoy_weight(self) -> float:
        """Weight of a training decoy in the q-values that pick the positives.

        The pseudo-targets hold the false targets, estimated by all decoys, plus the
        hidden decoys themselves; the training decoys are the remaining share.
        """
        return (1 + self.hidden_decoy_fraction) / (1 - self.hidden_decoy_fraction)

    def prepare(self, y: np.ndarray, precursor_idx: np.ndarray) -> np.ndarray:
        """Hide the decoys of a random share of the decoy precursors.

        All candidates of a decoy precursor share the fate, as a precursor enters the
        target-decoy competition as a whole.
        """
        unique_precursors, inverse = np.unique(precursor_idx, return_inverse=True)
        hidden_precursor = (
            self._np_rng.random(len(unique_precursors)) < self.hidden_decoy_fraction
        )
        is_hidden = (y == 1) & hidden_precursor[inverse]
        return np.where(is_hidden, self.decoy_weight, 0.0)

    def fit_predict(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        y: np.ndarray,
        decoy_weight: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        *,
        is_final: bool = False,
    ) -> TrainingResult:
        """Fit the classifier by self-training and score every PSM.

        See `SelfTrainer.fit_predict`.
        """
        is_hidden = decoy_weight > 0
        pseudo_target = (y == 0) | is_hidden

        logger.info(
            f"Hidden-decoy fit on {int(pseudo_target.sum()):,} pseudo-targets "
            f"({int(is_hidden.sum()):,} hidden decoys) and "
            f"{int((~pseudo_target).sum()):,} training decoys"
        )

        return self._self_train(
            classifier,
            x,
            pseudo_target,
            competition_group,
            precursor_idx,
            self._training_decoy_weight,
            is_final=is_final,
        )


class CrossFittedTrainer(SelfTrainer):
    """Self-training per fold; every PSM is scored by a model that never saw it.

    Folds are drawn by competition group, so a target and its decoy always share a
    fold. Every decoy counts in the FDR estimate.
    """

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        n_folds: int = 3,
        train_fdr: float = 0.01,
        n_iterations: int = 0,
        max_negative_ratio: float | None = None,
        min_positives: int = _MIN_POSITIVES,
        random_state: int | None = None,
    ):
        """Self-training per fold; every PSM is scored by a model that never saw it.

        Parameters
        ----------
        n_folds : int, default=3
            Number of folds.

        train_fdr, n_iterations, max_negative_ratio, min_positives, random_state
            See `SelfTrainer`.

        """
        if n_folds < 2:  # noqa: PLR2004
            raise ValueError("n_folds must be at least 2")

        super().__init__(
            train_fdr=train_fdr,
            n_iterations=n_iterations,
            max_negative_ratio=max_negative_ratio,
            min_positives=min_positives,
            random_state=random_state,
        )
        self.n_folds = n_folds

    def prepare(self, y: np.ndarray, precursor_idx: np.ndarray) -> np.ndarray:  # noqa: ARG002 # part of the interface
        """Count every decoy once."""
        return (y == 1).astype(float)

    def fit_predict(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        y: np.ndarray,
        decoy_weight: np.ndarray,  # noqa: ARG002 # part of the interface
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        *,
        is_final: bool = False,
    ) -> TrainingResult:
        """Fit one classifier per fold and score each fold with the others' model.

        The classifier passed in ends up fitted on the training rows of the last fold.
        See `SelfTrainer.fit_predict`.
        """
        unique_groups, inverse = np.unique(competition_group, return_inverse=True)
        fold = self._np_rng.permutation(len(unique_groups))[inverse] % self.n_folds
        pseudo_target = y == 0

        logger.info(
            f"Cross-fitted fit on {len(y):,} PSMs in {self.n_folds} folds "
            f"({int(pseudo_target.sum()):,} targets, {int((~pseudo_target).sum()):,} decoys)"
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
                pseudo_target[train_idx],
                competition_group[train_idx],
                precursor_idx[train_idx],
                1.0,
                is_final=is_final,
            )
            proba[in_fold] = fold_classifier.predict_proba(x[in_fold])[:, 1]

        return TrainingResult(
            proba=proba,
            train_idx=train_idx[result.train_idx],
            y_train=result.y_train,
        )

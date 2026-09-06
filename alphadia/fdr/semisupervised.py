"""Semi-supervised classifier training on decoys the FDR estimate never sees.

Every target PSM is labelled a positive example although most of them are false
matches, so the classifier learns from labels that are mostly wrong. Percolator-style
self-training fixes that: after a first fit, only the targets that pass a strict
q-value threshold are kept as positives and the model is refitted on them against the
decoys, for a few rounds.

Refitting on PSMs picked by the model's own scores, and scoring the PSMs the model was
fitted on, both bias the decoy count the FDR estimate rests on. RESET (Freestone, Noble
and Keich, 2024) restores it by splitting the decoys: one share trains the classifier,
the other is hidden among the targets during training and is the only share counted when
the FDR is estimated. A false target and a hidden decoy are then treated identically at
every step, so the hidden decoys stay a fair sample of the false targets whatever the
classifier did to the PSMs it was fitted on.
"""

import logging
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


class HiddenDecoyTrainer:
    """Self-training on confident targets, with a hidden decoy share for the FDR estimate.

    The first fit takes every pseudo-target (true targets and hidden decoys) as a
    positive example and the training decoys as negatives. Each further iteration
    keeps only the pseudo-targets below `train_fdr`, estimated against the training
    decoys after target-decoy competition, and refits on them. The classifier is
    warm-started from one iteration to the next.
    """

    def __init__(
        self,
        hidden_decoy_fraction: float = 0.5,
        train_fdr: float = 0.01,
        n_iterations: int = 5,
        min_positives: int = _MIN_POSITIVES,
        random_state: int | None = None,
    ):
        """Self-training on confident targets, with a hidden decoy share for the FDR estimate.

        Parameters
        ----------
        hidden_decoy_fraction : float, default=0.5
            Share of the decoy precursors hidden from the classifier and counted in the
            FDR estimate.

        train_fdr : float, default=0.01
            q-value below which a pseudo-target is kept as a positive example in a refit.

        n_iterations : int, default=5
            Number of refits after the first fit on every pseudo-target.

        min_positives : int, default=1000
            Below this many positives no refit is made and the previous model is kept.

        random_state : int, optional
            Seed of the decoy split.

        """
        if not 0 < hidden_decoy_fraction < 1:
            raise ValueError("hidden_decoy_fraction must be strictly between 0 and 1")

        self.hidden_decoy_fraction = hidden_decoy_fraction
        self.train_fdr = train_fdr
        self.n_iterations = n_iterations
        self.min_positives = min_positives
        self._np_rng = np.random.default_rng(seed=random_state)

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

    def assign_hidden(self, y: np.ndarray, precursor_idx: np.ndarray) -> np.ndarray:
        """Hide the decoys of a random share of the decoy precursors.

        All candidates of a decoy precursor share the fate, as a precursor enters the
        target-decoy competition as a whole.

        Parameters
        ----------
        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

        precursor_idx : np.ndarray
            Precursor index of every PSM.

        Returns
        -------
        is_hidden : np.ndarray, dtype=bool
            True for the PSMs of hidden decoy precursors.

        """
        unique_precursors, inverse = np.unique(precursor_idx, return_inverse=True)
        hidden_precursor = (
            self._np_rng.random(len(unique_precursors)) < self.hidden_decoy_fraction
        )
        return (y == 1) & hidden_precursor[inverse]

    def fit_predict(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        y: np.ndarray,
        is_hidden: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        *,
        is_final: bool = False,
    ) -> TrainingResult:
        """Fit the classifier by self-training and score every PSM.

        Parameters
        ----------
        classifier : Classifier
            Classifier to fit; it is warm-started from one iteration to the next.

        x : np.ndarray, dtype=float
            Features of shape (n_samples, n_features).

        y : np.ndarray, dtype=int
            Decoy labels of shape (n_samples,), 1 for decoys.

        is_hidden : np.ndarray, dtype=bool
            True for the PSMs of hidden decoy precursors.

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
        pseudo_target = (y == 0) | is_hidden
        y_fit = (~pseudo_target).astype(float)
        positives = pseudo_target

        logger.info(
            f"Semi-supervised fit on {int(pseudo_target.sum()):,} pseudo-targets "
            f"({int(is_hidden.sum()):,} hidden decoys) and "
            f"{int((~pseudo_target).sum()):,} training decoys"
        )

        for iteration in range(self.n_iterations + 1):
            train_idx = np.flatnonzero(positives | ~pseudo_target)
            classifier.fit(x[train_idx], y_fit[train_idx], is_final=is_final)
            proba = classifier.predict_proba(x)[:, 1]

            if iteration == self.n_iterations:
                break

            new_positives = self._select_positives(
                proba, pseudo_target, competition_group, precursor_idx
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

    def _select_positives(
        self,
        proba: np.ndarray,
        pseudo_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
    ) -> np.ndarray:
        """Pseudo-targets that win their competition group below `train_fdr`."""
        df = pd.DataFrame(
            {
                "proba": proba,
                "_decoy": (~pseudo_target).astype(int),
                _DECOY_WEIGHT_COLUMN: np.where(
                    pseudo_target, 0.0, self._training_decoy_weight
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

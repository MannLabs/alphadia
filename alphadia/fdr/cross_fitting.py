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

The first fit, on every target, is where the labels are worst, and not every model
survives it: decoys are not perfect copies of false targets (the two separate with an AUC
of about 0.6), and gradient boosted trees fitted on every target learn that difference and
rank false targets above decoys. On a low-input sample, where most targets are false, that
is most of what they learn and the reported FDR is off by an order of magnitude. A neural
network fitted the same way does not pick the difference up. So only the first member of
an ensemble is fitted on every target; it picks the positives the other members start from.

With a stage-1 score at hand (the prefilter's trees) the first fit takes every candidate
target, the targets of the elution groups the stage-1 score lets through, as a positive:
a tenth of the targets or less, and the positives the refits pick are among them. Taking
only the targets the stage-1 score puts below `train_fdr` instead starves the first fit on
a sample with few identifications: on plasma with 2,500 such targets against the decoy
sample the network collapsed to scoring everything a decoy in one fold out of five and
lost a quarter of the identifications; on every candidate target it lost none. The fit
cost is proportional to the rows the network processes, and the decoys far from the
boundary carry no information about it, so only `n_far_decoys` of them enter, drawn at
random and weighted by the inverse of the sampling rate: the risk stays unbiased, the
variance sits where it does not matter. The `n_near_decoys` hardest decoys enter whole:
the most target-like by the stage-1 score for the first fit and by the network's own
score for every refit (hard-negative mining), so the negatives are the decoys the current
model confuses with targets. Both counts are fixed, so the cost does not grow with the
candidates. Every row is scored out of fold; nothing is dropped and nothing is ranked by
the stage-1 score.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphadia.fdr.classifiers import Classifier, EnsembleClassifier
from alphadia.fdr.fdr import get_q_values, keep_best

logger = logging.getLogger()

# Below this many positives a refit would fit noise; the previous model is kept instead.
_MIN_POSITIVES = 1_000

# The folds are drawn at random, so every fold's model should identify about the same
# number of its fold's targets. A fold whose model came out degenerate (an unlucky set of
# start weights, a network that stopped responding to most features) identifies far fewer
# and drags the whole run down by its share; on the cluster such a run ended at two thirds
# of its siblings' identifications with nothing else in the log to show for it. Such a
# fold is refitted from fresh weights.
_MIN_FOLD_IDENTIFICATION_SHARE = 0.5
_MAX_FOLD_REFITS = 2

_MIN_FOLDS = 2

_COMPETITION_GROUP_COLUMN = "_competition_group"
_ROW_COLUMN = "_row"


@dataclass
class TrainingResult:
    """Scores of every PSM and the training set of the last fit."""

    proba: np.ndarray
    train_idx: np.ndarray
    y_train: np.ndarray


def _take(weight: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    return None if weight is None else weight[idx]


class CrossFittedTrainer:
    """Self-training per fold; every PSM is scored by a model that never saw it.

    Folds are drawn by competition group, so a target and its decoy always share a fold.
    The first fit of a fold takes every target of the other folds as a positive example
    and their decoys as negatives. Each refit keeps only the targets below `train_fdr`,
    estimated against the decoys after target-decoy competition, and fits on them against
    all decoys. The classifier is warm-started from one refit to the next. A fold whose
    model identifies far fewer of its targets than the other folds' models do is fitted
    again from fresh weights.
    """

    def __init__(
        self,
        n_folds: int = 3,
        train_fdr: float = 0.01,
        n_refits: int = 0,
        min_positives: int = _MIN_POSITIVES,
        n_near_decoys: int | None = None,
        n_far_decoys: int | None = None,
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

        n_near_decoys : int, optional
            With a stage-1 score, how many of the hardest decoys by the current score
            enter every refit whole. None enters all of them.

        n_far_decoys : int, optional
            With a stage-1 score, how many of the remaining decoys are drawn at random
            into every refit, weighted by the inverse of the sampling rate. None draws
            all of them.

        random_state : int, optional
            Seed of the fold assignment.

        """
        if n_folds < _MIN_FOLDS:
            raise ValueError(f"n_folds must be at least {_MIN_FOLDS}")

        self.n_folds = n_folds
        self.train_fdr = train_fdr
        self.n_refits = n_refits
        self.min_positives = min_positives
        self.n_near_decoys = n_near_decoys
        self.n_far_decoys = n_far_decoys
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
        sample_weight: np.ndarray | None = None,
        class_prior: float | None = None,
        stage1_proba: np.ndarray | None = None,
        candidate: np.ndarray | None = None,
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

        sample_weight : np.ndarray, optional
            Weight of every PSM in the classifier's loss.

        class_prior : float, optional
            Share of the targets that are true matches, passed on to the first fit of
            every fold; the refits see only confident positives.

        stage1_proba : np.ndarray, optional
            Out-of-fold stage-1 decoy probability of every PSM. When given it picks the
            sampled negatives, and the first fit is on the candidate targets only.

        candidate : np.ndarray, optional
            With `stage1_proba`, the PSMs whose targets are the first positives and among
            which the refits re-pick their positives; every PSM that can reach
            `train_fdr` must be one.

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

        # the passed classifier ends up as the last fold's model, the one that is stored
        fold_classifiers = [deepcopy(classifier) for _ in range(self.n_folds - 1)] + [
            classifier
        ]
        proba = np.empty(len(y))

        if stage1_proba is not None:
            positives = is_target & candidate
            weight = self._sampled_decoy_weights(stage1_proba, is_target)
            logger.info(
                f"Stage 1 keeps {int(positives.sum()):,} candidate targets as the first "
                f"positives; "
                f"{int((weight[~is_target] == 1.0).sum()):,} near decoys enter whole, "
                f"{int((weight[~is_target] > 1.0).sum()):,} far decoys at weight "
                f"{weight.max():.1f}"
            )

        def fit_fold(fold_idx: int) -> TrainingResult:
            in_fold = fold == fold_idx
            train_idx = np.flatnonzero(~in_fold)
            if stage1_proba is None:
                result = self._self_train(
                    fold_classifiers[fold_idx],
                    x[train_idx],
                    is_target[train_idx],
                    competition_group[train_idx],
                    precursor_idx[train_idx],
                    is_final=is_final,
                    sample_weight=_take(sample_weight, train_idx),
                    class_prior=class_prior,
                )
            else:
                result = self._refit(
                    fold_classifiers[fold_idx],
                    x[train_idx],
                    is_target[train_idx],
                    competition_group[train_idx],
                    precursor_idx[train_idx],
                    positives[train_idx],
                    weight[train_idx],
                    candidate[train_idx],
                    is_final=is_final,
                )
            proba[in_fold] = fold_classifiers[fold_idx].predict_proba(x[in_fold])[:, 1]
            return TrainingResult(
                proba=result.proba,
                train_idx=train_idx[result.train_idx],
                y_train=result.y_train,
            )

        results = [fit_fold(fold_idx) for fold_idx in range(self.n_folds)]

        for _ in range(_MAX_FOLD_REFITS):
            weak_folds = self._weak_folds(
                proba, is_target, competition_group, precursor_idx, fold
            )
            if len(weak_folds) == 0:
                break
            for fold_idx in weak_folds:
                fold_classifiers[fold_idx].reset()
                results[fold_idx] = fit_fold(fold_idx)

        return TrainingResult(
            proba=proba,
            train_idx=results[-1].train_idx,
            y_train=results[-1].y_train,
        )

    def _weak_folds(
        self,
        proba: np.ndarray,
        is_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        fold: np.ndarray,
    ) -> np.ndarray:
        """Folds whose model identifies far fewer of its targets than the best fold's does."""
        identified = self._select_positives(
            proba, is_target, competition_group, precursor_idx
        )
        per_fold = np.bincount(fold[identified], minlength=self.n_folds)
        counts = ", ".join(f"{n:,}" for n in per_fold)
        logger.info(f"Targets below train_fdr {self.train_fdr} per fold: {counts}")
        weak = np.flatnonzero(
            per_fold < _MIN_FOLD_IDENTIFICATION_SHARE * per_fold.max()
        )
        for fold_idx in weak:
            logger.warning(
                f"Fold {fold_idx} identifies {per_fold[fold_idx]:,} targets below "
                f"train_fdr {self.train_fdr}, the best fold {per_fold.max():,}; "
                f"refitting it from fresh weights"
            )
        return weak

    def _self_train(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        is_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        *,
        is_final: bool,
        sample_weight: np.ndarray | None = None,
        class_prior: float | None = None,
    ) -> TrainingResult:
        """Fit by self-training on the confident targets against all decoys.

        Only the first member of an ensemble is fitted on every target; the others are
        fitted on the positives it picks, see the module docstring.
        """
        y_fit = (~is_target).astype(float)
        members = (
            classifier.members
            if isinstance(classifier, EnsembleClassifier)
            else [classifier]
        )
        teacher, students = members[0], members[1:]

        teacher.fit_separating(
            x,
            y_fit,
            is_final=is_final,
            sample_weight=sample_weight,
            class_prior=class_prior,
        )
        positives = self._select_positives(
            teacher.predict_proba(x)[:, 1], is_target, competition_group, precursor_idx
        )
        n_positives = int(positives.sum())
        logger.info(
            f"First fit: {n_positives:,} of {int(is_target.sum()):,} targets below "
            f"train_fdr {self.train_fdr} are the positives"
        )
        if n_positives < self.min_positives:
            logger.warning(
                f"Only {n_positives:,} positives; fitting the other members on every target"
            )
            positives = is_target

        for refit in range(self.n_refits + 1):
            train_idx = np.flatnonzero(positives | ~is_target)
            # the positives here are the confident targets, not the whole mixture, so the
            # class prior does not apply to these fits
            for member in students if refit == 0 else members:
                member.fit_separating(
                    x[train_idx],
                    y_fit[train_idx],
                    is_final=is_final,
                    sample_weight=_take(sample_weight, train_idx),
                )
            proba = classifier.predict_proba(x)[:, 1]

            if refit == self.n_refits:
                break

            # the teacher alone picks the next positives: the trees rank false targets
            # above decoys by a margin that grows with the number of decoys they see,
            # and once they help pick their own positives the false ones snowball
            new_positives = self._select_positives(
                teacher.predict_proba(x)[:, 1],
                is_target,
                competition_group,
                precursor_idx,
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

    def _sampled_decoy_weights(
        self, score: np.ndarray, is_target: np.ndarray
    ) -> np.ndarray:
        """Weight of every PSM in a refit: 1 for targets and the hardest decoys by
        `score`, the inverse sampling rate for the far decoys drawn, 0 for the rest."""
        weight = np.ones(len(score))
        decoy_idx = np.flatnonzero(~is_target)
        if self.n_near_decoys is None or len(decoy_idx) <= self.n_near_decoys:
            return weight
        far = decoy_idx[np.argsort(score[decoy_idx])][self.n_near_decoys :]
        if self.n_far_decoys is None or len(far) <= self.n_far_decoys:
            return weight
        rate = self.n_far_decoys / len(far)
        drawn = self._np_rng.random(len(far)) < rate
        weight[far] = 0.0
        weight[far[drawn]] = 1.0 / rate
        return weight

    def _refit(  # noqa: PLR0913 # Too many arguments
        self,
        classifier: Classifier,
        x: np.ndarray,
        is_target: np.ndarray,
        competition_group: np.ndarray,
        precursor_idx: np.ndarray,
        positives: np.ndarray,
        weight: np.ndarray,
        candidate: np.ndarray,
        *,
        is_final: bool,
    ) -> TrainingResult:
        """Fit on the positives against the sampled decoys, re-picking the positives
        among the candidates with the classifier's own score before each refit."""
        y_fit = (~is_target).astype(float)
        candidate_idx = np.flatnonzero(candidate)
        decoy_idx = np.flatnonzero(~is_target)
        for refit in range(self.n_refits + 1):
            train_idx = np.flatnonzero(positives | (~is_target & (weight > 0)))
            classifier.fit_separating(
                x[train_idx],
                y_fit[train_idx],
                is_final=is_final,
                sample_weight=weight[train_idx],
            )
            if refit == self.n_refits:
                break

            new_positives = np.zeros(len(x), dtype=bool)
            new_positives[candidate_idx] = self._select_positives(
                classifier.predict_proba(x[candidate_idx])[:, 1],
                is_target[candidate_idx],
                competition_group[candidate_idx],
                precursor_idx[candidate_idx],
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
            decoy_score = np.ones(len(x))
            decoy_score[decoy_idx] = classifier.predict_proba(x[decoy_idx])[:, 1]
            weight = self._sampled_decoy_weights(decoy_score, is_target)

        # the candidates' scores are what the refits rest on; the rest is scored out of
        # fold by the caller
        proba = np.ones(len(x))
        proba[candidate_idx] = classifier.predict_proba(x[candidate_idx])[:, 1]
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
        df = keep_best(
            df, group_columns=[_COMPETITION_GROUP_COLUMN], decoy_column="_decoy"
        )
        df = get_q_values(df)

        positives = np.zeros(len(proba), dtype=bool)
        winners = df[(df["_decoy"] == 0) & (df["qval"] <= self.train_fdr)]
        positives[winners[_ROW_COLUMN].to_numpy()] = True
        return positives

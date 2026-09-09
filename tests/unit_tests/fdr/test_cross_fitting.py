import numpy as np
import pandas as pd
import pytest

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import EnsembleClassifier, LightGBMClassifier
from alphadia.fdr.cross_fitting import CrossFittedTrainer

N_CANDIDATES = 3


def _gen_psms(n_precursors: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Target precursors of which a third are true; every false candidate looks like a decoy.

    Each precursor has three candidates; a true precursor has one true candidate. The
    single informative feature separates true candidates from the rest.
    """
    rng = np.random.default_rng(seed)
    rows = []
    precursor_idx = 0
    for group in range(n_precursors):
        is_true = group % 3 == 0
        for decoy in (0, 1):
            for rank in range(N_CANDIDATES):
                true_candidate = is_true and decoy == 0 and rank == 0
                feature = rng.normal(2.0 if true_candidate else 0.0, 0.5)
                rows.append(
                    {
                        "precursor_idx": precursor_idx,
                        "elution_group_idx": group,
                        "channel": 0,
                        "decoy": decoy,
                        "rank": rank,
                        "feature": feature,
                        "noise": rng.normal(),
                        "is_true": true_candidate,
                    }
                )
            precursor_idx += 1
    return pd.DataFrame(rows)


def _classifier() -> LightGBMClassifier:
    return LightGBMClassifier(
        n_estimators=30,
        final_n_estimators=30,
        min_child_samples=5,
        num_threads=1,
        random_state=0,
    )


def _fit_predict(trainer: CrossFittedTrainer, psm_df: pd.DataFrame, classifier=None):
    return trainer.fit_predict(
        classifier or _classifier(),
        psm_df[["feature", "noise"]].to_numpy(),
        psm_df["decoy"].to_numpy(),
        psm_df["elution_group_idx"].to_numpy(),
        psm_df["precursor_idx"].to_numpy(),
    )


def test_fit_predict_scores_every_row_out_of_fold():
    # Given: a separable feature set and three folds
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    classifier = _classifier()
    trainer = CrossFittedTrainer(n_folds=3, random_state=0)

    # When: the classifier is cross-fitted
    result = _fit_predict(trainer, psm_df, classifier)

    # Then: every row is scored, the passed classifier holds the last fold's model, which
    # was fitted on every row of the other two folds, and the true candidates score low
    assert result.proba.shape == (len(y),)
    assert classifier.fitted
    assert 0.6 < len(result.train_idx) / len(y) < 0.73
    is_true = psm_df["is_true"].to_numpy()
    assert (result.proba[is_true] < 0.5).mean() > 0.8
    assert result.proba[is_true].mean() < result.proba[y == 1].mean()


def test_fit_predict_refits_on_the_confident_targets_against_all_decoys():
    # Given: a separable feature set and two refits
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    trainer = CrossFittedTrainer(
        n_folds=2, train_fdr=0.01, n_refits=2, min_positives=100, random_state=0
    )

    # When: the classifier is cross-fitted
    result = _fit_predict(trainer, psm_df)

    # Then: the last fit's positives are the true candidates, a small part of the training
    # fold's targets, and its negatives are every decoy of the training fold (half of all)
    y_train_true = psm_df["is_true"].to_numpy()[result.train_idx]
    assert y_train_true[result.y_train == 0].mean() > 0.9
    assert (result.y_train == 0).sum() < 0.2 * (y == 0).sum() / 2
    assert (result.y_train == 1).sum() == (y == 1).sum() // 2


class _RecordingClassifier(LightGBMClassifier):
    """LightGBM that records how many positives each of its fits saw."""

    def __init__(self):
        super().__init__(
            n_estimators=30,
            final_n_estimators=30,
            min_child_samples=5,
            num_threads=1,
            random_state=0,
        )
        self.positives_per_fit = []

    def fit(self, x, y, *, is_final=False):
        self.positives_per_fit.append(int((y == 0).sum()))
        super().fit(x, y, is_final=is_final)


def test_fit_predict_fits_only_the_first_ensemble_member_on_every_target():
    # Given: a two-member ensemble and one refit
    psm_df = _gen_psms()
    n_targets_per_fold = (psm_df["decoy"] == 0).sum() // 2
    teacher, student = _RecordingClassifier(), _RecordingClassifier()
    trainer = CrossFittedTrainer(
        n_folds=2, train_fdr=0.01, n_refits=1, min_positives=100, random_state=0
    )

    # When: the ensemble is cross-fitted
    _fit_predict(trainer, psm_df, EnsembleClassifier([teacher, student]))

    # Then: the first member's first fit saw every target of its fold (folds are drawn at
    # random, so about half of all), the second member never saw more than the confident
    # ones, and both were refitted once on those
    assert (
        abs(teacher.positives_per_fit[0] - n_targets_per_fold)
        < 0.1 * n_targets_per_fold
    )
    assert len(teacher.positives_per_fit) == len(student.positives_per_fit) == 2
    assert max(student.positives_per_fit) < 0.5 * n_targets_per_fold
    assert student.positives_per_fit[1] == teacher.positives_per_fit[1]


def test_fit_predict_keeps_the_previous_model_when_too_few_positives(caplog):
    # Given: more positives required than there are true candidates
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    trainer = CrossFittedTrainer(
        n_folds=2,
        n_refits=3,
        min_positives=int(psm_df["is_true"].sum()) + 1,
        random_state=0,
    )

    # When: the classifier is cross-fitted
    result = _fit_predict(trainer, psm_df)

    # Then: the first fit on every target of the training folds stands
    assert (result.y_train == 0).sum() == (y[result.train_idx] == 0).sum()
    assert "keeping the model after 0 refit(s)" in caplog.text


class _FlakyClassifier(LightGBMClassifier):
    """Shifts its scores behind every other fold's after its first fit, until reset.

    Stands in for a fold model whose ranking is intact but whose scale is off against the
    other folds' models, which no check on the fit itself can see; the fold contributes
    no identifications to the merged ranking.
    """

    broken_fits_left = 0

    def fit(self, x, y, *, is_final=False):
        super().fit(x, y, is_final=is_final)
        if _FlakyClassifier.broken_fits_left > 0:
            _FlakyClassifier.broken_fits_left -= 1
            self.broken = True

    def reset(self):
        super().reset()
        self.broken = False

    def predict_proba(self, x):
        proba = super().predict_proba(x)
        if getattr(self, "broken", False):
            proba[:, 1] = 0.5 + proba[:, 1] / 2
            proba[:, 0] = 1 - proba[:, 1]
        return proba


def _flaky_classifier() -> _FlakyClassifier:
    return _FlakyClassifier(
        n_estimators=30,
        final_n_estimators=30,
        min_child_samples=5,
        num_threads=1,
        random_state=0,
    )


def test_fit_predict_refits_a_fold_whose_model_came_out_degenerate(caplog):
    # Given: a classifier whose very first fit, the first fold's, scores its fold behind
    # the others, on enough PSMs for the fold cuts of a healthy run to agree
    psm_df = _gen_psms(n_precursors=4000)
    _FlakyClassifier.broken_fits_left = 1
    healthy = _fit_predict(CrossFittedTrainer(n_folds=3, random_state=0), psm_df)

    # When: the folds are fitted
    result = _fit_predict(
        CrossFittedTrainer(n_folds=3, random_state=0), psm_df, _flaky_classifier()
    )

    # Then: the fold is refitted from fresh weights and scores as well as a healthy run
    assert "refitting it from fresh weights" in caplog.text
    assert _FlakyClassifier.broken_fits_left == 0
    true_rows = psm_df["is_true"].to_numpy()
    assert (result.proba[true_rows] < 0.5).mean() > 0.95
    assert np.corrcoef(result.proba, healthy.proba)[0, 1] > 0.9


def test_fit_predict_leaves_healthy_folds_alone(caplog):
    psm_df = _gen_psms(n_precursors=4000)

    _fit_predict(CrossFittedTrainer(n_folds=3, random_state=0), psm_df)

    assert "refitting it from fresh weights" not in caplog.text


def test_cross_fitted_needs_two_folds():
    with pytest.raises(ValueError, match="n_folds"):
        CrossFittedTrainer(n_folds=1)


def test_perform_fdr_with_a_trainer_reports_the_true_precursors():
    # Given: a separable feature set
    psm_df = _gen_psms()
    trainer = CrossFittedTrainer(
        n_folds=2, n_refits=1, min_positives=100, random_state=0
    )

    # When: the final round is run with the trainer
    result = fdr.perform_fdr(
        _classifier(),
        ["feature", "noise"],
        psm_df[psm_df["decoy"] == 0].copy(),
        psm_df[psm_df["decoy"] == 1].copy(),
        competitive=True,
        random_state=0,
        is_final=True,
        trainer=trainer,
    )

    # Then: the accepted targets are the true precursors
    accepted = result[(result["_decoy"] == 0) & (result["qval"] <= 0.05)]
    n_true = psm_df["is_true"].sum()
    assert len(accepted) > 0.8 * n_true
    assert accepted["is_true"].mean() > 0.9
    assert result["_decoy"].sum() > 0


def test_perform_fdr_ignores_the_trainer_in_optimization_rounds():
    # Given: a trainer and an optimization round
    psm_df = _gen_psms(n_precursors=600)
    trainer = CrossFittedTrainer(n_folds=2, random_state=0)

    # When: a non-final round is run with and without the trainer
    kwargs = {
        "competitive": True,
        "group_channels": True,
        "random_state": 0,
        "is_final": False,
    }
    with_trainer = fdr.perform_fdr(
        _classifier(),
        ["feature", "noise"],
        psm_df[psm_df["decoy"] == 0].copy(),
        psm_df[psm_df["decoy"] == 1].copy(),
        trainer=trainer,
        **kwargs,
    )
    without_trainer = fdr.perform_fdr(
        _classifier(),
        ["feature", "noise"],
        psm_df[psm_df["decoy"] == 0].copy(),
        psm_df[psm_df["decoy"] == 1].copy(),
        **kwargs,
    )

    # Then: the results are identical
    pd.testing.assert_frame_equal(with_trainer, without_trainer)


def test_perform_fdr_final_round_scores_with_the_first_ensemble_member():
    # Given: an ensemble of two recording members and a trainer
    psm_df = _gen_psms()
    first, second = _RecordingClassifier(), _RecordingClassifier()
    kwargs = {
        "competitive": True,
        "group_channels": True,
        "random_state": 0,
        "trainer": CrossFittedTrainer(n_folds=2, min_positives=100, random_state=0),
    }

    # When: an optimization round and the final round are run
    fdr.perform_fdr(
        EnsembleClassifier([first, second]),
        ["feature", "noise"],
        psm_df[psm_df["decoy"] == 0].copy(),
        psm_df[psm_df["decoy"] == 1].copy(),
        is_final=False,
        **kwargs,
    )
    n_fits_optimization = len(second.positives_per_fit)
    fdr.perform_fdr(
        EnsembleClassifier([first, second]),
        ["feature", "noise"],
        psm_df[psm_df["decoy"] == 0].copy(),
        psm_df[psm_df["decoy"] == 1].copy(),
        is_final=True,
        **kwargs,
    )

    # Then: both members were fitted in the optimization round, only the first in the
    # cross-fitted final round
    assert n_fits_optimization == 1
    assert len(second.positives_per_fit) == 1
    assert len(first.positives_per_fit) > 1

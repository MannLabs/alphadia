import numpy as np
import pandas as pd
import pytest

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import LightGBMClassifier
from alphadia.fdr.semisupervised import HiddenDecoyTrainer

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


def test_assign_hidden_hides_whole_decoy_precursors_only():
    # Given: candidates of target and decoy precursors
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    trainer = HiddenDecoyTrainer(hidden_decoy_fraction=0.5, random_state=0)

    # When: the hidden share is drawn
    is_hidden = trainer.assign_hidden(y, psm_df["precursor_idx"].to_numpy())

    # Then: no target is hidden, about half of the decoys are, and a precursor is hidden as a whole
    assert not is_hidden[y == 0].any()
    assert 0.4 < is_hidden[y == 1].mean() < 0.6
    per_precursor = pd.Series(is_hidden).groupby(psm_df["precursor_idx"]).nunique()
    assert (per_precursor == 1).all()


def test_fit_predict_keeps_only_confident_pseudo_targets_as_positives():
    # Given: a separable feature set
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    x = psm_df[["feature", "noise"]].to_numpy()
    trainer = HiddenDecoyTrainer(
        hidden_decoy_fraction=0.5,
        train_fdr=0.01,
        n_iterations=2,
        min_positives=100,
        random_state=0,
    )
    is_hidden = trainer.assign_hidden(y, psm_df["precursor_idx"].to_numpy())

    # When: the classifier is self-trained
    result = trainer.fit_predict(
        _classifier(),
        x,
        y,
        is_hidden,
        psm_df["elution_group_idx"].to_numpy(),
        psm_df["precursor_idx"].to_numpy(),
        is_final=True,
    )

    # Then: the last fit used the true candidates and the training decoys, not the false targets
    y_train_true = psm_df["is_true"].to_numpy()[result.train_idx]
    assert y_train_true[result.y_train == 0].mean() > 0.9
    assert (result.y_train == 1).sum() == int((~is_hidden & (y == 1)).sum())
    assert result.proba.shape == (len(psm_df),)
    assert result.proba[psm_df["is_true"]].mean() < result.proba[y == 1].mean()


def test_fit_predict_caps_the_training_decoys_of_a_refit():
    # Given: a cap of two training decoys per positive
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    x = psm_df[["feature", "noise"]].to_numpy()
    trainer = HiddenDecoyTrainer(
        hidden_decoy_fraction=0.5,
        train_fdr=0.01,
        n_iterations=1,
        max_negative_ratio=2.0,
        min_positives=100,
        random_state=0,
    )
    is_hidden = trainer.assign_hidden(y, psm_df["precursor_idx"].to_numpy())

    # When: the classifier is refitted once
    result = trainer.fit_predict(
        _classifier(),
        x,
        y,
        is_hidden,
        psm_df["elution_group_idx"].to_numpy(),
        psm_df["precursor_idx"].to_numpy(),
    )

    # Then: the refit saw exactly twice as many training decoys as positives
    n_positives = int((result.y_train == 0).sum())
    assert int((result.y_train == 1).sum()) == 2 * n_positives
    assert n_positives < int((~is_hidden & (y == 1)).sum()) / 2


def test_fit_predict_scores_hidden_decoys_like_false_targets():
    # Given: false targets drawn from the decoy distribution
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    x = psm_df[["feature", "noise"]].to_numpy()
    trainer = HiddenDecoyTrainer(
        hidden_decoy_fraction=0.5,
        train_fdr=0.01,
        n_iterations=1,
        min_positives=100,
        random_state=0,
    )
    is_hidden = trainer.assign_hidden(y, psm_df["precursor_idx"].to_numpy())

    # When: the classifier is self-trained
    result = trainer.fit_predict(
        _classifier(),
        x,
        y,
        is_hidden,
        psm_df["elution_group_idx"].to_numpy(),
        psm_df["precursor_idx"].to_numpy(),
        is_final=True,
    )

    # Then: hidden decoys and false targets pass a score cutoff at the same rate
    false_target = (y == 0) & ~psm_df["is_true"].to_numpy()
    cutoff = np.quantile(result.proba[false_target], 0.1)
    false_target_rate = (result.proba[false_target] <= cutoff).mean()
    hidden_rate = (result.proba[is_hidden] <= cutoff).mean()
    assert hidden_rate == pytest.approx(false_target_rate, abs=0.02)


def test_fit_predict_keeps_the_previous_model_when_too_few_positives(caplog):
    # Given: more positives required than there are true candidates
    psm_df = _gen_psms()
    y = psm_df["decoy"].to_numpy()
    x = psm_df[["feature", "noise"]].to_numpy()
    trainer = HiddenDecoyTrainer(
        hidden_decoy_fraction=0.5,
        train_fdr=0.01,
        n_iterations=3,
        min_positives=int(psm_df["is_true"].sum()) + 1,
        random_state=0,
    )
    is_hidden = trainer.assign_hidden(y, psm_df["precursor_idx"].to_numpy())

    # When: the classifier is self-trained
    result = trainer.fit_predict(
        _classifier(),
        x,
        y,
        is_hidden,
        psm_df["elution_group_idx"].to_numpy(),
        psm_df["precursor_idx"].to_numpy(),
    )

    # Then: the first fit on every pseudo-target stands
    assert len(result.train_idx) == len(psm_df)
    assert "keeping the model of iteration 0" in caplog.text


def test_hidden_decoy_fraction_must_be_a_proper_share():
    with pytest.raises(ValueError):
        HiddenDecoyTrainer(hidden_decoy_fraction=1.0)


def test_get_q_values_with_decoy_weights_counts_only_weighted_decoys():
    # Given: two hidden decoys weighted 2, two training decoys weighted 0, six targets
    df = pd.DataFrame(
        {
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "_decoy": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            "_decoy_weight": [0, 0, 0, 0.0, 0, 2.0, 0, 0.0, 0, 2.0],
            "precursor_idx": np.arange(10),
        }
    )

    # When: q-values are computed with the weights
    result = fdr.get_q_values(df, decoy_weight_column="_decoy_weight")

    # Then: the training decoys do not raise the FDR, the hidden ones count double
    qval = result.set_index("precursor_idx")["qval"]
    assert qval[3] == pytest.approx(0.0)
    assert qval[5] == pytest.approx(2 / 6)
    assert qval[9] == pytest.approx(4 / 6)


def test_perform_fdr_with_trainer_reports_the_true_precursors():
    # Given: a separable feature set
    psm_df = _gen_psms()
    trainer = HiddenDecoyTrainer(
        hidden_decoy_fraction=0.5,
        train_fdr=0.01,
        n_iterations=2,
        min_positives=100,
        random_state=0,
    )

    # When: the final round is run with the trainer
    result = fdr.perform_fdr(
        _classifier(),
        ["feature", "noise"],
        psm_df[psm_df["decoy"] == 0].copy(),
        psm_df[psm_df["decoy"] == 1].copy(),
        competitive=True,
        group_channels=True,
        is_final=True,
        trainer=trainer,
    )

    # Then: the accepted targets are the true precursors and the helper column is gone
    accepted = result[(result["_decoy"] == 0) & (result["qval"] <= 0.01)]
    n_true = psm_df["is_true"].sum()
    assert 0.8 * n_true < len(accepted) <= n_true
    assert accepted["is_true"].mean() > 0.97
    assert "_decoy_weight" not in result.columns
    assert result["_decoy"].sum() > 0


def test_perform_fdr_ignores_the_trainer_in_optimization_rounds():
    # Given: a trainer and an optimization round
    psm_df = _gen_psms(n_precursors=600)
    trainer = HiddenDecoyTrainer(hidden_decoy_fraction=0.5, random_state=0)

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

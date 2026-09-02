import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import (
    BinaryClassifierLegacyNewBatching,
    Classifier,
    LightGBMClassifier,
)


def test_keep_best():
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "channel": [0, 0, 1, 0, 1, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    best_df = fdr.keep_best(
        test_df, score_column="proba", group_columns=["precursor_idx"]
    )

    assert best_df.shape[0] == 3
    assert np.allclose(best_df["proba"].values, np.array([0.1, 0.4, 0.7]))

    best_df = fdr.keep_best(
        test_df, score_column="proba", group_columns=["channel", "precursor_idx"]
    )

    assert best_df.shape[0] == 6
    assert np.allclose(
        best_df["proba"].values, np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.9])
    )


def test_keep_best_2():
    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "elution_group_idx": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )

    result_df = fdr.keep_best(test_df, group_columns=["channel", "elution_group_idx"])
    pd.testing.assert_frame_equal(result_df, test_df)

    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "elution_group_idx": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )
    result_df = fdr.keep_best(test_df, group_columns=["channel", "elution_group_idx"])
    result_expected = pd.DataFrame(
        {
            "channel": [0, 0, 4, 4, 8, 8],
            "elution_group_idx": [0, 1, 0, 1, 0, 1],
            "proba": [0.1, 0.3, 0.4, 0.6, 0.1, 0.3],
        }
    )
    pd.testing.assert_frame_equal(result_df, result_expected)

    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "precursor_idx": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )
    result_df = fdr.keep_best(test_df, group_columns=["channel", "precursor_idx"])
    result_expected = pd.DataFrame(
        {
            "channel": [0, 0, 4, 4, 8, 8],
            "precursor_idx": [0, 1, 0, 1, 0, 1],
            "proba": [0.1, 0.3, 0.4, 0.6, 0.1, 0.3],
        }
    )
    pd.testing.assert_frame_equal(result_df, result_expected)


def test_fdr_to_q_values():
    test_fdr = np.array([0.2, 0.1, 0.05, 0.3, 0.26, 0.25, 0.5])

    test_q_values = fdr._fdr_to_q_values(test_fdr)

    assert np.allclose(
        test_q_values, np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.5])
    )


def test_get_q_values():
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "_decoy": [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        }
    )

    test_df = fdr.get_q_values(test_df, "proba", "_decoy")

    assert np.allclose(
        test_df["qval"].values,
        np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]),
    )


def gen_data_np(
    n_features=10,
    n_samples=10000,
    max_mean=100,
    max_var=0.1,
):
    mean = np.random.random(n_features * 2) * max_mean
    var = np.random.random(n_features * 2) * max_var
    data = np.random.multivariate_normal(
        mean, np.eye(n_features * 2) * var, size=n_samples
    )
    return data.reshape(-1, n_features), np.tile([0, 1], n_samples)


def test_feed_forward():
    x, y = gen_data_np()

    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100,
    )

    classifier.fit(x, y)
    # assert classifier.metrics["test_accuracy"][-1] > 0.99
    # assert classifier.metrics["train_accuracy"][-1] > 0.99

    y_pred = classifier.predict(x)  # noqa: F841  # TODO fix this test
    # assert np.all(y_pred == y)

    y_proba = classifier.predict_proba(x)[:, 1]  # noqa: F841  # TODO fix this test
    # assert np.all(np.round(y_proba) == y)


def test_feed_forward_save():
    tempfolder = tempfile.gettempdir()
    x, y = gen_data_np()

    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100,
    )

    classifier.fit(x, y)

    torch.save(
        classifier.to_state_dict(),
        os.path.join(tempfolder, "test_feed_forward_save.pth"),
    )

    new_classifier = BinaryClassifierLegacyNewBatching()
    new_classifier.from_state_dict(
        torch.load(
            os.path.join(tempfolder, "test_feed_forward_save.pth"), weights_only=False
        )
    )

    y_pred = new_classifier.predict(x)  # noqa: F841  # TODO fix this test
    # assert np.all(y_pred == y)


def test_classifier_reset():
    # Given: a fitted classifier
    x, y = gen_data_np()
    classifier = BinaryClassifierLegacyNewBatching(batch_size=100)
    classifier.fit(x, y)
    first_weights = classifier.network.fc_layers[0].weight.detach().numpy().copy()

    # When: the test resets it
    classifier.reset()

    # Then: the classifier is unfitted, with no network and no metrics
    assert classifier.fitted is False
    assert classifier.network is None
    assert all(len(values) == 0 for values in classifier.metrics.values())

    # And: a new fit uses new random weights, not the first ones
    classifier.fit(x, y)
    second_weights = classifier.network.fc_layers[0].weight.detach().numpy()
    assert classifier.fitted is True
    assert not np.allclose(first_weights, second_weights)


class _CollapsingClassifier(Classifier):
    """Give a constant probability until the caller resets it `n_collapses` times."""

    def __init__(self, n_collapses: int):
        self._n_collapses = n_collapses
        self._fitted = False
        self.reset_count = 0

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, x, y, *, is_final=False):
        self._fitted = True

    def reset(self):
        self.reset_count += 1
        self._fitted = False

    def predict(self, x):
        return self.predict_proba(x)[:, 1]

    def predict_proba(self, x):
        if self.reset_count < self._n_collapses:
            return np.full((len(x), 2), 0.5)
        proba = np.linspace(0.0, 1.0, len(x))
        return np.stack([1 - proba, proba], axis=1)

    def to_state_dict(self):
        return {}

    def from_state_dict(self, state_dict):
        pass


def _gen_target_decoy_dfs(n_samples: int = 200):
    feature = np.linspace(0.0, 1.0, n_samples)
    target_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_samples),
            "decoy": 0,
            "feature": feature,
        }
    )
    decoy_df = target_df.assign(
        precursor_idx=np.arange(n_samples, 2 * n_samples), decoy=1
    )
    return target_df, decoy_df


def test_perform_fdr_resets_collapsed_classifier():
    # Given: a classifier that collapses once, then separates targets from decoys
    classifier = _CollapsingClassifier(n_collapses=1)
    target_df, decoy_df = _gen_target_decoy_dfs()

    # When: perform_fdr runs
    psm_df = fdr.perform_fdr(classifier, ["feature"], target_df, decoy_df)

    # Then: perform_fdr resets the classifier once and uses the new probabilities
    assert classifier.reset_count == 1
    assert psm_df["proba"].std() > 0.0


def test_perform_fdr_stops_after_max_reinits():
    # Given: a classifier that never recovers
    classifier = _CollapsingClassifier(n_collapses=1000)
    target_df, decoy_df = _gen_target_decoy_dfs()

    # When: perform_fdr runs
    psm_df = fdr.perform_fdr(classifier, ["feature"], target_df, decoy_df)

    # Then: perform_fdr stops after the maximum number of retries
    assert classifier.reset_count == fdr._MAX_FDR_CLASSIFIER_REINITS
    assert psm_df["proba"].std() == 0.0


def _get_lightgbm_classifier() -> LightGBMClassifier:
    return LightGBMClassifier(
        n_estimators=20, min_child_samples=5, num_threads=1, random_state=0
    )


def test_lightgbm_fit_predict():
    # Given: separable data and an unfitted classifier
    x, y = gen_data_np()
    classifier = _get_lightgbm_classifier()

    # When: the classifier is fitted
    classifier.fit(x, y)

    # Then: it is fitted, predicts the classes and returns proper probabilities
    assert classifier.fitted is True
    assert np.mean(classifier.predict(x) == y) > 0.95  # noqa: PLR2004

    y_proba = classifier.predict_proba(x)
    assert y_proba.shape == (len(x), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_lightgbm_state_dict_roundtrip():
    # Given: a fitted classifier saved to disk
    x, y = gen_data_np()
    classifier = _get_lightgbm_classifier()
    classifier.fit(x, y)

    path = os.path.join(tempfile.gettempdir(), "test_lightgbm_save.pth")
    torch.save(classifier.to_state_dict(), path)

    # When: a new classifier is loaded from the state dict
    new_classifier = LightGBMClassifier()
    new_classifier.from_state_dict(torch.load(path, weights_only=False))

    # Then: it is fitted and predicts the same probabilities
    assert new_classifier.fitted is True
    assert np.allclose(classifier.predict_proba(x), new_classifier.predict_proba(x))
    assert new_classifier.to_state_dict()["early_stopping_rounds"] == 50  # noqa: PLR2004
    assert new_classifier.to_state_dict()["validation_fraction"] == 0.1  # noqa: PLR2004
    assert new_classifier.to_state_dict()["final_n_estimators"] == 3000  # noqa: PLR2004
    assert new_classifier.to_state_dict()["final_validation_fraction"] == 0.075  # noqa: PLR2004
    assert new_classifier.to_state_dict()["max_bin"] == 123  # noqa: PLR2004
    assert new_classifier.to_state_dict()["data_sample_strategy"] == "bagging"
    assert new_classifier.to_state_dict()["learning_rate_start"] == 0.15  # noqa: PLR2004
    assert new_classifier.to_state_dict()["learning_rate_end"] == 0.03  # noqa: PLR2004
    assert new_classifier.to_state_dict()["learning_rate_decay_rounds"] == 1600  # noqa: PLR2004


def test_lightgbm_learning_rate_schedule_decays_between_the_endpoints():
    # Given: a classifier with a short decay horizon
    classifier = LightGBMClassifier(
        learning_rate_start=0.15,
        learning_rate_end=0.03,
        learning_rate_decay_rounds=100,
    )

    # When / Then: the rate starts high, decays monotonically and holds at the floor
    rates = [classifier._learning_rate_at(i) for i in range(120)]  # noqa: SLF001
    assert rates[0] == 0.15  # noqa: PLR2004
    assert all(
        later <= earlier for earlier, later in zip(rates, rates[1:], strict=False)
    )
    assert rates[100] == 0.03  # noqa: PLR2004
    assert rates[119] == 0.03  # noqa: PLR2004
    assert 0.03 < rates[50] < 0.15  # noqa: PLR2004


def _gen_weak_signal_data() -> tuple[np.ndarray, np.ndarray]:
    """Features carrying just enough signal that boosting starts to overfit."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 5))
    y = (x[:, 0] + rng.normal(scale=3.0, size=4000) > 0).astype(int)
    return x, y


def _get_early_stopping_classifier() -> LightGBMClassifier:
    return LightGBMClassifier(
        n_estimators=200,
        final_n_estimators=300,
        early_stopping_rounds=5,
        min_child_samples=5,
        num_threads=1,
        random_state=0,
    )


def test_lightgbm_early_stopping_uses_fewer_trees_than_the_maximum():
    # Given: data the maximum number of trees would overfit
    x, y = _gen_weak_signal_data()
    classifier = _get_early_stopping_classifier()

    # When: the classifier is fitted
    classifier.fit(x, y)

    # Then: boosting stops well before the maximum number of trees
    assert 0 < classifier._booster.num_trees() < 200  # noqa: PLR2004, SLF001


def test_lightgbm_final_fit_uses_the_final_round_budget():
    # Given: the same data fitted as an optimization round and as the final round
    x, y = _gen_weak_signal_data()

    optimization = _get_early_stopping_classifier()
    optimization.fit(x, y)

    final = _get_early_stopping_classifier()
    final.fit(x, y, is_final=True)

    # Then: both stop early, each within its own budget ...
    assert 0 < optimization._booster.num_trees() < 200  # noqa: PLR2004, SLF001
    assert 0 < final._booster.num_trees() < 300  # noqa: PLR2004, SLF001

    # ... and the final round trains on more data, giving a different model
    assert not np.allclose(final.predict_proba(x), optimization.predict_proba(x))


def test_lightgbm_from_state_dict_without_model_stays_unfitted():
    # Given: a state dict of a different classifier type
    state_dict = BinaryClassifierLegacyNewBatching().to_state_dict()
    classifier = LightGBMClassifier()

    # When: it is loaded into the lightgbm classifier
    classifier.from_state_dict(state_dict)

    # Then: the classifier stays unfitted
    assert classifier.fitted is False


def test_lightgbm_predict_before_fit_raises():
    x, _ = gen_data_np()
    classifier = LightGBMClassifier()

    with pytest.raises(ValueError, match="has not been fitted"):
        classifier.predict(x)


def test_perform_fdr_lightgbm_separates_targets_and_decoys():
    # Given: targets and decoys with a separable feature
    n_samples = 200
    target_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_samples),
            "decoy": 0,
            "feature": np.linspace(0.0, 1.0, n_samples),
        }
    )
    decoy_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_samples, 2 * n_samples),
            "decoy": 1,
            "feature": np.linspace(1.0, 2.0, n_samples),
        }
    )

    # When: perform_fdr runs with the lightgbm classifier
    psm_df = fdr.perform_fdr(
        _get_lightgbm_classifier(), ["feature"], target_df, decoy_df, random_state=0
    )

    # Then: targets get lower decoy probabilities and q-values than decoys
    target_psms = psm_df[psm_df["_decoy"] == 0]
    decoy_psms = psm_df[psm_df["_decoy"] == 1]
    assert target_psms["proba"].median() < decoy_psms["proba"].median()
    assert target_psms["qval"].median() < decoy_psms["qval"].median()

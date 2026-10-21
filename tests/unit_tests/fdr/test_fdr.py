import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from alphadia.constants.keys import FeatureTransform
from alphadia.fdr import fdr
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching, Classifier


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

    def fit(self, x, y):
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


def test_fit_feature_transform_falls_back_to_raw_features_for_few_rows():
    # Given: a classifier with the quantile transform and fewer rows than the minimum
    classifier = BinaryClassifierLegacyNewBatching(
        feature_transform=FeatureTransform.QUANTILE
    )
    x, _ = gen_data_np(n_samples=100)

    # When: the transform is fitted
    transformer = classifier._fit_feature_transform(x)

    # Then: no transformer is fitted, so the raw features are used
    assert transformer is None


def test_classifier_raises_for_unknown_transform():
    # When/Then: constructing a classifier with an unknown transform fails explicitly
    with pytest.raises(ValueError, match="Unknown feature transform"):
        BinaryClassifierLegacyNewBatching(feature_transform="unknown")


def test_quantile_transform_state_dict_round_trip():
    # Given: a classifier fitted with the quantile transform
    x, y = gen_data_np(n_samples=2000)
    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100, feature_transform=FeatureTransform.QUANTILE
    )
    classifier.fit(x, y)
    assert classifier._quantile_transformer is not None

    # When: the state dict is round tripped through a new classifier
    new_classifier = BinaryClassifierLegacyNewBatching()
    new_classifier.from_state_dict(classifier.to_state_dict())

    # Then: the predictions are identical
    assert np.array_equal(new_classifier.predict_proba(x), classifier.predict_proba(x))


# one isolation window from 400 to 600 m/z, no mobility
_DIA_CYCLE = np.array([[[[400.0, 600.0]]]])


def _run_perform_fdr_with_mocked_competition(**fragment_kwargs):
    """Run perform_fdr with compete_for_fragments replaced by a pass-through mock.

    The competition itself is tested in alphadia-search-rs; this only checks which
    fragments perform_fdr hands to it.
    """
    target_df, decoy_df = _gen_target_decoy_dfs()
    with patch.object(
        fdr, "compete_for_fragments", side_effect=lambda psm_df, *_: psm_df
    ) as mock_compete:
        fdr.perform_fdr(
            _CollapsingClassifier(n_collapses=0),
            ["feature"],
            target_df,
            decoy_df,
            dia_cycle=_DIA_CYCLE,
            **fragment_kwargs,
        )
    return mock_compete


def test_perform_fdr_quantifies_fragments_on_demand_for_the_competing_psms():
    # Given: a provider instead of fragments
    provider = MagicMock(name="provider")

    # When: perform_fdr runs
    mock_compete = _run_perform_fdr_with_mocked_competition(fragment_provider=provider)

    # Then: the provider is asked once, only for the PSMs below the heuristic q-value
    provider.assert_called_once()
    requested_psms = provider.call_args.args[0]
    assert len(requested_psms) > 0
    assert requested_psms["qval"].max() < 0.1
    # and its fragments go to the competition
    assert mock_compete.call_args.args[1] is provider.return_value


def test_perform_fdr_prefers_given_fragments_over_the_provider():
    # Given: fragments up front, and a provider
    df_fragments = pd.DataFrame()
    provider = MagicMock(name="provider")

    # When: perform_fdr runs with both
    mock_compete = _run_perform_fdr_with_mocked_competition(
        df_fragments=df_fragments, fragment_provider=provider
    )

    # Then: the given fragments go to the competition, and the provider is not asked
    provider.assert_not_called()
    assert mock_compete.call_args.args[1] is df_fragments

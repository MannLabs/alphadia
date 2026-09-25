import os
import tempfile

import numpy as np
import pandas as pd
import torch

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


def test_get_q_values_ties_share_a_q_value():
    """A tied-score block must not be credited with its targets-first ordering.

    The sort puts all four targets of the 0.5 block ahead of both its decoys, so a
    running ratio taken mid-block sees no decoys at all and reads 0. Every member of
    the block is accepted or rejected together, so all six must carry the ratio as it
    stands once the block is through: 2 decoys / 5 targets. The lone 0.1 target is a
    block of its own and keeps its own ratio.
    """
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4, 5, 6],
            "proba": [0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "_decoy": [0, 0, 0, 0, 0, 1, 1],
        }
    )

    test_df = fdr.get_q_values(test_df, "proba", "_decoy")

    assert np.allclose(test_df["qval"].values, np.array([0.0, *[2 / 5] * 6]))


def test_get_q_values_reads_one_while_no_target_has_been_seen():
    # Given: decoys ranked ahead of every target
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3],
            "proba": [0.1, 0.2, 0.3, 0.4],
            "_decoy": [1, 1, 0, 0],
        }
    )

    test_df = fdr.get_q_values(test_df, "proba", "_decoy")

    # Then: the leading decoys carry a q-value of one rather than a division by zero
    assert np.allclose(test_df["qval"].values, [1.0, 1.0, 1.0, 1.0])


def test_get_q_values_with_decoy_offset():
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "_decoy": [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        }
    )

    test_df = fdr.get_q_values(test_df, "proba", "_decoy", decoy_offset=1)

    assert np.allclose(
        test_df["qval"].values,
        np.array([1 / 3, 1 / 3, 1 / 3, 0.4, 0.4, 0.4, 0.6, 0.8, 1.0, 1.2]),
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


def _gen_competing_psms(n_samples: int = 200):
    """Targets 0 and 1 elute together and share their fragments; every other PSM elutes alone."""
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples)
    for df in (target_df, decoy_df):
        df["rank"] = 0
        df["mz_observed"] = 500.0
        df["rt_observed"] = df["precursor_idx"] * 10.0
    target_df.loc[target_df["precursor_idx"] == 1, "rt_observed"] = 0.0
    return target_df, decoy_df


def _fragments_of(psm_df: pd.DataFrame) -> pd.DataFrame:
    """Three fragments per PSM; PSMs 0 and 1 have the same ones, all others unique ones."""
    precursor_idx = psm_df["precursor_idx"].to_numpy()
    base_mz = np.where(precursor_idx == 1, 0, precursor_idx) * 10.0 + 200.0
    return pd.DataFrame(
        {
            "precursor_idx": np.repeat(precursor_idx, 3),
            "rank": 0,
            "mz_observed": (base_mz[:, None] + np.array([0.0, 1.0, 2.0])).ravel(),
        }
    )


# one isolation window from 400 to 600 m/z, no mobility
_DIA_CYCLE = np.array([[[[400.0, 600.0]]]])


def test_perform_fdr_fragment_provider_removes_the_weaker_of_two_psms_sharing_fragments():
    # Given: PSMs 0 and 1 claim the same fragments at the same time, and PSM 0 scores better
    classifier = _CollapsingClassifier(n_collapses=0)
    target_df, decoy_df = _gen_competing_psms()
    requested = []

    def provider(psm_df):
        requested.append(psm_df)
        return _fragments_of(psm_df)

    # When: perform_fdr gets the fragments on demand
    psm_df = fdr.perform_fdr(
        classifier,
        ["feature"],
        target_df,
        decoy_df,
        fragment_provider=provider,
        dia_cycle=_DIA_CYCLE,
    )

    # Then: the provider is asked once, only for the PSMs below the heuristic q-value
    assert len(requested) == 1
    assert requested[0]["qval"].max() < 0.1
    # and PSM 1 lost its fragments to PSM 0, while PSM 0 and the lone PSMs survive
    assert 1 not in psm_df["precursor_idx"].to_numpy()
    assert {0, 2, 3} <= set(psm_df["precursor_idx"])


def test_perform_fdr_prefers_given_fragments_over_the_provider():
    # Given: the fragments of all PSMs up front, and a provider that must not be needed
    classifier = _CollapsingClassifier(n_collapses=0)
    target_df, decoy_df = _gen_competing_psms()
    all_psms = pd.concat([target_df, decoy_df])

    def provider(psm_df):
        raise AssertionError(
            "the provider must not be called when df_fragments is given"
        )

    # When: perform_fdr runs with both
    psm_df = fdr.perform_fdr(
        classifier,
        ["feature"],
        target_df,
        decoy_df,
        df_fragments=_fragments_of(all_psms),
        fragment_provider=provider,
        dia_cycle=_DIA_CYCLE,
    )

    # Then: the given fragments decide the competition
    assert 1 not in psm_df["precursor_idx"].to_numpy()

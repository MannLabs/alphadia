import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm

from alphadia.constants.keys import FeatureTransform
from alphadia.fdr import classifiers, fdr
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


class _PlateauClassifier(_CollapsingClassifier):
    """Give the best `n_tied` PSMs one probability until the caller resets it `n_collapses` times."""

    def __init__(self, n_collapses: int, n_tied: int):
        super().__init__(n_collapses)
        self._n_tied = n_tied

    def predict_proba(self, x):
        proba = np.linspace(0.0, 1.0, len(x))
        if self.reset_count < self._n_collapses:
            proba[: self._n_tied] = 0.0
        return np.stack([1 - proba, proba], axis=1)


def test_perform_fdr_resets_classifier_that_ties_the_best_psms():
    # Given: a classifier whose best PSMs share one probability, with spread-out probabilities elsewhere
    classifier = _PlateauClassifier(n_collapses=1, n_tied=fdr._MAX_TIED_BEST_PROBA + 1)
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples=1000)

    # When: perform_fdr runs
    psm_df = fdr.perform_fdr(classifier, ["feature"], target_df, decoy_df)

    # Then: perform_fdr resets the classifier once and no longer ties the best PSMs
    assert classifier.reset_count == 1
    assert (psm_df["proba"] == psm_df["proba"].min()).sum() == 1


def test_perform_fdr_keeps_classifier_with_few_tied_best_psms():
    # Given: a classifier whose best PSMs tie, but no more than the allowed number
    classifier = _PlateauClassifier(n_collapses=1, n_tied=fdr._MAX_TIED_BEST_PROBA)
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples=1000)

    # When: perform_fdr runs
    fdr.perform_fdr(classifier, ["feature"], target_df, decoy_df)

    # Then: perform_fdr keeps it
    assert classifier.reset_count == 0


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


def _gen_quantile_transform_data(n_samples: int = 5000, random_state: int = 42):
    """Return a matrix with a constant and a heavy-tailed column."""
    rng = np.random.default_rng(random_state)
    return np.stack(
        [np.full(n_samples, 7.0), rng.lognormal(mean=0.0, sigma=2.0, size=n_samples)],
        axis=1,
    )


def test_fit_transform_maps_features_to_normal_scores():
    # Given: a classifier with the quantile transform and a constant and a heavy-tailed feature
    classifier = BinaryClassifierLegacyNewBatching(
        feature_transform=FeatureTransform.QUANTILE
    )
    x = _gen_quantile_transform_data()

    # When: the transform is fitted and applied
    x_transformed = classifier._fit_transform(x)

    # Then: the constant feature maps to zero
    assert np.all(x_transformed[:, 0] == 0.0)

    # And: the heavy-tailed feature follows a standard normal
    probabilities = np.arange(0.1, 1.0, 0.1)
    assert np.allclose(
        np.quantile(x_transformed[:, 1], probabilities),
        norm.ppf(probabilities),
        atol=0.05,
    )


def test_apply_transform_keeps_out_of_distribution_values_finite():
    # Given: a fitted quantile transform
    classifier = BinaryClassifierLegacyNewBatching(
        feature_transform=FeatureTransform.QUANTILE
    )
    x = _gen_quantile_transform_data()
    classifier._fit_transform(x)

    # When: values far outside the training range are transformed
    x_extreme = np.array([[7.0, x[:, 1].max() * 1e6], [7.0, x[:, 1].min() * 1e-6]])
    x_transformed = classifier._apply_transform(x_extreme)

    # Then: they stay at the edge of the distribution instead of outside it
    assert np.all(np.isfinite(x_transformed))
    assert x_transformed[0, 1] == norm.ppf(1 - classifiers._QUANTILE_CLIP)
    assert x_transformed[1, 1] == norm.ppf(classifiers._QUANTILE_CLIP)


def test_fit_transform_falls_back_to_raw_features_for_few_rows():
    # Given: a classifier with the quantile transform and fewer rows than the minimum
    classifier = BinaryClassifierLegacyNewBatching(
        feature_transform=FeatureTransform.QUANTILE
    )
    x = _gen_quantile_transform_data(n_samples=100)

    # When: the transform is fitted
    x_transformed = classifier._fit_transform(x)

    # Then: the features are passed through unchanged
    assert classifier._quantiles is None
    assert np.array_equal(x_transformed, x)


def test_fit_transform_raises_for_unknown_transform():
    # Given: a classifier with an unknown feature transform
    classifier = BinaryClassifierLegacyNewBatching(feature_transform="rank")

    # When/Then: fitting the transform fails explicitly
    with pytest.raises(ValueError, match="Unknown feature transform"):
        classifier._fit_transform(_gen_quantile_transform_data())


def test_quantile_transform_state_dict_round_trip():
    # Given: a classifier fitted with the quantile transform
    x, y = gen_data_np(n_samples=2000)
    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100, feature_transform=FeatureTransform.QUANTILE
    )
    classifier.fit(x, y)
    assert classifier._quantiles is not None

    # When: the state dict is round tripped through a new classifier
    new_classifier = BinaryClassifierLegacyNewBatching()
    new_classifier.from_state_dict(classifier.to_state_dict())

    # Then: the transform and its quantile table are preserved
    assert new_classifier.feature_transform == FeatureTransform.QUANTILE
    assert np.array_equal(new_classifier._quantiles, classifier._quantiles)

    # And: the predictions are identical
    assert np.array_equal(new_classifier.predict_proba(x), classifier.predict_proba(x))


def test_from_state_dict_defaults_to_no_transform():
    # Given: a state dict written before the feature transform existed
    x, y = gen_data_np(n_samples=2000)
    classifier = BinaryClassifierLegacyNewBatching(batch_size=100)
    classifier.fit(x, y)
    state_dict = classifier.to_state_dict()
    del state_dict["feature_transform"]
    del state_dict["_quantiles"]

    # When: it is loaded into a classifier requesting the quantile transform
    new_classifier = BinaryClassifierLegacyNewBatching(
        feature_transform=FeatureTransform.QUANTILE
    )
    new_classifier.from_state_dict(state_dict)

    # Then: the loaded weights keep being used with raw features
    assert new_classifier.feature_transform == FeatureTransform.NONE
    assert new_classifier._quantiles is None


class _MemorizingClassifier(_CollapsingClassifier):
    """Record the feature values of every fit and of every prediction."""

    def __init__(self):
        super().__init__(n_collapses=0)
        self.fits = []
        self.predictions = []

    def __deepcopy__(self, memo):
        # the fold models are copies; they log into the same lists
        clone = _MemorizingClassifier()
        clone.fits, clone.predictions = self.fits, self.predictions
        return clone

    def fit(self, x, y):
        super().fit(x, y)
        self.fits.append(set(x[:, 0]))

    def predict_proba(self, x):
        self.predictions.append(set(x[:, 0]))
        return super().predict_proba(x)


def _gen_elution_group_psms(n_samples: int = 1000):
    """Every target shares its elution group with one decoy; each PSM has a unique feature value."""
    target_df, decoy_df = _gen_target_decoy_dfs(n_samples)
    target_df["elution_group_idx"] = np.arange(n_samples)
    decoy_df["elution_group_idx"] = np.arange(n_samples)
    decoy_df["feature"] = decoy_df["feature"] + 2.0
    return target_df, decoy_df


def test_perform_fdr_cross_fit_never_scores_a_psm_with_a_model_fitted_on_it():
    # Given: a classifier that records what it was fitted on and what it scored
    classifier = _MemorizingClassifier()
    target_df, decoy_df = _gen_elution_group_psms()

    # When: perform_fdr runs cross-fitted
    psm_df = fdr.perform_fdr(
        classifier, ["feature"], target_df, decoy_df, cross_fit=True
    )

    # Then: every fold scores PSMs its model has not seen, and all PSMs are scored once
    assert len(classifier.fits) == fdr._CROSS_FIT_FOLDS
    assert all(
        not (fit & scored)
        for fit, scored in zip(classifier.fits, classifier.predictions, strict=True)
    )
    assert sum(len(scored) for scored in classifier.predictions) == len(psm_df)


def test_perform_fdr_cross_fit_keeps_a_target_and_its_decoy_in_one_fold():
    # Given: targets and decoys paired by elution group
    classifier = _MemorizingClassifier()
    target_df, decoy_df = _gen_elution_group_psms()

    # When: perform_fdr runs cross-fitted
    fdr.perform_fdr(classifier, ["feature"], target_df, decoy_df, cross_fit=True)

    # Then: the decoy of every target is scored by the same fold model as the target
    for scored in classifier.predictions:
        targets = {value for value in scored if value < 2.0}
        assert {value + 2.0 for value in targets} == scored - targets

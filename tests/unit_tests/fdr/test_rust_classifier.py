import pickle
from copy import deepcopy

import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer

from alphadia.constants.keys import FeatureTransform
from alphadia.fdr.classifiers import RustMLPClassifier

N_FEATURES = 12
N_SHIFTED_FEATURES = 4


def _separable_data(
    n_samples: int = 6000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Targets and decoys shifted apart in a few features, one heavy-tailed feature."""
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n_samples // 2)
    x = rng.normal(size=(n_samples, N_FEATURES))
    x[y == 1, :N_SHIFTED_FEATURES] += 2.0
    x[:, -1] = np.exp(x[:, -1])
    return x, y


def _classifier(**kwargs) -> RustMLPClassifier:
    defaults = {"layers": [32, 16], "epochs": 5, "batch_size": 256, "random_state": 1}
    return RustMLPClassifier(**{**defaults, **kwargs})


def test_fit_separates_targets_and_decoys():
    # Given separable data
    x, y = _separable_data()
    classifier = _classifier()

    # When fitting and predicting
    classifier.fit(x, y)
    y_pred = classifier.predict(x)
    proba = classifier.predict_proba(x)

    # Then nearly all rows are classified correctly and the probabilities are consistent
    assert classifier.fitted
    assert np.mean(y_pred == y) > 0.95
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(y_pred, (proba[:, 1] > 0.5).astype(int))


def test_predict_before_fit_raises():
    # Given an unfitted classifier
    x, _ = _separable_data(100)

    # When / Then predicting raises
    with pytest.raises(ValueError, match="not been fitted"):
        _classifier().predict_proba(x)


def test_predict_with_wrong_number_of_features_raises():
    # Given a fitted classifier
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)

    # When / Then predicting on a different number of features raises
    with pytest.raises(ValueError, match="shape"):
        classifier.predict(x[:, :-1])


def test_state_dict_pickle_and_deepcopy_reproduce_predictions():
    # Given a fitted classifier
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)
    expected = classifier.predict_proba(x)

    # When restoring it from its state dict, a pickle and a deep copy
    restored = _classifier()
    restored.from_state_dict(classifier.to_state_dict())
    unpickled = pickle.loads(pickle.dumps(classifier))
    copied = deepcopy(classifier)

    # Then all give the same probabilities
    for other in (restored, unpickled, copied):
        assert np.allclose(other.predict_proba(x), expected)


def test_from_state_dict_rejects_torch_state():
    # Given a state dict of the torch classifier
    torch_state = {"_fitted": True, "input_dim": 3, "network_state_dict": {}}

    # When / Then loading it raises
    with pytest.raises(ValueError, match="params"):
        _classifier().from_state_dict(torch_state)


def test_reset_and_warm_start():
    # Given a fitted classifier
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)
    first = classifier.to_state_dict()["params"]

    # When fitting again, the parameters continue from the first fit
    classifier.fit(x, y)
    second = classifier.to_state_dict()["params"]

    # And when resetting
    classifier.reset()

    # Then the second fit changed the parameters and the reset unfits the classifier
    assert second.shape == first.shape
    assert not np.allclose(first, second)
    assert not classifier.fitted


def test_changed_input_dimension_reinitializes_with_warning():
    # Given a classifier fitted on 12 features
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)

    # When fitting on 11 features
    with pytest.warns(UserWarning, match="reinitialized"):
        classifier.fit(x[:, :-1], y)

    # Then it predicts on 11 features
    assert classifier.input_dim == N_FEATURES - 1
    assert classifier.predict(x[:, :-1]).shape == (len(x),)


def test_quantile_transform_matches_sklearn():
    # Given a classifier with the quantile transform and heavy-tailed data
    x, y = _separable_data(20000)
    classifier = _classifier(feature_transform=FeatureTransform.QUANTILE)
    classifier.fit(x, y)

    # When transforming the features
    transformed = classifier._transform_features(x.astype(np.float32))

    # Then they agree with sklearn's QuantileTransformer
    expected = QuantileTransformer(
        n_quantiles=1001, output_distribution="normal", subsample=None
    ).fit_transform(x.astype(np.float32))
    assert np.abs(transformed - expected).mean() < 1e-3
    assert np.abs(transformed - expected).max() < 0.05


def test_feature_transform_falls_back_to_raw_features_for_few_rows():
    # Given fewer rows than the quantile transform needs
    x, y = _separable_data(500)
    classifier = _classifier(feature_transform=FeatureTransform.QUANTILE)

    # When fitting
    classifier.fit(x, y)

    # Then no quantiles are fitted and the raw features are used
    assert classifier.to_state_dict()["quantiles"] is None
    assert classifier.fitted


def test_feature_transform_none_keeps_raw_features():
    # Given a classifier without feature transform
    x, y = _separable_data()
    classifier = _classifier(feature_transform=FeatureTransform.NONE)

    # When fitting
    classifier.fit(x, y)

    # Then no quantiles are stored
    assert classifier.to_state_dict()["quantiles"] is None


def test_unknown_feature_transform_raises():
    # When / Then an unknown transform raises
    with pytest.raises(ValueError, match="Unknown feature transform"):
        RustMLPClassifier(feature_transform="unknown")


def test_hyperparameter_tuning_fits_small_data():
    # Given hyperparameter tuning, which scales the batch size down for few rows
    x, y = _separable_data(3000)
    classifier = _classifier(experimental_hyperparameter_tuning=True)

    # When fitting
    classifier.fit(x, y)

    # Then the classifier is fitted and separates the data
    assert np.mean(classifier.predict(x) == y) > 0.9


@pytest.mark.parametrize(
    ("lr_schedule", "weight_averaging"),
    [("cosine", False), ("constant", True), ("cosine", True)],
)
def test_configured_training_changes_the_fit_and_still_separates(
    lr_schedule, weight_averaging
):
    # Given separable data and the plain (constant, last parameters) fit
    x, y = _separable_data()
    plain = _classifier()
    plain.fit(x, y)

    # When fitting with the configured smoothing, as FDRManager sets it
    smoothed = _classifier()
    smoothed.configure_training(
        lr_schedule=lr_schedule, weight_averaging=weight_averaging
    )
    smoothed.fit(x, y)

    # Then the network differs from the plain fit and still separates the classes
    assert not np.array_equal(smoothed._params, plain._params)
    assert np.mean(smoothed.predict(x) == y) > 0.95


def test_configure_training_keeps_unset_values_and_rejects_unknown_schedule():
    # Given a classifier with the cosine schedule
    classifier = _classifier(lr_schedule="cosine")

    # When only the averaging is configured
    classifier.configure_training(weight_averaging=True)

    # Then the schedule is kept, and an unknown schedule raises
    assert (classifier.lr_schedule, classifier.weight_averaging) == ("cosine", True)
    with pytest.raises(ValueError, match="learning rate schedule"):
        classifier.configure_training(lr_schedule="linear")


def test_state_dict_carries_the_training_configuration():
    # Given a classifier with smoothing
    source = _classifier(lr_schedule="cosine", weight_averaging=True)

    # When loading its state with hyperparameters into a plain one
    target = _classifier()
    target.from_state_dict(source.to_state_dict(), load_hyperparameters=True)

    # Then the training configuration is carried over
    assert (target.lr_schedule, target.weight_averaging) == ("cosine", True)


def test_mid_rank_ties_centres_a_feature_that_is_mostly_zero():
    # Given a feature that is 0 (its minimum) in 90 % of the rows
    x, y = _separable_data()
    x[:, 0] = np.abs(x[:, 0]) + 1.0
    x[: int(0.9 * len(x)), 0] = 0.0
    clamped, mid_ranked = _classifier(), _classifier(mid_rank_ties=True)

    # When fitting the quantile transform with and without mid-ranked ties
    for classifier in (clamped, mid_ranked):
        classifier._quantiles = classifier._fit_feature_transform(
            np.ascontiguousarray(x, dtype=np.float32), seed=0
        )
    z_clamped = clamped._transform_features(np.ascontiguousarray(x, dtype=np.float32))
    z_mid = mid_ranked._transform_features(np.ascontiguousarray(x, dtype=np.float32))

    # Then sklearn's rule puts the zeros at the clip value, the mid rank near the middle of their group
    assert z_clamped[0, 0] < -5.0
    assert abs(z_mid[0, 0]) < 0.2


@pytest.mark.parametrize(
    "options",
    [
        {"mid_rank_ties": True},
        {"input_batch_norm": True},
        {"mid_rank_ties": True, "input_batch_norm": True},
    ],
)
def test_options_fit_separate_and_survive_the_state_dict(options):
    # Given separable data
    x, y = _separable_data()
    classifier = _classifier(**options)

    # When fitting, and restoring the state into a plain classifier
    classifier.fit(x, y)
    restored = _classifier()
    restored.from_state_dict(classifier.to_state_dict())

    # Then the classes separate and the restored classifier predicts the same
    assert np.mean(classifier.predict(x) == y) > 0.95
    assert np.allclose(restored.predict_proba(x), classifier.predict_proba(x))

import pickle
from copy import deepcopy

import numpy as np
import pytest

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


def _data_with_a_mostly_zero_feature() -> tuple[np.ndarray, np.ndarray]:
    """Separable data whose first feature is 0, its minimum, in 90 % of the rows."""
    x, y = _separable_data()
    x[:, 0] = np.abs(x[:, 0]) + 1.0
    x[: int(0.9 * len(x)), 0] = 0.0
    return x, y


def _classifier(**kwargs) -> RustMLPClassifier:
    defaults = {"layers": [32, 16], "epochs": 5, "batch_size": 256, "random_state": 1}
    return RustMLPClassifier(**{**defaults, **kwargs})


def test_fit_separates_targets_and_decoys():
    # Given: separable data
    x, y = _separable_data()
    classifier = _classifier()

    # When: fitting and predicting
    classifier.fit(x, y)
    y_pred = classifier.predict(x)
    proba = classifier.predict_proba(x)

    # Then: nearly all rows are classified correctly and the probabilities are consistent
    assert np.mean(y_pred == y) > 0.95
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(y_pred, (proba[:, 1] > 0.5).astype(int))


def test_predict_before_fit_raises():
    # Given: an unfitted classifier
    x, _ = _separable_data(100)

    # When / Then: predicting raises
    with pytest.raises(ValueError, match="not been fitted"):
        _classifier().predict_proba(x)


def test_predict_with_wrong_number_of_features_raises():
    # Given: a fitted classifier
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)

    # When / Then: predicting on a different number of features raises
    with pytest.raises(ValueError, match="shape"):
        classifier.predict(x[:, :-1])


def test_state_dict_pickle_and_deepcopy_reproduce_predictions():
    # Given: a fitted classifier
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)
    expected = classifier.predict_proba(x)

    # When: restoring it from its state dict, a pickle and a deep copy
    restored = _classifier()
    restored.from_state_dict(classifier.to_state_dict())
    unpickled = pickle.loads(pickle.dumps(classifier))
    copied = deepcopy(classifier)

    # Then: all give the same probabilities
    for other in (restored, unpickled, copied):
        assert np.allclose(other.predict_proba(x), expected)


def test_from_state_dict_rejects_torch_state():
    # Given: a state dict of the torch classifier
    torch_state = {"_fitted": True, "input_dim": 3, "network_state_dict": {}}

    # When / Then: loading it raises
    with pytest.raises(ValueError, match="params"):
        _classifier().from_state_dict(torch_state)


def test_second_fit_continues_from_the_first_and_reset_starts_over():
    # Given: two classifiers with the same seed, fitted once
    x, y = _separable_data()
    warm, reset = _classifier(), _classifier()
    warm.fit(x, y)
    reset.fit(x, y)

    # When: resetting one of them
    reset.reset()

    # Then: the reset classifier is unfitted
    assert not reset.fitted

    # When: both classifiers are fitted again
    warm.fit(x, y)
    reset.fit(x, y)

    # Then: the networks differ, because one continues from its first fit and the other starts from
    # new random weights
    assert not np.array_equal(warm._params, reset._params)


def test_fit_takes_one_hot_labels():
    # Given: the same labels as a vector and as one-hot rows
    x, y = _separable_data()
    vector, one_hot = _classifier(), _classifier()

    # When: fitting with each
    vector.fit(x, y)
    one_hot.fit(x, np.eye(2)[y])

    # Then: the networks are the same
    assert np.array_equal(vector._params, one_hot._params)


def test_failed_fit_keeps_the_previous_state():
    # Given: a fitted classifier
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)
    quantiles, params = classifier._quantiles, classifier._params

    # When: a second fit fails in training
    classifier.layers = []
    with pytest.raises(ValueError, match="hidden layers"):
        classifier.fit(x[::2], y[::2])

    # Then: the quantiles and the network are those of the first fit
    assert classifier._quantiles is quantiles
    assert classifier._params is params


def test_changed_input_dimension_reinitializes_with_warning():
    # Given: a classifier fitted on 12 features
    x, y = _separable_data()
    classifier = _classifier()
    classifier.fit(x, y)

    # When: fitting on 11 features
    with pytest.warns(UserWarning, match="reinitialized"):
        classifier.fit(x[:, :-1], y)

    # Then: it predicts on 11 features
    assert classifier.input_dim == N_FEATURES - 1
    assert classifier.predict(x[:, :-1]).shape == (len(x),)


@pytest.mark.parametrize(
    ("feature_transform", "n_samples", "expected_shape"),
    [
        (FeatureTransform.QUANTILE, 6000, (N_FEATURES, 1001)),
        (FeatureTransform.QUANTILE, 500, None),
        (FeatureTransform.NONE, 6000, None),
    ],
)
def test_fit_uses_quantiles_only_for_the_quantile_transform_with_enough_rows(
    feature_transform, n_samples, expected_shape
):
    # Given: a classifier with the feature transform of the case, and the rows of the case
    x, y = _separable_data(n_samples)
    classifier = _classifier(feature_transform=feature_transform)

    # When: fitting
    classifier.fit(x, y)

    # Then: it stores the quantiles of every feature, or it stores no quantiles and uses the raw features
    quantiles = classifier.to_state_dict()["quantiles"]
    assert (None if quantiles is None else quantiles.shape) == expected_shape


def test_quantile_subsample_fits_the_quantiles_on_fewer_rows():
    # Given: data and a classifier with a quantile subsample
    x, _ = _separable_data()
    x = x.astype(np.float32)
    full, subsampled = _classifier(), _classifier(quantile_subsample=500)

    # When: fitting the quantiles
    q_full = full._fit_feature_transform(x, seed=0)
    q_subsampled = subsampled._fit_feature_transform(x, seed=0)

    # Then: the subsampled quantiles differ from the quantiles of all rows
    assert not np.array_equal(q_subsampled, q_full)


def test_unknown_feature_transform_raises():
    # When / Then: an unknown transform raises
    with pytest.raises(ValueError, match="Unknown feature transform"):
        RustMLPClassifier(feature_transform="unknown")


def test_hyperparameter_tuning_fits_small_data():
    # Given: hyperparameter tuning and a constant learning rate. For few rows, the tuning makes the
    # batch size smaller. The tuned rate is small, so with a cosine decay, 5 epochs on 3000 rows
    # train too little.
    x, y = _separable_data(3000)
    classifier = _classifier(
        experimental_hyperparameter_tuning=True,
        lr_schedule="constant",
        weight_averaging=False,
    )

    # When: fitting
    classifier.fit(x, y)

    # Then: the classifier separates the data
    assert np.mean(classifier.predict(x) == y) > 0.9


@pytest.mark.parametrize(
    ("lr_schedule", "weight_averaging"), [("cosine", False), ("constant", True)]
)
def test_configured_training_changes_the_fit(lr_schedule, weight_averaging):
    # Given: separable data and the plain (constant, last parameters) fit
    x, y = _separable_data()
    plain = _classifier(lr_schedule="constant", weight_averaging=False)
    plain.fit(x, y)

    # When: fitting with the configured smoothing, as FDRManager sets it
    smoothed = _classifier(lr_schedule="constant", weight_averaging=False)
    smoothed.configure_training(
        lr_schedule=lr_schedule, weight_averaging=weight_averaging
    )
    smoothed.fit(x, y)

    # Then: the network differs from the plain fit
    assert not np.array_equal(smoothed._params, plain._params)


def test_configure_training_keeps_unset_values_and_rejects_unknown_schedule():
    # Given: a classifier with the cosine schedule and without averaging
    classifier = _classifier(lr_schedule="cosine", weight_averaging=False)

    # When: only the averaging is configured
    classifier.configure_training(weight_averaging=True)

    # Then: the schedule is kept, and an unknown schedule raises
    assert (classifier.lr_schedule, classifier.weight_averaging) == ("cosine", True)
    with pytest.raises(ValueError, match="learning rate schedule"):
        classifier.configure_training(lr_schedule="linear")


def test_training_is_smoothed_by_default():
    # When: creating a classifier without training options
    classifier = _classifier()

    # Then: it trains with the cosine schedule and weight averaging
    assert (classifier.lr_schedule, classifier.weight_averaging) == ("cosine", True)


def test_state_dict_carries_the_training_configuration():
    # Given: a classifier with smoothing
    source = _classifier(lr_schedule="cosine", weight_averaging=True)

    # When: loading its state with hyperparameters into a plain one
    target = _classifier(lr_schedule="constant", weight_averaging=False)
    target.from_state_dict(source.to_state_dict(), load_hyperparameters=True)

    # Then: the training configuration is carried over
    assert (target.lr_schedule, target.weight_averaging) == ("cosine", True)


def test_mid_rank_ties_centres_a_feature_that_is_mostly_zero():
    # Given: a feature that is 0 (its minimum) in 90 % of the rows
    x, _ = _data_with_a_mostly_zero_feature()
    x = x.astype(np.float32)
    clamped, mid_ranked = _classifier(mid_rank_ties=False), _classifier()

    # When: the feature is transformed with the same quantiles, with and without mid-ranked ties
    quantiles = clamped._fit_feature_transform(x, seed=0)
    z_clamped = clamped._transform_features(x, quantiles)
    z_mid = mid_ranked._transform_features(x, quantiles)

    # Then: sklearn's rule puts the zeros at the clip value, and the mid rank puts them near the
    # middle of their group
    assert z_clamped[0, 0] < -5.0
    assert abs(z_mid[0, 0]) < 0.2


@pytest.mark.parametrize(
    "options", [{"mid_rank_ties": False}, {"input_batch_norm": True}]
)
def test_options_survive_the_state_dict(options):
    # Given: a classifier fitted with an option other than the default, on a feature with tied
    # minima
    x, y = _data_with_a_mostly_zero_feature()
    classifier = _classifier(**options)
    classifier.fit(x, y)

    # When: restoring its state into a default classifier
    restored = _classifier()
    restored.from_state_dict(classifier.to_state_dict())

    # Then: the restored classifier predicts the same
    assert np.allclose(restored.predict_proba(x), classifier.predict_proba(x))

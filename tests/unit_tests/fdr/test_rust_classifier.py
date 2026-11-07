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

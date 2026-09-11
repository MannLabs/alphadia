import numpy as np
import pytest
import torch

from alphadia.fdr.classifiers import (
    LOSS_BCE,
    LOSS_GCE,
    LOSS_NNPU,
    LOSS_SYMMETRIC,
    BinaryClassifierLegacyNewBatching,
    LightGBMClassifier,
    _class_weighted,
)


def _mixture(n: int = 4000, true_share: float = 0.25, seed: int = 0):
    """Targets are a mixture of true (shifted) and false rows; decoys look like the false rows."""
    rng = np.random.default_rng(seed)
    n_true = int(true_share * n)
    x = np.vstack(
        [
            rng.normal(2.0, 1.0, (n_true, 4)),
            rng.normal(0.0, 1.0, (n - n_true, 4)),
            rng.normal(0.0, 1.0, (n, 4)),
        ]
    )
    y = np.concatenate([np.zeros(n), np.ones(n)])
    is_true = np.concatenate([np.ones(n_true), np.zeros(2 * n - n_true)]).astype(bool)
    return x, y, is_true


def _network(loss: str) -> BinaryClassifierLegacyNewBatching:
    return BinaryClassifierLegacyNewBatching(
        test_size=0.1,
        batch_size=256,
        learning_rate=1e-3,
        epochs=3,
        loss=loss,
        random_state=0,
    )


def test_class_weighted_scales_the_decoys_by_twice_the_false_share():
    is_decoy = np.array([0, 1, 1, 0])

    weight = _class_weighted(np.array([1.0, 1.0, 2.0, 3.0]), is_decoy, 0.25)

    np.testing.assert_allclose(weight, [1.0, 1.5, 3.0, 3.0])


def test_class_weighted_leaves_the_nnpu_loss_to_apply_the_prior_itself():
    weight = _class_weighted(None, np.array([0, 1]), 0.25, LOSS_NNPU)

    np.testing.assert_allclose(weight, [1.0, 1.0])


@pytest.mark.parametrize("loss", [LOSS_BCE, LOSS_NNPU, LOSS_SYMMETRIC, LOSS_GCE])
def test_every_loss_separates_true_targets_with_a_class_prior(loss):
    x, y, is_true = _mixture()
    torch.manual_seed(0)
    classifier = _network(loss)

    classifier.fit(x, y, sample_weight=np.ones(len(y)), class_prior=0.25)
    proba = classifier.predict_proba(x)[:, 1]

    assert proba[is_true].mean() < proba[~is_true].mean() - 0.3


def test_nnpu_batch_loss_ascends_a_negative_positive_risk():
    classifier = _network(LOSS_NNPU)
    # every target row is scored as a sure target and the decoys as sure decoys, so the
    # targets' positive loss is far below the decoys' and the estimate goes negative
    y_pred = torch.tensor([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]])
    y_true = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    weight = torch.ones(4)

    loss = classifier._batch_loss(y_pred, y_true, weight, class_prior=0.2)

    pos_target = -np.log(0.99)
    pos_decoy = -np.log(0.01)
    assert loss.item() == pytest.approx(-(pos_target - 0.8 * pos_decoy))


def test_nnpu_batch_loss_without_a_prior_is_the_cross_entropy():
    classifier = _network(LOSS_NNPU)
    y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = classifier._batch_loss(y_pred, y_true, torch.ones(2), class_prior=None)

    assert loss.item() == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)


def test_batch_loss_weights_the_rows():
    classifier = _network(LOSS_BCE)
    y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = classifier._batch_loss(y_pred, y_true, torch.tensor([3.0, 1.0]), None)

    assert loss.item() == pytest.approx(-(3 * np.log(0.8) + np.log(0.7)) / 4)


def test_lightgbm_fit_takes_sample_weights():
    x, y, is_true = _mixture(n=1000)
    classifier = LightGBMClassifier(
        n_estimators=30, final_n_estimators=30, min_child_samples=5, random_state=0
    )

    classifier.fit(x, y, sample_weight=np.full(len(y), 0.5), class_prior=0.25)
    proba = classifier.predict_proba(x)[:, 1]

    assert proba[is_true].mean() < proba[~is_true].mean() - 0.3

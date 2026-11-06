import numpy as np
import pytest
import torch
from torch import optim

from alphadia.fdr.classifiers import (
    BinaryClassifierLegacyNewBatching,
    LRSchedule,
    _update_average_state,
)


def _data(rng: np.random.Generator, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, n).astype(float)
    x = rng.normal(0, 1, (n, 5)) + y[:, None]
    return x, y


def test_cosine_schedule_decays_the_learning_rate_to_zero():
    # given
    classifier = BinaryClassifierLegacyNewBatching(lr_schedule=LRSchedule.COSINE)
    optimizer = optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = classifier._lr_scheduler(optimizer, num_steps=10)

    # when
    for _ in range(10):
        optimizer.step()
        scheduler.step()

    # then
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_constant_schedule_keeps_the_learning_rate():
    # given
    classifier = BinaryClassifierLegacyNewBatching()
    optimizer = optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = classifier._lr_scheduler(optimizer, num_steps=10)

    # when
    for _ in range(10):
        optimizer.step()
        scheduler.step()

    # then
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="Unknown learning rate schedule"):
        BinaryClassifierLegacyNewBatching().configure_training(lr_schedule="linear")


def test_configure_training_leaves_unset_settings():
    # given
    classifier = BinaryClassifierLegacyNewBatching(
        lr_schedule=LRSchedule.COSINE, weight_averaging=True
    )

    # when
    classifier.configure_training(weight_averaging=False)

    # then
    assert classifier.lr_schedule == LRSchedule.COSINE
    assert classifier.weight_averaging is False


def test_average_state_is_the_running_mean_of_float_tensors():
    # given
    states = [
        {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(1)},
        {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(2)},
    ]

    # when
    averaged, count = None, 0
    for state in states:
        averaged, count = _update_average_state(averaged, count, state)

    # then
    assert count == 2
    assert torch.allclose(averaged["w"], torch.tensor([2.0, 4.0]))
    assert averaged["n"].item() == 2


def test_fit_with_cosine_and_averaging_separates_the_classes():
    # given
    rng = np.random.default_rng(0)
    x, y = _data(rng)
    classifier = BinaryClassifierLegacyNewBatching(
        input_dim=5,
        batch_size=200,
        epochs=3,
        learning_rate=0.01,
        random_state=1,
        lr_schedule=LRSchedule.COSINE,
        weight_averaging=True,
    )

    # when
    classifier.fit(x, y)
    proba = classifier.predict_proba(x)[:, 1]

    # then
    assert proba[y == 1].mean() > proba[y == 0].mean() + 0.3

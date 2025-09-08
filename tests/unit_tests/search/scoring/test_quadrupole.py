# global
import numpy as np
import pytest

from alphadia.search.plexscoring.quadrupole import (
    SimpleQuadrupole,
    logistic_rectangle,
    quadrupole_transfer_function_single,
)


@pytest.fixture()
def get_fake_quadrupole():
    fake_cycle = np.array([[25.0, 55.0], [15, 40]])
    fake_cycle = np.repeat(fake_cycle[:, np.newaxis, :], 10, axis=1)[
        np.newaxis, :, :, :
    ]

    return SimpleQuadrupole(fake_cycle)


def test_fit_quadrupole(get_fake_quadrupole):
    mz_train1 = np.concatenate(
        [np.random.uniform(20, 30, 20), np.random.uniform(45, 65, 20)]
    )
    precursor_train1 = np.zeros(40, dtype=np.int64)
    int_train1 = logistic_rectangle(25, 60, 0.5, 0.5, mz_train1) + np.random.normal(
        0, 0.05, 40
    )
    scan_train1 = np.random.randint(0, 4, 40)

    mz_train2 = np.concatenate(
        [np.random.uniform(10, 20, 20), np.random.uniform(35, 45, 20)]
    )
    precursor_train2 = np.ones(40, dtype=np.int64)
    int_train2 = logistic_rectangle(15, 45, 0.5, 0.5, mz_train2) + np.random.normal(
        0, 0.05, 40
    )
    scan_train2 = np.random.randint(0, 4, 40)

    mz_train = np.concatenate([mz_train1, mz_train2])
    precursor_train = np.concatenate([precursor_train1, precursor_train2])
    int_train = np.concatenate([int_train1, int_train2])
    scan_train = np.concatenate([scan_train1, scan_train2])

    get_fake_quadrupole.fit(precursor_train, scan_train, mz_train, int_train)


def test_predict(get_fake_quadrupole):
    test_mz = np.linspace(0, 70, 1000)

    for precursor in [0, 1]:
        test_scan = np.zeros(1000, dtype=np.int64)
        test_precursor = np.ones(1000, dtype=np.int64) * precursor

        get_fake_quadrupole.jit.predict(test_precursor, test_scan, test_mz)


def test_qtf():
    fake_cycle = np.array([[780.0, 801], [801, 820]])
    fake_cycle = np.repeat(fake_cycle[:, np.newaxis, :], 10, axis=1)[
        np.newaxis, :, :, :
    ]

    quad = SimpleQuadrupole(fake_cycle)

    isotope_mz = np.array([800.0, 800.1, 800.2, 802.42944, 802.9311, 803.1])
    observation_indices = np.array([0, 1])
    scan_indices = np.array(np.arange(2, 9))

    qtf = quadrupole_transfer_function_single(
        quad.jit, observation_indices, scan_indices, isotope_mz
    )

    assert qtf.shape == (6, 2, 7)
    assert np.all(qtf[:3, 0, :] > 0.9)
    assert np.all(qtf[:3:, 1, :] < 0.1)
    assert np.all(qtf[3:, 0, :] < 0.1)
    assert np.all(qtf[3:, 1, :] > 0.9)

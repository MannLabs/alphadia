import numpy as np
import pandas as pd
import pytest

from alphadia.fdr import fragment_features

B, Y = ord("b"), ord("y")


def test_weighted_correlations_weight_by_library_and_observed_shares():
    mz = np.array([[300.0, 400.0, 0.0]])
    library = np.array([[3.0, 1.0, 5.0]])
    observed = np.array([[1.0, 3.0, 7.0]])
    correlation = np.array([[1.0, 0.0, 0.9]])

    values = fragment_features.weighted_correlations(mz, library, observed, correlation)

    # the empty slot (m/z 0) counts neither in the weights nor in the correlations
    assert values[fragment_features.LIBRARY_WEIGHTED][0] == pytest.approx(0.75)
    assert values[fragment_features.OBSERVED_WEIGHTED][0] == pytest.approx(0.25)


def test_entropy_similarity_is_one_for_identical_spectra_and_zero_without_signal():
    present = np.ones((2, 3), dtype=bool)
    library = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    observed = np.array([[6.0, 4.0, 2.0], [0.0, 0.0, 0.0]])

    similarity = fragment_features.entropy_similarity(library, observed, present)

    assert similarity[0] == pytest.approx(1.0)
    assert similarity[1] == 0.0


def _fragments(n_rows: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    k = 6
    mz = np.tile(np.linspace(300.0, 800.0, k), (n_rows, 1))
    library = np.tile(np.arange(k, 0, -1, dtype=float), (n_rows, 1))
    observed = rng.random((n_rows, k)) * (rng.random((n_rows, k)) > 0.3)
    correlation = np.where(observed > 0, rng.uniform(-0.5, 1.0, (n_rows, k)), 0.0)
    ion_type = np.tile([B, Y, B, Y, Y, B], (n_rows, 1))
    number = np.tile([2, 3, 5, 8, 4, 9], (n_rows, 1))
    charge = np.tile([1, 1, 2, 1, 2, 1], (n_rows, 1))
    return (
        mz,
        library,
        observed,
        correlation,
        ion_type,
        number,
        charge,
        np.full(n_rows, 2.0),
    )


def test_compute_gives_every_feature_finite_for_every_row():
    rng = np.random.default_rng(0)
    fragments = _fragments(400, rng)
    rank = np.repeat([0, 1], 200)

    values = fragment_features.compute(fragments, rank, np.full(400, 550.0))

    assert set(values) == set(fragment_features.FEATURES)
    for name, column in values.items():
        assert column.shape == (400,), name
        assert np.isfinite(column).all(), name


def test_ion_fisher_rewards_a_correlating_fragment_noise_rarely_gives():
    # given: rank >= 1 rows (the noise model) where the slot-0 ion is never seen, and one rank-0 row where it correlates
    n = 200
    present = np.ones((n, 2), dtype=bool)
    observed = np.zeros((n, 2))
    observed[:, 1] = 1.0
    correlation = np.zeros((n, 2))
    correlation[:, 1] = 0.9
    observed[0, 0], correlation[0, 0] = 1.0, 0.9
    rank = np.ones(n, dtype=int)
    rank[0] = 0
    ion_type, number, charge = (
        np.tile([Y, B], (n, 1)),
        np.tile([6, 6], (n, 1)),
        np.ones((n, 2), dtype=int),
    )

    values = fragment_features.ion_features(
        present,
        present & (observed > 0),
        correlation,
        np.full((n, 2), 0.5),
        ion_type,
        number,
        charge,
        rank >= 1,
    )

    # then: the rare hit makes row 0 far more surprising than the rows with only the common hit
    assert values["ion_fisher"][0] > values["ion_fisher"][1] + 5
    assert values["ion_coherent_y_mid"][0] == 1
    assert values["ion_coherent_b_mid"][1] == 1


def test_serving_provider_serves_rows_by_candidate():
    fragments = (np.array([[1.0], [2.0], [3.0]]), np.array([[10.0], [20.0], [30.0]]))
    provider = fragment_features.serving_provider(np.array([7, 5, 9]), fragments)

    served = provider(pd.DataFrame({"_candidate_idx": [9, 7]}))

    np.testing.assert_array_equal(served[0], [[3.0], [1.0]])
    np.testing.assert_array_equal(served[1], [[30.0], [10.0]])
    with pytest.raises(KeyError):
        provider(pd.DataFrame({"_candidate_idx": [4]}))

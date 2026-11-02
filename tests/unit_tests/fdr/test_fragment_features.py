import numpy as np
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
    mz = np.tile(np.linspace(300.0, 800.0, k, dtype=np.float32), (n_rows, 1))
    library = np.tile(np.arange(k, 0, -1, dtype=np.float32), (n_rows, 1))
    observed = (rng.random((n_rows, k)) * (rng.random((n_rows, k)) > 0.3)).astype(
        np.float32
    )
    correlation = np.where(
        observed > 0, rng.uniform(-0.5, 1.0, (n_rows, k)), 0.0
    ).astype(np.float32)
    ion_type = np.tile(np.array([B, Y, B, Y, Y, B], dtype=np.uint8), (n_rows, 1))
    number = np.tile(np.array([2, 3, 5, 8, 4, 9], dtype=np.uint8), (n_rows, 1))
    charge = np.tile(np.array([1, 1, 2, 1, 2, 1], dtype=np.uint8), (n_rows, 1))
    return mz, library, observed, correlation, ion_type, number, charge


def test_compute_gives_every_feature_finite_for_every_row():
    rng = np.random.default_rng(0)
    fragments = _fragments(400, rng)
    rank = np.repeat([0, 1], 200)

    values = fragment_features.compute(fragments, rank)

    assert set(values) == set(fragment_features.FEATURES)
    for name, column in values.items():
        assert column.shape == (400,), name
        assert column.dtype == np.float32, name
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
    assert values["ion_fisher_lib"][0] > values["ion_fisher_lib"][1] + 2
    assert values["ion_coherent_y_mid"][0] == 1
    assert values["ion_coherent_y_mid"][1] == 0


def test_split_agreement_describes_only_the_masked_fragments():
    mask = np.array([[True, True, False]])
    library = np.array([[3.0, 1.0, 6.0]])
    observed = np.array([[1.0, 1.0, 1.0]])
    correlation = np.array([[0.9, 0.1, 0.9]])

    libw_corr, explained, n_coherent = fragment_features.split_agreement(
        mask, library, observed, correlation
    )

    assert libw_corr[0] == pytest.approx((3 * 0.9 + 1 * 0.1) / 4)
    assert explained[0] == pytest.approx(0.75)
    assert n_coherent[0] == 1

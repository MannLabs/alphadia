import numpy as np
import pandas as pd
import pytest

from alphadia.fdr import fragment_features


def test_weighted_correlations_weight_by_library_and_observed_shares():
    mz = np.array([[300.0, 400.0, 0.0]])
    library = np.array([[3.0, 1.0, 5.0]])
    observed = np.array([[1.0, 3.0, 7.0]])
    correlation = np.array([[1.0, 0.0, 0.9]])

    values = fragment_features.weighted_correlations(mz, library, observed, correlation)

    # the empty slot (m/z 0) counts neither in the weights nor in the correlations
    assert values[fragment_features.LIBRARY_WEIGHTED][0] == pytest.approx(0.75)
    assert values[fragment_features.OBSERVED_WEIGHTED][0] == pytest.approx(0.25)


def test_weighted_correlations_are_zero_without_signal():
    mz = np.array([[300.0, 400.0]])
    zeros = np.zeros((1, 2))

    values = fragment_features.weighted_correlations(mz, np.ones((1, 2)), zeros, zeros)

    assert values[fragment_features.OBSERVED_WEIGHTED][0] == 0.0
    assert values[fragment_features.LIBRARY_WEIGHTED][0] == 0.0


def test_serving_provider_serves_rows_by_candidate():
    fragments = (np.array([[1.0], [2.0], [3.0]]), np.array([[10.0], [20.0], [30.0]]))
    provider = fragment_features.serving_provider(np.array([7, 5, 9]), fragments)

    served = provider(pd.DataFrame({"_candidate_idx": [9, 7]}))

    np.testing.assert_array_equal(served[0], [[3.0], [1.0]])
    np.testing.assert_array_equal(served[1], [[30.0], [10.0]])
    with pytest.raises(KeyError):
        provider(pd.DataFrame({"_candidate_idx": [4]}))

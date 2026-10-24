import logging

import numpy as np
import pytest

from alphadia.outputtransform.quantification.library_anchored import (
    library_anchored_estimate,
)

# the fit itself is tested in alphadia_search_rs (src/quantification/tests.rs), these tests
# cover the conversion of the matrices


def test_library_anchored_estimate_returns_sorted_groups_with_their_runs():
    # Given two precursors in reversed order whose fragments follow the library pattern,
    # precursor 20 at twice the level in run2 and missing in run3
    library = np.array([1.0, 0.5, 0.25, 1.0, 0.5])
    group = np.array([20, 20, 20, 10, 10])
    totals = {20: [1000.0, 2000.0, 0.0], 10: [500.0, 500.0, 500.0]}
    intensity = np.array(
        [
            np.array(totals[g]) * fragment_library / library[group == g].sum()
            for fragment_library, g in zip(library, group, strict=True)
        ]
    )

    # When
    groups, quantity = library_anchored_estimate(intensity, library, group)

    # Then
    assert groups.tolist() == [10, 20]
    assert quantity.shape == (2, 3)
    np.testing.assert_allclose(np.power(2.0, quantity[0]), 500.0, rtol=1e-4)
    np.testing.assert_allclose(
        np.power(2.0, quantity[1, :2]), [1000.0, 2000.0], rtol=1e-4
    )
    assert np.isnan(quantity[1, 2])


def test_library_anchored_estimate_warns_about_fragments_without_library(caplog):
    # Given one fragment without a library intensity
    intensity = np.array([[100.0, 200.0], [50.0, 100.0], [10.0, 99.0]])

    # When
    with caplog.at_level(logging.WARNING):
        _, quantity = library_anchored_estimate(
            intensity, np.array([1.0, 0.5, np.nan]), np.zeros(3, dtype=int)
        )

    # Then
    assert any("without library intensity" in r.message for r in caplog.records)
    np.testing.assert_allclose(quantity[0, 1] - quantity[0, 0], 1.0, atol=1e-4)


def test_library_anchored_estimate_passes_the_weighting():
    # Given fragments following the library pattern
    library = np.array([1.0, 0.5, 0.25])
    intensity = np.outer(library, [100.0, 200.0])

    for weighting in ["variance", "share"]:
        # When
        _, quantity = library_anchored_estimate(
            intensity, library, np.zeros(3, dtype=int), weighting=weighting
        )

        # Then
        np.testing.assert_allclose(quantity[0, 1] - quantity[0, 0], 1.0, atol=1e-4)

    # Then an unknown weighting is rejected
    with pytest.raises(ValueError, match="weighting"):
        library_anchored_estimate(
            intensity, library, np.zeros(3, dtype=int), weighting="uniform"
        )

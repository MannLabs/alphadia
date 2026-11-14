import numpy as np
import pytest
from scipy import stats

from alphadia.calibration.tolerance import (
    posterior_error_probability,
    true_match_interval,
)


def _scores(rng, n_targets, n_decoys):
    """Classifier probabilities: true targets score high, decoys and false targets score low."""
    true_targets = rng.beta(8, 1, n_targets)
    decoys = rng.beta(1, 3, n_decoys)
    return np.concatenate([true_targets, decoys]), np.concatenate(
        [np.zeros(n_targets, bool), np.ones(n_decoys, bool)]
    )


def test_posterior_error_probability_falls_with_the_score():
    """Test that the PEP lies in [0, 1], is high where decoys score and falls as the score rises."""
    # Given: targets that score high and decoys that score low
    rng = np.random.default_rng(0)
    score, is_decoy = _scores(rng, 2000, 200)

    # When: the PEP is estimated
    pep = posterior_error_probability(score, is_decoy)

    # Then: it lies in [0, 1] and never rises with the score
    assert np.all((pep >= 0) & (pep <= 1))
    order = np.argsort(score)
    assert np.all(np.diff(pep[order]) <= 1e-12)
    assert pep[order][-1] < 0.01
    assert pep[order][0] > 0.5


def test_posterior_error_probability_is_zero_without_decoys():
    """Test that a set without decoys gives no evidence of false matches."""
    # Given: only targets
    score = np.linspace(0.1, 0.9, 50)

    # When: the PEP is estimated
    pep = posterior_error_probability(score, np.zeros(50, bool))

    # Then: every target is taken as a true match
    assert np.all(pep == 0)


def _residuals(rng, n, sigma, nu, window):
    """|residuals| and PEPs of n candidates: half have PEP 0.05, half 0.9, and each is a true match with probability
    1 - PEP. True matches follow a Student-t cut at the window, false matches are uniform in the window."""
    pep = np.where(rng.random(n) < 0.5, 0.05, 0.9)
    is_true = rng.random(n) > pep
    distribution = stats.t(nu, scale=sigma)
    upper = distribution.cdf(window)
    residuals = rng.uniform(0, window, n)
    residuals[is_true] = np.abs(
        distribution.ppf(rng.uniform(1 - upper, upper, is_true.sum()))
    )
    return residuals, pep


@pytest.mark.parametrize("n", [600, 6000])
def test_true_match_interval_recovers_the_interval_of_the_true_matches(n):
    """Test that the interval follows the true matches and ignores the uniform false matches."""
    # Given: true matches with a 99% interval of 4.03 * 20 s, mixed with false matches spread over a 300 s window
    rng = np.random.default_rng(1)
    residuals, pep = _residuals(rng, n, sigma=20, nu=5, window=300)
    expected = stats.t(5, scale=20).ppf(0.995)

    # When: the 99% interval of the true matches is estimated
    interval = true_match_interval(residuals, pep, window=300, quantile=0.99)

    # Then: it is within 20% of the interval of the true matches
    assert interval == pytest.approx(expected, rel=0.2)


def test_true_match_interval_does_not_shrink_with_the_window():
    """Test that a window that cuts the tail of the true matches does not shrink the interval estimate."""
    # Given: the candidates of the previous test, searched in a window 1.5 times the 99% interval of the true matches
    rng = np.random.default_rng(2)
    expected = stats.t(5, scale=20).ppf(0.995)
    residuals, pep = _residuals(rng, 6000, sigma=20, nu=5, window=120)

    # When: the 99% interval is estimated within that window
    interval = true_match_interval(residuals, pep, window=120, quantile=0.99)

    # Then: the estimate keeps the interval of the full distribution
    assert interval == pytest.approx(expected, rel=0.2)


def test_true_match_interval_needs_true_matches():
    """Test that no interval is estimated without residuals that are likely true matches."""
    # Given: two residuals that are almost surely false matches
    # When / Then: the estimate returns None
    assert (
        true_match_interval(np.array([10.0, 50.0]), np.array([1.0, 1.0]), 100, 0.99)
        is None
    )

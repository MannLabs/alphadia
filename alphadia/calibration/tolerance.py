"""Search tolerances from the calibrated residuals of true matches, weighted by the classifier."""

import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression

# The fit needs at least this expected number of true matches, the sum of 1 - PEP.
MIN_EXPECTED_TRUE_MATCHES = 3


def posterior_error_probability(score: np.ndarray, is_decoy: np.ndarray) -> np.ndarray:
    """Estimate the posterior error probability of each candidate from its classifier score.

    Target-decoy competition makes a false target as likely as a decoy at every score. With p(s) the probability that
    a candidate at score s is a decoy, the share of false targets among the targets at s is p / (1 - p). p(s) is a
    logistic function of the logit of the score, fitted to the decoy labels. It is smooth, never rises with the score
    and has two parameters, so it holds with the few decoys of a low-input search.

    Parameters
    ----------
    score : np.ndarray
        Classifier probability of each candidate being a target.
    is_decoy : np.ndarray
        Whether each candidate is a decoy.

    Returns
    -------
    np.ndarray
        PEP of each candidate in [0, 1]. Zero for every candidate when there are no decoys or no targets.

    """
    is_decoy = np.asarray(is_decoy, dtype=bool)
    if is_decoy.all() or not is_decoy.any():
        return np.zeros(len(is_decoy))
    score = np.clip(np.asarray(score, dtype=np.float64), 1e-9, 1 - 1e-9)
    logit = np.log(score / (1 - score)).reshape(-1, 1)
    model = LogisticRegression(C=1e6).fit(logit, is_decoy)
    if model.coef_[0, 0] > 0:
        # Decoys that score higher than targets give no evidence on the error rate.
        return np.ones(len(is_decoy))
    p_decoy = model.predict_proba(logit)[:, 1]
    return np.clip(p_decoy / np.maximum(1 - p_decoy, 1e-12), 0, 1)


def true_match_interval(
    residuals: np.ndarray, pep: np.ndarray, window: float, quantile: float
) -> float | None:
    """Estimate the central interval that holds `quantile` of the calibrated residuals of the true matches.

    Each residual is a true match with probability 1 - PEP and then follows a Student-t distribution centred on zero.
    Otherwise it is a false match, spread uniformly over the searched window. The scale and the degrees of freedom of
    the t distribution are fitted by maximum likelihood, with the t distribution normalised to the window. The
    estimate thus uses every candidate, weighted by the classifier, and does not shrink when the window cuts the tail.

    Parameters
    ----------
    residuals : np.ndarray
        Calibrated residuals of the target candidates.
    pep : np.ndarray
        Posterior error probability of each target candidate.
    window : float
        Tolerance that the residuals were searched with.
    quantile : float
        Share of the true matches that the interval holds.

    Returns
    -------
    float | None
        Half width of the interval, or None when the residuals hold too few likely true matches.

    """
    residuals = np.abs(np.asarray(residuals, dtype=np.float64))
    pep = np.clip(np.asarray(pep, dtype=np.float64), 1e-6, 1 - 1e-6)
    inside = residuals < window
    residuals, pep = residuals[inside], pep[inside]
    if np.sum(1 - pep) < MIN_EXPECTED_TRUE_MATCHES:
        return None

    def negative_log_likelihood(params: np.ndarray) -> float:
        scale, df = np.exp(params)
        distribution = stats.t(df, scale=scale)
        true_density = (
            2 * distribution.pdf(residuals) / (2 * distribution.cdf(window) - 1)
        )
        return -np.sum(np.log((1 - pep) * true_density + pep / window + 1e-300))

    order = np.argsort(residuals)
    cumulative_weight = np.cumsum(1 - pep[order])
    weighted_median = residuals[order][
        np.searchsorted(cumulative_weight, cumulative_weight[-1] / 2)
    ]
    start = np.log([max(weighted_median, window * 1e-6), 5.0])
    fit = optimize.minimize(negative_log_likelihood, start, method="Nelder-Mead")
    scale, df = np.exp(fit.x)
    return float(stats.t(df, scale=scale).ppf(0.5 + quantile / 2))

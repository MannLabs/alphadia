"""Variance-model precursor quantification with missing fragments as censored observations.

The variance model (`alphadia_search_rs.VarModelQuantification`) fits y = log2(I / share) of fragment f in run r as
q_r + o_f + noise, with a Cauchy-robust weight per cell from an intensity-dependent variance curve and a shrinkage
prior on the fragment offsets, and leaves fragments without signal out. The backend reports a fragment only when its
summed XIC carries signal, so a missing fragment of a quantified run says "below the detection limit". Leaving it out
reads every precursor near the limit too high in its low runs, which compresses ratios.

Here each run gets a dropout curve fitted on its own data,
    P(missing | mu) = pi + (1 - pi) * Phi((L - mu) / s),    mu = expected log2 fragment intensity,
and the fit maximises the observed-data likelihood: observed cells keep the variance model's robust weights, a missing
cell of a quantified run adds log P(missing | mu), entered as an EM pseudo-observation (its conditional mean under
the censored component, weight = posterior censoring probability / s^2), which has the exact score of that term.
Without censoring the fit is the variance model's (numpy port, same estimates).

Optionally each observed cell's variance also depends on the fragment's XIC correlation C in that run, as in QuantUMS
(Kistner et al.): variance * exp(gamma * (1 - sqrt(C))). A value with a poor correlation in one run (interference,
background) is down-weighted in that run only, instead of dropping the fragment in every run.
"""

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, logit

logger = logging.getLogger()

CAUCHY_TUNING_CONSTANT = 2.385
MAD_TO_SIGMA = 1.4826
VARIANCE_BINS = 30
MIN_VARIANCE_BIN_SIZE = 50
MIN_OFFSET_VARIANCE = 1e-4
MIN_ROBUST_SCALE = 1e-3
MAX_SWEEPS = 60
TOLERANCE = 1e-4
# a run quantity needs this many observed fragments
MIN_IONS = 2
# precursor runs with this many observed fragments have a trustworthy expected intensity for every fragment
CURVE_MIN_OBSERVED = 4
CURVE_ROUNDS = 2
CURVE_MAX_CELLS = 300_000
CURVE_SEED = 0
CURVE_START_QUANTILE = 0.02
CURVE_START_PLATEAU = 1e-3
PROBABILITY_FLOOR = 1e-12
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


@dataclass
class Padded:
    y: (
        np.ndarray
    )  # (precursors, fragments, runs), log2(I / share), NaN where missing or padding
    log_share: np.ndarray  # (precursors, fragments), NaN for padding


def pad_quality(
    correlation: np.ndarray,
    precursor_index: np.ndarray,
    n_precursors: int,
    n_fragments: int,
) -> np.ndarray:
    """1 - sqrt(clip(C, 0, 1)) per cell in the (precursors, fragments, runs) layout of `pad`, 1 for padding."""
    position = pd.Series(precursor_index).groupby(precursor_index).cumcount().to_numpy()
    t = np.ones((n_precursors, n_fragments, correlation.shape[1]))
    t[precursor_index, position] = 1 - np.sqrt(
        np.clip(np.nan_to_num(correlation), 0.0, 1.0)
    )
    return t


@dataclass
class Noise:
    sigma: float
    tau: float
    lam: float
    curve_x: np.ndarray
    curve_log_var: np.ndarray

    def variance(self, expected: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(expected, self.curve_x, self.curve_log_var))


@dataclass
class Dropout:
    limit: np.ndarray  # per run, log2 intensity
    width: np.ndarray
    plateau: np.ndarray


def pad(
    intensity: np.ndarray,
    intensity_library: np.ndarray,
    precursor_index: np.ndarray,
    n_precursors: int,
) -> Padded:
    """Fragment rows into a (precursors, fragments, runs) cube."""
    position = pd.Series(precursor_index).groupby(precursor_index).cumcount().to_numpy()
    n_fragments = int(position.max()) + 1
    share = (
        intensity_library
        / np.bincount(precursor_index, intensity_library, n_precursors)[precursor_index]
    )
    y = np.full((n_precursors, n_fragments, intensity.shape[1]), np.nan)
    with np.errstate(divide="ignore"):
        y[precursor_index, position] = np.where(
            intensity > 0, np.log2(intensity / share[:, None]), np.nan
        )
    log_share = np.full((n_precursors, n_fragments), np.nan)
    log_share[precursor_index, position] = np.log2(share)
    return Padded(y, log_share)


def weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    numerator = np.where(weights > 0, weights * np.nan_to_num(values), 0).sum(axis)
    denominator = weights.sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def robust_scale(values: np.ndarray) -> float:
    return max(
        MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))), MIN_ROBUST_SCALE
    )


def initial_quantities(p: Padded) -> np.ndarray:
    share = np.nan_to_num(np.exp2(p.log_share))[:, :, None]
    return weighted_mean(p.y, np.where(np.isfinite(p.y), share, 0), axis=1)


def estimate_noise(p: Padded, q0: np.ndarray) -> Noise:
    """Residual scale, variance curve over the expected intensity and offset shrinkage, as in the variance model."""
    observed = np.isfinite(p.y)
    usable = observed & (observed.sum(1) >= MIN_IONS)[:, None, :]
    share = np.nan_to_num(np.exp2(p.log_share))[:, :, None]
    share_sum = (share * usable).sum(1)[:, None, :]
    share_square_sum = (share**2 * usable).sum(1)[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = share / share_sum
        leverage = 1 - 2 * w + share_square_sum / share_sum**2
        deviation = p.y - q0[:, None, :]
        n = usable.sum(2)
        mean = np.where(usable, deviation, 0).sum(2) / n
        scale = np.sqrt(n / (n - 1))
        residual = (
            (deviation - mean[:, :, None]) / np.sqrt(leverage) * scale[:, :, None]
        )
    keep = usable & (n >= MIN_IONS)[:, :, None] & (leverage > 0)
    values = residual[keep]
    expected = (p.log_share[:, :, None] + q0[:, None, :])[keep]
    sigma = robust_scale(values)

    order = np.argsort(expected)
    expected, values = expected[order], values[order]
    n_bins = int(np.clip(len(values) // MIN_VARIANCE_BIN_SIZE, 1, VARIANCE_BINS))
    edges = (np.arange(n_bins + 1) * len(values)) // n_bins
    curve_x = np.array([expected[(a + b) // 2] for a, b in itertools.pairwise(edges)])
    curve_log_var = np.array(
        [2 * np.log(robust_scale(values[a:b])) for a, b in itertools.pairwise(edges)]
    )
    # a measurement is at least as noisy as any brighter one
    curve_log_var = np.maximum.accumulate(curve_log_var[::-1])[::-1]

    enough = n >= MIN_IONS
    noise_part = np.mean(sigma**2 / n[enough])
    lam = sigma**2 / max(mean[enough].var() - noise_part, MIN_OFFSET_VARIANCE)
    return Noise(sigma, CAUCHY_TUNING_CONSTANT * sigma, lam, curve_x, curve_log_var)


def censored_pseudo_observations(
    p: Padded, q: np.ndarray, o: np.ndarray, dropout: Dropout
) -> tuple[np.ndarray, np.ndarray]:
    """Value (y units) and weight of each missing cell's EM pseudo-observation under the censored component."""
    mu = q[:, None, :] + o[:, :, None] + p.log_share[:, :, None]
    s = dropout.width[None, None, :]
    with np.errstate(invalid="ignore", over="ignore"):
        z = (dropout.limit[None, None, :] - mu) / s
        log_cdf = log_ndtr(z)
        mills = np.exp(-0.5 * z**2 - LOG_SQRT_2PI - log_cdf)
        cdf = np.exp(log_cdf)
        pi = dropout.plateau[None, None, :]
        responsibility = (1 - pi) * cdf / (pi + (1 - pi) * cdf)
    # a run whose curve never censors (limit -inf, plateau 0) gives 0/0 here; its missing cells carry no
    # information, and a NaN would spread through the fragment offsets into the other runs
    weight = np.nan_to_num(responsibility / s**2)
    value = np.where(weight > 0, mu - s * mills - p.log_share[:, :, None], 0.0)
    return value, weight


def fit(
    p: Padded,
    noise: Noise,
    dropout: Dropout | None = None,
    log_variance_factor: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run quantities (log2, NaN with fewer than MIN_IONS observed fragments) and fragment offsets."""
    q0 = initial_quantities(p)
    loss = (noise.sigma / noise.tau) ** 2
    prior_weight = noise.lam / noise.sigma**2
    observed = np.isfinite(p.y)
    valid = np.isfinite(p.log_share)[:, :, None]
    quantified = observed.any(1)
    censored = valid & ~observed & quantified[:, None, :]
    # fixed at the share-weighted first pass, as in the variance model, so the weights cannot chase the fit
    variance = noise.variance(p.log_share[:, :, None] + q0[:, None, :]) * np.exp(
        log_variance_factor
    )
    y = np.where(observed, p.y, 0.0)
    q = np.nan_to_num(q0)
    o = np.zeros(p.log_share.shape)
    for _ in range(MAX_SWEEPS):
        e = y - q[:, None, :] - o[:, :, None]
        w = np.where(observed, 1 / (variance + loss * e**2), 0.0)
        values = y
        if dropout is not None:
            pseudo, pseudo_weight = censored_pseudo_observations(p, q, o, dropout)
            values = np.where(censored, pseudo, y)
            w = np.where(censored, pseudo_weight, w)
        o = (w * (values - q[:, None, :])).sum(2) / (w.sum(2) + prior_weight)
        new_q = weighted_mean(values - o[:, :, None], w, axis=1)
        new_q = np.where(quantified, np.nan_to_num(new_q), 0.0)
        change = np.abs(new_q - q).max()
        q = new_q
        if change < TOLERANCE:
            break
    return np.where(observed.sum(1) >= MIN_IONS, q, np.nan), o


def fit_dropout(p: Padded, q: np.ndarray, o: np.ndarray) -> Dropout:
    """Per-run maximum-likelihood dropout curve on the precursor runs whose expected intensities are trustworthy."""
    observed = np.isfinite(p.y)
    valid = np.isfinite(p.log_share)[:, :, None]
    trusted = (
        valid
        & (observed.sum(1) >= CURVE_MIN_OBSERVED)[:, None, :]
        & np.isfinite(q)[:, None, :]
    )
    mu = np.nan_to_num(q)[:, None, :] + o[:, :, None] + p.log_share[:, :, None]
    rng = np.random.default_rng(CURVE_SEED)
    limits, widths, plateaus = [], [], []
    for run in range(p.y.shape[2]):
        sel = trusted[:, :, run]
        m = mu[:, :, run][sel]
        miss = ~observed[:, :, run][sel]
        if miss.sum() == 0 or miss.all():
            # no missing (or no observed) fragments to fit a curve on: a curve that never censors
            limits.append(-np.inf)
            widths.append(1.0)
            plateaus.append(0.0)
            continue
        if len(m) > CURVE_MAX_CELLS:
            pick = rng.choice(len(m), CURVE_MAX_CELLS, replace=False)
            m, miss = m[pick], miss[pick]
        start = [
            np.quantile(m[~miss], CURVE_START_QUANTILE),
            0.0,
            logit(max(miss.mean() / 2, CURVE_START_PLATEAU)),
        ]

        def negative_log_likelihood(theta, m=m, miss=miss):
            limit, log_width, plateau_logit = theta
            pi = expit(plateau_logit)
            cdf = np.exp(log_ndtr((limit - m) / np.exp(log_width)))
            prob = np.clip(
                pi + (1 - pi) * cdf, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR
            )
            return -np.sum(np.where(miss, np.log(prob), np.log1p(-prob)))

        best = minimize(negative_log_likelihood, start, method="Nelder-Mead")
        limits.append(best.x[0])
        widths.append(np.exp(best.x[1]))
        plateaus.append(expit(best.x[2]))
    return Dropout(np.array(limits), np.array(widths), np.array(plateaus))


def censored_quantities(
    intensity: np.ndarray,
    intensity_library: np.ndarray,
    precursor_index: np.ndarray,
    n_precursors: int,
    censored: bool = True,
    correlation: np.ndarray | None = None,
    quality_exponent: float = 0.0,
) -> np.ndarray:
    """Precursor quantities (precursors, runs), log2, NaN where not quantified.

    Parameters
    ----------
    intensity : np.ndarray
        (fragments, runs) fragment intensities, 0 or NaN where missing.
    intensity_library : np.ndarray
        (fragments,) library intensities, > 0.
    precursor_index : np.ndarray
        (fragments,) precursor of each fragment, 0..n_precursors-1.
    n_precursors : int
        Number of precursors; one without fragments is not quantified.
    censored : bool
        Missing fragments of quantified runs as censored observations; False gives the plain variance model.
    correlation : np.ndarray | None
        (fragments, runs) XIC correlation of each fragment value; needed when quality_exponent > 0.
    quality_exponent : float
        gamma of the per-cell variance factor exp(gamma * (1 - sqrt(correlation))); 0 leaves it out.

    """
    p = pad(np.nan_to_num(intensity), intensity_library, precursor_index, n_precursors)
    log_variance_factor = 0.0
    if quality_exponent > 0:
        log_variance_factor = quality_exponent * pad_quality(
            correlation, precursor_index, n_precursors, p.y.shape[1]
        )
    noise = estimate_noise(p, initial_quantities(p))
    q, o = fit(p, noise, log_variance_factor=log_variance_factor)
    if censored:
        for _ in range(CURVE_ROUNDS):
            dropout = fit_dropout(p, q, o)
            q, o = fit(p, noise, dropout, log_variance_factor)
        logger.info(
            f"Censored variance model: detection limit {np.round(dropout.limit, 2)} log2, width "
            f"{np.round(dropout.width, 2)}, abundance-independent dropout {np.round(dropout.plateau, 4)}"
        )
    return q

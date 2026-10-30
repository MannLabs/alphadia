"""Final-round classifier features from the raw fragment traces of each candidate.

The rust FragmentTraceExport gives per candidate its riding fragment slots (riding.TOP_K_FRAGMENTS, non-zero library
intensity, no y1, by library intensity, m/z 0 = empty) over TRACE_WINDOW_CYCLES cycles: the scored cycles (columns
SCORED_START to SCORED_STOP, apex column APEX) and TRACE_EXTEND_CYCLES cycles on each side. Per slot and cycle it holds
the XIC, the intensity-weighted centroid m/z error, and the XICs of the fragment's M+1 isotope and of its half-spacing
companion (+C13/2 for a 1+ fragment: present when the matched peak is a 2+ ion of something else). A decoy is extracted
at its own fragment m/z like a false target, so the features describe the evidence of a target and of its decoy alike.

- mass: a log likelihood ratio of the fragment mass errors under an intensity-dependent precision against a uniform
  background over the tolerance, and the drift of the centroid across the peak.
- isotopes: co-elution of the M+1 isotope with the coherent fragments, deviation of the M+1 / M0 ratio from averagine,
  intensity at the half-spacing companion of 1+ fragments.
- peak: apex over flank intensity.
- window ion current at the apex, read in the isolation window(s) of the precursor m/z of the candidate's elution group
  target. A decoy is lighter than its target in most groups (inner decoys), so its own window's ion current would
  partly encode the decoy's composition.
"""

import logging
import time
import warnings
from collections.abc import Callable

import numpy as np

logger = logging.getLogger()

TRACE_EXTEND_CYCLES = 6
TRACE_WINDOW_CYCLES = 19
# the export also counts occupied m/z bins near each fragment at the apex; no feature uses them, and a zero width keeps
# that count cheap
TRACE_NEIGHBOUR_PPM = 0.0
# the trace columns of the scored cycles, of the apex, of the three cycles around it and of the flanks
SCORED_START, SCORED_STOP = 6, 13
APEX = 9
CENTRE = slice(8, 11)
FLANKS = np.r_[0:3, 16:19]
# a fragment co-elutes with its candidate above this correlation to the median profile of its fragments
COHERENT_CORRELATION = 0.5
# M+1 / M0 of an averagine fragment per Dalton of neutral mass (13C, 15N, 2H, 17O)
AVERAGINE_M1_PER_DA = 0.00053
MASS_PROTON = 1.007276
MIN_CYCLES_FOR_DRIFT = 2
# the precision fit: per bin of log intensity (INTENSITY_BINS quantile bins) a normal core with mean 0 on a uniform
# background, on the fragments of PRECISION_SAMPLE_ROWS random candidates
INTENSITY_BINS = 12
PRECISION_SAMPLE_ROWS = 333_333
EM_ITERATIONS = 50
EM_START_SHARE = 0.5
EM_START_SIGMA_PPM = 1.0
EM_MIN_SIGMA_PPM = 0.1
EM_SEED = 20261029
# the traces are exported and reduced in chunks of this many candidates: a chunk's arrays take ~1 GB
CHUNK = 250_000
MAX_COMPANION_RATIO = 2.0
EPS = 1e-9

MASS = ("t5_me_llr_sum", "t5_me_llr_libw", "t5_mz_drift_libw")
ISOTOPES = ("t5_iso_coh_share", "t5_iso_dev_libw", "t5_wrongz_libw")
PEAK = ("t5_snr_libw",)
WINDOW_TIC = "t5_log_tic_tw"
FEATURES = (*MASS, *ISOTOPES, *PEAK, WINDOW_TIC)

Precision = tuple[np.ndarray, np.ndarray]


def _scored(values: np.ndarray) -> np.ndarray:
    return values[:, :, SCORED_START:SCORED_STOP]


def _pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis; 0 where either side is constant."""
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    return np.where(den > EPS, (x * y).sum(-1) / np.maximum(den, EPS), 0.0)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = weights.sum(1)
    return np.where(total > 0, (values * weights).sum(1) / np.maximum(total, EPS), 0.0)


def fragment_errors(
    xic: np.ndarray, mz_error_ppm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted centroid error over the scored cycles (NaN without signal) and the summed scored intensity
    per fragment."""
    x = _scored(xic)
    e = np.nan_to_num(_scored(mz_error_ppm).astype(np.float32))
    total = x.sum(-1)
    mean = np.where(total > 0, (x * e).sum(-1) / np.maximum(total, EPS), np.nan)
    return mean, total


def sample_rows(n_rows: int) -> np.ndarray:
    """The sorted random rows on whose fragments the precision is fitted."""
    rng = np.random.default_rng(EM_SEED)
    return np.sort(
        rng.choice(n_rows, size=min(n_rows, PRECISION_SAMPLE_ROWS), replace=False)
    )


def fit_precision(
    error: np.ndarray, total: np.ndarray, tolerance_ppm: float
) -> Precision:
    """Bin centres of log10 fragment intensity and the sd of the fragment mass error in each bin, from the mean errors
    and summed intensities of `fragment_errors`. Label-blind: EM of a normal core (mean 0) on a uniform background over
    +-`tolerance_ppm`, over every fragment with signal."""
    keep = total > 0
    err, logi = error[keep], np.log10(total[keep])
    edges = np.quantile(logi, np.linspace(0, 1, INTENSITY_BINS + 1))
    sd = np.empty(INTENSITY_BINS)
    background = 1.0 / (2 * tolerance_ppm)
    for b in range(INTENSITY_BINS):
        e = err[(logi >= edges[b]) & (logi <= edges[b + 1])]
        share, sigma = EM_START_SHARE, EM_START_SIGMA_PPM
        for _ in range(EM_ITERATIONS):
            core = (
                share * np.exp(-0.5 * (e / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            )
            resp = core / (core + (1 - share) * background)
            share = resp.mean()
            sigma = max(
                np.sqrt((resp * e * e).sum() / max(resp.sum(), EPS)), EM_MIN_SIGMA_PPM
            )
        sd[b] = sigma
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, sd


def chunk_features(
    traces: dict[str, np.ndarray], precision: Precision, tolerance_ppm: float
) -> dict[str, np.ndarray]:
    """FEATURES of the candidates of one export chunk: the FragmentTraceExport arrays plus `reference_tic`, the window
    ion current at the apex at the candidate's reference precursor m/z."""
    xic = np.asarray(traces["xic"], dtype=np.float32)
    present = np.asarray(traces["fragment_mz"]) > 0
    lib = np.where(present, np.asarray(traces["library_intensity"], np.float32), 0.0)
    scored = _scored(xic)
    signal = scored.sum(-1) > 0
    libw = np.where(signal, lib, 0.0)
    out = {}

    # the reference profile is the median of the scale-normalised traces of the fragments with signal (as in the rust
    # scoring); a candidate without signal has an all-NaN slice and a flat reference
    norm = scored / np.maximum(scored.max(-1, keepdims=True), EPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_ref = np.nan_to_num(
            np.nanmedian(np.where(signal[:, :, None], norm, np.nan), axis=1)
        )
    coherent = signal & (
        _pearson(scored, median_ref[:, None, :]) > COHERENT_CORRELATION
    )

    flank = xic[:, :, FLANKS].mean(-1)
    snr = np.log1p(xic[:, :, CENTRE].mean(-1)) - np.log1p(flank)
    out["t5_snr_libw"] = _weighted_mean(snr, libw)

    mz_error = np.asarray(traces["mz_error_ppm"])
    err, total = fragment_errors(xic, mz_error)
    centres, sd = precision
    sigma = np.interp(np.log10(np.maximum(total, 1.0)), centres, sd)
    llr = (
        -0.5 * (err / sigma) ** 2
        - np.log(sigma * np.sqrt(2 * np.pi))
        + np.log(2 * tolerance_ppm)
    )
    llr = np.where(signal & np.isfinite(llr), llr, 0.0)
    out["t5_me_llr_sum"] = np.where(coherent, llr, 0.0).sum(1)
    out["t5_me_llr_libw"] = _weighted_mean(llr, libw)
    e = _scored(mz_error).astype(np.float32)
    has = np.isfinite(e) & (scored > 0)
    w = np.where(has, scored, 0.0)
    wsum = w.sum(-1)
    mean = (np.nan_to_num(e) * w).sum(-1) / np.maximum(wsum, EPS)
    var = (w * (np.nan_to_num(e) - mean[:, :, None]) ** 2).sum(-1) / np.maximum(
        wsum, EPS
    )
    drift_ok = has.sum(-1) >= MIN_CYCLES_FOR_DRIFT
    out["t5_mz_drift_libw"] = _weighted_mean(np.sqrt(var), np.where(drift_ok, lib, 0.0))

    iso = _scored(np.asarray(traces["isotope_xic"], dtype=np.float32))
    iso_corr = _pearson(scored, iso)
    n_coherent = coherent.sum(1)
    out["t5_iso_coh_share"] = np.where(
        n_coherent > 0,
        (coherent & (iso_corr > COHERENT_CORRELATION)).sum(1)
        / np.maximum(n_coherent, 1),
        0.0,
    )
    charge = np.asarray(traces["fragment_charge"]).astype(np.float32)
    mass = (np.asarray(traces["fragment_mz"]) - MASS_PROTON) * charge
    expected = np.log(np.maximum(AVERAGINE_M1_PER_DA * mass, EPS))
    iso_sum, frag_sum = iso.sum(-1), scored.sum(-1)
    both = (iso_sum > 0) & (frag_sum > 0)
    deviation = np.abs(
        np.log(np.maximum(iso_sum, EPS) / np.maximum(frag_sum, EPS)) - expected
    )
    out["t5_iso_dev_libw"] = _weighted_mean(
        np.where(both, deviation, 0.0), np.where(both, lib, 0.0)
    )
    companion = _scored(np.asarray(traces["companion_xic"], dtype=np.float32)).sum(-1)
    single = signal & (charge == 1)
    out["t5_wrongz_libw"] = _weighted_mean(
        np.minimum(companion / np.maximum(frag_sum, EPS), MAX_COMPANION_RATIO),
        np.where(single, lib, 0.0),
    )

    out[WINDOW_TIC] = np.log1p(np.asarray(traces["reference_tic"], dtype=np.float32))
    return {name: np.asarray(out[name], dtype=np.float32) for name in FEATURES}


def reference_mz(
    own_mz: np.ndarray,
    elution_group_idx: np.ndarray,
    decoy: np.ndarray,
    library_elution_group_idx: np.ndarray,
    library_decoy: np.ndarray,
    library_mz: np.ndarray,
) -> np.ndarray:
    """The precursor m/z at which each candidate reads the window ion current: its own for a target, its elution group
    target's for a decoy (its own if the group has no target in the library)."""
    is_target = library_decoy == 0
    groups, first = np.unique(library_elution_group_idx[is_target], return_index=True)
    target_mz = library_mz[is_target][first]
    position = np.clip(np.searchsorted(groups, elution_group_idx), 0, len(groups) - 1)
    found = (len(groups) > 0) & (groups[position] == elution_group_idx)
    mz = np.where(decoy == 0, own_mz, np.where(found, target_mz[position], own_mz))
    missing = int(((decoy != 0) & ~found).sum())
    if missing > 0:
        logger.info(
            f"{missing} decoy candidates without a library target read the window ion current at their own m/z"
        )
    return mz


def compute(
    n_rows: int,
    export: Callable[[np.ndarray], dict[str, np.ndarray]],
    tolerance_ppm: float,
) -> dict[str, np.ndarray]:
    """FEATURES of `n_rows` candidates. `export(rows)` returns the trace arrays (with `reference_tic`) of the candidates
    `rows`; it is called on a random sample for the precision fit, then chunk by chunk."""
    start_time = time.time()
    sample = sample_rows(n_rows)
    errors, totals = [], []
    for start in range(0, len(sample), CHUNK):
        traces = export(sample[start : start + CHUNK])
        error, total = fragment_errors(traces["xic"], traces["mz_error_ppm"])
        errors.append(error)
        totals.append(total)
    precision = fit_precision(
        np.concatenate(errors), np.concatenate(totals), tolerance_ppm
    )
    out = {name: np.empty(n_rows, dtype=np.float32) for name in FEATURES}
    for start in range(0, n_rows, CHUNK):
        rows = np.arange(start, min(start + CHUNK, n_rows))
        for name, values in chunk_features(
            export(rows), precision, tolerance_ppm
        ).items():
            out[name][rows] = values
    logger.info(
        f"Raw fragment features of {n_rows} candidates in {time.time() - start_time:.0f} s"
    )
    return out

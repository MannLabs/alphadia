"""MS1 mass calibration for the final-round MS1 features, fitted on confident target candidates only.

The rust path never calibrates the precursor m/z. A pre-pass extracts the MS1 isotopes of the best-scoring target
candidates at a wide tolerance; the candidates whose apex isotope pattern matches the library (STRICT_* rules) give
the mass error, which is modelled as an additive m/z curve plus an RT curve (binned medians, linearly interpolated,
constant outside the fitted range). The spread left after the correction sets the extraction tolerance.
"""

from dataclasses import dataclass

import numpy as np

# pre-pass candidates count if their apex carries the monoisotope, this many isotopes and this isotope cosine
STRICT_MIN_ISOTOPES = 3
STRICT_MIN_ISO_COSINE = 0.9
# errors further than this from the median of the strict candidates are other peaks in the wide window
TRIM_PPM = 4.0
MIN_CANDIDATES = 200
CURVE_BINS = 8
MAD_TO_SIGMA = 1.4826
TOLERANCE_SIGMAS = 4.0
TOLERANCE_MIN_PPM = 3.0
TOLERANCE_MAX_PPM = 15.0
FALLBACK_TOLERANCE_PPM = 10.0


@dataclass
class Ms1Calibration:
    """Expected MS1 mass error (ppm) as a function of precursor m/z and RT, and the extraction tolerance."""

    mz_knots: np.ndarray
    mz_offsets: np.ndarray
    rt_knots: np.ndarray
    rt_offsets: np.ndarray
    sigma_ppm: float
    tolerance_ppm: float
    num_candidates: int

    @classmethod
    def identity(
        cls, tolerance_ppm: float = FALLBACK_TOLERANCE_PPM
    ) -> "Ms1Calibration":
        zero = np.zeros(1)
        return cls(zero, zero, zero, zero, np.nan, tolerance_ppm, 0)

    def offset_ppm(self, mz: np.ndarray, rt: np.ndarray) -> np.ndarray:
        return np.interp(mz, self.mz_knots, self.mz_offsets) + np.interp(
            rt, self.rt_knots, self.rt_offsets
        )

    def describe(self) -> str:
        mz = ", ".join(
            f"{k:.0f}:{o:+.2f}"
            for k, o in zip(self.mz_knots, self.mz_offsets, strict=True)
        )
        rt = ", ".join(
            f"{k:.0f}:{o:+.2f}"
            for k, o in zip(self.rt_knots, self.rt_offsets, strict=True)
        )
        return (
            f"{self.num_candidates} strict target candidates, residual sd {self.sigma_ppm:.2f} ppm, "
            f"tolerance {self.tolerance_ppm:.2f} ppm; m/z curve [{mz}]; RT curve [{rt}]"
        )


def _binned_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(x, np.linspace(0, 1, CURVE_BINS + 1))
    bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, CURVE_BINS - 1)
    knots = np.array([np.median(x[bins == b]) for b in range(CURVE_BINS)])
    offsets = np.array([np.median(y[bins == b]) for b in range(CURVE_BINS)])
    return knots, offsets


def fit_ms1_calibration(
    mz: np.ndarray,
    rt: np.ndarray,
    ppm_signed: np.ndarray,
    n_isotopes: np.ndarray,
    log_m0_apex: np.ndarray,
    iso_cosine: np.ndarray,
) -> Ms1Calibration:
    """Fit on pre-pass candidates, all of them targets; falls back to no correction if too few are strict."""
    strict = (
        (log_m0_apex > 0)
        & (n_isotopes >= STRICT_MIN_ISOTOPES)
        & (iso_cosine >= STRICT_MIN_ISO_COSINE)
    )
    error = ppm_signed[strict].astype(np.float64)
    keep = np.abs(error - np.median(error)) < TRIM_PPM if strict.any() else strict[:0]
    if keep.sum() < MIN_CANDIDATES:
        return Ms1Calibration.identity()
    mz, rt, error = mz[strict][keep], rt[strict][keep], error[keep]

    mz_knots, mz_offsets = _binned_curve(mz, error)
    residual = error - np.interp(mz, mz_knots, mz_offsets)
    rt_knots, rt_offsets = _binned_curve(rt, residual)
    residual = residual - np.interp(rt, rt_knots, rt_offsets)
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
    tolerance = float(
        np.clip(TOLERANCE_SIGMAS * sigma, TOLERANCE_MIN_PPM, TOLERANCE_MAX_PPM)
    )
    return Ms1Calibration(
        mz_knots, mz_offsets, rt_knots, rt_offsets, sigma, tolerance, len(error)
    )

"""The fragment competition module contains functionality to maintain the exclusive assignment of signal to identifications.

The sweep is a numba port of `alphadia_search_rs.FragmentCompetition`, because the Rust
version hard-codes a minimum of 3 shared ions and this module needs 2.
"""

import logging
import warnings

import numba as nb
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from alphadia.constants.keys import CalibCols
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash

logger = logging.getLogger(__name__)

DEFAULT_RT_TOLERANCE_SECONDS = 3.0
DEFAULT_MASS_TOLERANCE_PPM = 15.0
# DIA-NN-like interference removal: a shuffled entrapment twin often shares only two ions
# with the real peptide whose peak it rides, so the Rust default of 3 lets it survive.
MIN_OVERLAPPING_FRAGMENTS = 2
# A worse PSM also loses if the ions a better PSM shares with it carry this fraction of its observed fragment
# intensity, even below MIN_OVERLAPPING_FRAGMENTS: its evidence is mostly the better PSM's signal. This is the
# claimed-intensity information of the CandidateContext features as a decoy-symmetric removal rule.
MIN_CLAIMED_INTENSITY_FRACTION = 0.3


@nb.njit(cache=True)
def _fragment_overlap(
    frag_mz_1: np.ndarray, frag_mz_2: np.ndarray, mass_tol_ppm: float
) -> int:
    count = 0
    for mz1 in frag_mz_1:
        for mz2 in frag_mz_2:
            if abs(mz1 - mz2) / mz1 * 1e6 < mass_tol_ppm:
                count += 1
    return count


def _window_index(precursor_mz: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Window of each precursor, window 0 if none contains it (as in the Rust version)."""
    lower = cycle[0, :, :, 0].min(axis=1)
    upper = cycle[0, :, :, 1].max(axis=1)
    inside = (precursor_mz[:, None] >= lower[None, :]) & (
        precursor_mz[:, None] < upper[None, :]
    )
    return np.where(inside.any(axis=1), inside.argmax(axis=1), 0)


@nb.njit(cache=True)
def _claimed_fraction(
    frag_mz_better: np.ndarray,
    frag_mz_worse: np.ndarray,
    intensity_worse: np.ndarray,
    mass_tol_ppm: float,
) -> float:
    total = intensity_worse.sum()
    if total <= 0:
        return 0.0
    claimed = 0.0
    for k, mz in enumerate(frag_mz_worse):
        for other in frag_mz_better:
            if abs(mz - other) / mz * 1e6 < mass_tol_ppm:
                claimed += intensity_worse[k]
                break
    return claimed / total


@nb.njit(cache=True)
def _sweep(
    order: np.ndarray,
    window_idx: np.ndarray,
    rt_observed: np.ndarray,
    frag_start_idx: np.ndarray,
    frag_stop_idx: np.ndarray,
    fragment_mz: np.ndarray,
    fragment_intensity: np.ndarray,
    rt_tol_seconds: float,
    mass_tol_ppm: float,
    min_overlap: int,
    min_claimed_fraction: float,
) -> np.ndarray:
    valid = np.ones(len(order), dtype=np.bool_)
    start = 0
    while start < len(order):
        stop = start
        while stop < len(order) and window_idx[order[stop]] == window_idx[order[start]]:
            stop += 1
        for i in range(start, stop):
            row_i = order[i]
            if not valid[row_i]:
                continue
            frags_i = fragment_mz[frag_start_idx[row_i] : frag_stop_idx[row_i]]
            # only a better PSM removes a worse one: the claimed-intensity rule is not symmetric
            for j in range(i + 1, stop):
                row_j = order[j]
                if not valid[row_j]:
                    continue
                if abs(rt_observed[row_i] - rt_observed[row_j]) >= rt_tol_seconds:
                    continue
                frags_j = fragment_mz[frag_start_idx[row_j] : frag_stop_idx[row_j]]
                overlap = _fragment_overlap(frags_i, frags_j, mass_tol_ppm)
                if overlap >= min_overlap or (
                    overlap > 0
                    and _claimed_fraction(
                        frags_i,
                        frags_j,
                        fragment_intensity[
                            frag_start_idx[row_j] : frag_stop_idx[row_j]
                        ],
                        mass_tol_ppm,
                    )
                    >= min_claimed_fraction
                ):
                    valid[row_j] = False
        start = stop
    return valid


def compete_for_fragments(
    psm_df: pd.DataFrame,
    frag_df: pd.DataFrame,
    cycle: np.ndarray,
    rt_tol_seconds: float = DEFAULT_RT_TOLERANCE_SECONDS,
    mass_tol_ppm: float = DEFAULT_MASS_TOLERANCE_PPM,
    min_overlap: int = MIN_OVERLAPPING_FRAGMENTS,
    min_claimed_fraction: float | None = MIN_CLAIMED_INTENSITY_FRACTION,
) -> pd.DataFrame:
    """Remove PSMs that share fragments with other PSMs.

    PSMs only compete against others in the same DIA window and within
    `rt_tol_seconds`, and the lower `proba` wins. Row order does not matter and is
    preserved.

    Parameters
    ----------
    psm_df: pd.DataFrame
        The PSM dataframe.

    frag_df: pd.DataFrame
        The fragment dataframe.

    cycle: np.ndarray
        DIA cycle.

    rt_tol_seconds: float
        The retention time tolerance in seconds.

    mass_tol_ppm: float
        The mass tolerance in ppm.

    min_overlap: int
        Two PSMs compete if they share at least this number of ions.

    min_claimed_fraction: float | None
        A worse PSM also loses if the ions it shares with a better one carry at least this fraction of its observed
        fragment intensity; None switches the rule off. Without an `intensity` column all fragments weigh the same.

    Returns
    -------
    pd.DataFrame
        The PSM dataframe, reduced to the PSMs that won their fragments.

    """
    # TODO: this method raises SettingWithCopyWarning. Resolve without increasing memory usage.

    warnings.simplefilter(action="ignore", category=(SettingWithCopyWarning))

    psm_df["_candidate_idx"] = candidate_hash(
        psm_df["precursor_idx"].values, psm_df["rank"].values
    )
    frag_df["_candidate_idx"] = candidate_hash(
        frag_df["precursor_idx"].values, frag_df["rank"].values
    )

    psm_df = add_frag_start_stop_idx(psm_df, frag_df)

    window_idx = _window_index(
        psm_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
        np.asarray(cycle, dtype=np.float32),
    )
    proba = psm_df["proba"].to_numpy(dtype=np.float64)
    precursor_idx = psm_df["precursor_idx"].to_numpy(dtype=np.int64)
    # window first, then the best (lowest) proba; precursor_idx breaks ties
    order = np.lexsort((precursor_idx, proba, window_idx))
    valid = _sweep(
        order,
        window_idx,
        psm_df[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
        psm_df["_frag_start_idx"].to_numpy(dtype=np.int64),
        psm_df["_frag_stop_idx"].to_numpy(dtype=np.int64),
        frag_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
        frag_df["intensity"].to_numpy(dtype=np.float32)
        if "intensity" in frag_df.columns
        else np.ones(len(frag_df), dtype=np.float32),
        rt_tol_seconds,
        mass_tol_ppm,
        min_overlap,
        np.inf if min_claimed_fraction is None else min_claimed_fraction,
    )
    logger.info(
        f"fragment competition (min {min_overlap} shared ions or {min_claimed_fraction} claimed intensity): {int((~valid).sum()):,} of {len(valid):,} PSMs removed"
    )

    # these only exist to point the sweep at each PSM's fragments
    psm_df.drop(columns=["_frag_start_idx", "_frag_stop_idx"], inplace=True)

    warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
    return psm_df[valid]

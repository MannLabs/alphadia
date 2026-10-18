"""The fragment competition module contains functionality to maintain the exclusive assignment of signal to identifications.

The sweep is a numba port of `alphadia_search_rs.FragmentCompetition`, because the Rust
version lets targets and decoys compete with each other. Here targets compete only with
targets and decoys only with decoys, as in DIA-NN's removal of interfering precursors.
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
MIN_OVERLAPPING_FRAGMENTS = 3


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


def _orphaned_decoys(
    elution_group_idx: np.ndarray, is_decoy: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Decoys whose elution group had targets in the competition and lost all of them.

    A target that loses its fragments to a better target leaves the target-decoy competition. Its decoy would then
    win the elution group without a competitor, so the decoys would count more nulls than the false targets that are
    left. Removing the decoy with its targets keeps every elution group either complete or out.
    """
    targets = ~is_decoy
    had_target = np.isin(elution_group_idx, elution_group_idx[targets])
    kept_target = np.isin(elution_group_idx, elution_group_idx[targets & valid])
    return is_decoy & had_target & ~kept_target


def _window_index(precursor_mz: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Window of each precursor, window 0 if none contains it (as in the Rust version)."""
    lower = cycle[0, :, :, 0].min(axis=1)
    upper = cycle[0, :, :, 1].max(axis=1)
    inside = (precursor_mz[:, None] >= lower[None, :]) & (
        precursor_mz[:, None] < upper[None, :]
    )
    return np.where(inside.any(axis=1), inside.argmax(axis=1), 0)


@nb.njit(cache=True)
def _sweep(
    order: np.ndarray,
    window_idx: np.ndarray,
    rt_observed: np.ndarray,
    frag_start_idx: np.ndarray,
    frag_stop_idx: np.ndarray,
    fragment_mz: np.ndarray,
    rt_tol_seconds: float,
    mass_tol_ppm: float,
    min_overlap: int,
    is_decoy: np.ndarray,
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
            for j in range(i + 1, stop):
                row_j = order[j]
                # DIA-NN does not drop a decoy for a better target or a target for a better decoy
                # (measured on its own reports). A decoy that loses its fragments to a co-eluting real
                # peptide leaves the null weaker than the false targets that ride the same peaks.
                if not valid[row_j] or is_decoy[row_j] != is_decoy[row_i]:
                    continue
                if abs(rt_observed[row_i] - rt_observed[row_j]) >= rt_tol_seconds:
                    continue
                frags_j = fragment_mz[frag_start_idx[row_j] : frag_stop_idx[row_j]]
                if _fragment_overlap(frags_i, frags_j, mass_tol_ppm) >= min_overlap:
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
) -> pd.DataFrame:
    """Remove PSMs that share fragments with other PSMs.

    PSMs only compete against others in the same DIA window and within
    `rt_tol_seconds`, and the lower `proba` wins. Targets compete only with targets and decoys
    (`_decoy` == 1) only with decoys. A decoy also leaves when its elution group loses all its targets.
    Row order does not matter and is preserved.

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
    # `_decoy` is the decoy flag of the target-decoy competition; without it every PSM is a target
    is_decoy = (
        psm_df["_decoy"].to_numpy() == 1
        if "_decoy" in psm_df.columns
        else np.zeros(len(psm_df), dtype=bool)
    )
    valid = _sweep(
        order,
        window_idx,
        psm_df[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
        psm_df["_frag_start_idx"].to_numpy(dtype=np.int64),
        psm_df["_frag_stop_idx"].to_numpy(dtype=np.int64),
        frag_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
        rt_tol_seconds,
        mass_tol_ppm,
        min_overlap,
        is_decoy,
    )
    n_removed = int((~valid).sum())
    if "elution_group_idx" in psm_df.columns:
        valid &= ~_orphaned_decoys(
            psm_df["elution_group_idx"].to_numpy(), is_decoy, valid
        )
    logger.info(
        f"fragment competition (min {min_overlap} shared ions, within targets and within decoys): {n_removed:,} of "
        f"{len(valid):,} PSMs removed, plus {int((~valid).sum()) - n_removed:,} decoys whose elution group lost all its targets"
    )

    # these only exist to point the sweep at each PSM's fragments
    psm_df.drop(columns=["_frag_start_idx", "_frag_stop_idx"], inplace=True)

    warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
    return psm_df[valid]

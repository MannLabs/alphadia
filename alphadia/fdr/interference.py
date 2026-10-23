"""Removal of target candidates whose signal is explained by a better co-eluting target."""

import numba as nb
import numpy as np
import pandas as pd

from alphadia.constants.keys import CalibCols

# Winners are the targets at this competed q-value of the first fit; only they can remove other targets.
WINNER_QVAL = 0.01
FRAGMENT_PPM = 10.0
ISOBARIC_PPM = 10.0
ISOTOPE_SPACING = 1.00335
ISOTOPE_OFFSETS = np.array([-1, 0, 1], dtype=np.int64)
# b1/b2/y1/y2 are shared by almost every pair of co-eluting peptides, so they are no evidence of interference.
MIN_ION_NUMBER = 3
MIN_SHARED_FRAGMENTS = 1
# a candidate stops looking for further winners once this many partners have been checked
MAX_PARTNERS = 64

SEQUENCE_ID_COLUMN = "_sequence_id"
REQUIRED_COLUMNS = (
    "precursor_idx",
    "rank",
    "charge",
    SEQUENCE_ID_COLUMN,
    "fwhm_rt",
    CalibCols.RT_OBSERVED,
    CalibCols.MZ_OBSERVED,
)


@nb.njit(cache=True)
def _mz_close(
    mz: float,
    charge: int,
    winner_mz: float,
    offsets: np.ndarray,
    ppm: float,
    spacing: float,
) -> bool:
    for offset in offsets:
        if abs(mz + offset * spacing / charge - winner_mz) <= ppm * 1e-6 * winner_mz:
            return True
    return False


@nb.njit(parallel=True, cache=True)
def _find_interfered(  # noqa: PLR0913 # one array per candidate and winner attribute
    cand_frag_start: np.ndarray,
    cand_frag_mz: np.ndarray,
    cand_rt: np.ndarray,
    cand_fwhm: np.ndarray,
    cand_sequence: np.ndarray,
    cand_proba: np.ndarray,
    cand_mz: np.ndarray,
    cand_charge: np.ndarray,
    winner_frag_mz: np.ndarray,
    winner_frag_owner: np.ndarray,
    winner_rt: np.ndarray,
    winner_fwhm: np.ndarray,
    winner_sequence: np.ndarray,
    winner_proba: np.ndarray,
    winner_mz: np.ndarray,
    min_shared: int,
    fragment_ppm: float,
    offsets: np.ndarray,
    isobaric_ppm: float,
    spacing: float,
    max_partners: int,
) -> np.ndarray:
    n = cand_rt.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in nb.prange(n):
        partners = np.empty(max_partners, dtype=np.int64)
        shared = np.zeros(max_partners, dtype=np.int64)
        last_fragment = np.full(max_partners, -1, dtype=np.int64)
        n_partners = 0
        for f in range(cand_frag_start[i], cand_frag_start[i + 1]):
            x = cand_frag_mz[f]
            lo = np.searchsorted(winner_frag_mz, x * (1 - fragment_ppm * 1e-6))
            hi = np.searchsorted(winner_frag_mz, x * (1 + fragment_ppm * 1e-6))
            for t in range(lo, hi):
                w = winner_frag_owner[t]
                if winner_sequence[w] == cand_sequence[i]:
                    continue
                if winner_proba[w] >= cand_proba[i]:
                    continue
                if abs(cand_rt[i] - winner_rt[w]) > max(cand_fwhm[i], winner_fwhm[w]):
                    continue
                slot = -1
                for u in range(n_partners):
                    if partners[u] == w:
                        slot = u
                        break
                if slot < 0:
                    if n_partners == max_partners:
                        continue
                    slot = n_partners
                    partners[slot] = w
                    n_partners += 1
                if last_fragment[slot] != f:
                    last_fragment[slot] = f
                    shared[slot] += 1
                    if shared[slot] >= min_shared and _mz_close(
                        cand_mz[i],
                        cand_charge[i],
                        winner_mz[w],
                        offsets,
                        isobaric_ppm,
                        spacing,
                    ):
                        out[i] = True
                        break
            if out[i]:
                break
    return out


def _fragments_by_candidate(
    rows: pd.DataFrame, fragments: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """CSR layout (start offsets, m/z) of each row's library fragments with ion number >= MIN_ION_NUMBER."""
    usable = fragments[fragments["number"].to_numpy() >= MIN_ION_NUMBER]
    keys = pd.MultiIndex.from_arrays([usable["precursor_idx"], usable["rank"]])
    row_keys = pd.MultiIndex.from_arrays([rows["precursor_idx"], rows["rank"]])
    position = row_keys.get_indexer(keys)
    found = position >= 0
    position = position[found]
    mz = usable["mz_library"].to_numpy(np.float64)[found]
    order = np.argsort(position, kind="stable")
    counts = np.bincount(position, minlength=len(rows))
    start = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    return start, mz[order]


def find_interfered_targets(
    candidates: pd.DataFrame,
    winners: pd.DataFrame,
    fragments: pd.DataFrame,
) -> np.ndarray:
    """Flag the `candidates` whose signal a better co-eluting target of `winners` explains.

    A candidate is flagged when a winner of another peptide sequence with a lower `proba` (P(decoy)) elutes within
    the larger of both fwhm_rt, shares at least MIN_SHARED_FRAGMENTS library fragments of ion number >=
    MIN_ION_NUMBER within FRAGMENT_PPM, and has its precursor m/z within ISOBARIC_PPM of the candidate's at an
    isotope offset of ISOTOPE_OFFSETS. Both frames need the columns of REQUIRED_COLUMNS and `proba`; `fragments`
    needs precursor_idx, rank, number and mz_library for the rows of both.
    """
    cand_start, cand_mz = _fragments_by_candidate(candidates, fragments)
    win_start, win_mz = _fragments_by_candidate(winners, fragments)
    win_owner = np.repeat(np.arange(len(winners)), np.diff(win_start))
    order = np.argsort(win_mz, kind="stable")
    return _find_interfered(
        cand_start,
        cand_mz,
        candidates[CalibCols.RT_OBSERVED].to_numpy(np.float64),
        candidates["fwhm_rt"].to_numpy(np.float64),
        candidates[SEQUENCE_ID_COLUMN].to_numpy(np.int64),
        candidates["proba"].to_numpy(np.float64),
        candidates[CalibCols.MZ_OBSERVED].to_numpy(np.float64),
        candidates["charge"].to_numpy(np.int64),
        win_mz[order],
        win_owner[order].astype(np.int64),
        winners[CalibCols.RT_OBSERVED].to_numpy(np.float64),
        winners["fwhm_rt"].to_numpy(np.float64),
        winners[SEQUENCE_ID_COLUMN].to_numpy(np.int64),
        winners["proba"].to_numpy(np.float64),
        winners[CalibCols.MZ_OBSERVED].to_numpy(np.float64),
        MIN_SHARED_FRAGMENTS,
        FRAGMENT_PPM,
        ISOTOPE_OFFSETS,
        ISOBARIC_PPM,
        ISOTOPE_SPACING,
        MAX_PARTNERS,
    )

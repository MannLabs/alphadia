"""Removal of candidates that ride on the fragment signal of a better co-eluting candidate."""

import numba as nb
import numpy as np
import pandas as pd

from alphadia.constants.keys import CalibCols

# Partners that can explain a candidate: the best candidate of an elution group whose proba is at most the proba of the
# last target at this competed q-value (targets and decoys alike). A q 1 % pool misses a third of the interfered false
# targets a q 5 % pool finds at the same loss of real ones; beyond that the pool fills with false partners.
WINNER_QVAL = 0.05
# Partners co-elute when their apexes are within this many of the larger of both fwhm_rt.
CO_ELUTION_FWHM = 2.0
# A fragment is shared when its library m/z is within this many MS2 tolerances of the partner's: the traces of two
# fragments are the same signal up to one tolerance and unrelated beyond 1.25.
SHARED_TOLERANCE_FACTOR = 1.0
# A fragment co-elutes with its candidate when its profile correlates at least this much with the median profile.
COHERENT_CORRELATION = 0.5
# A candidate is removed when partners explain at least this share of its fragment evidence ... Set where the removal
# of interfered false targets (ARATH entrapment, 5 ng) costs 3 % of the real ones: below ~0.1 the real losses grow
# faster than the false removals.
MIN_RIDING = 0.11
# ... and at most this share of the library intensity of its unshared fragments co-elutes.
MAX_OWN_SUPPORT = 1.0
# Only PSMs up to this pooled q-value of the first fit can be removed (None: all), which bounds the PSMs whose
# fragments have to be quantified.
LOSER_POOLED_QVAL = None
# Partners of either class explain a PSM; otherwise only partners of its own class (targets explain targets, decoys
# explain decoys). At a low MIN_RIDING the class-blind rule removes far more decoys than false targets (decoys ride
# real peaks more often), which inflates the identifications; the within-class rule keeps the decoys a fair model.
CLASS_BLIND = False
TOP_K_FRAGMENTS = 12
# a riding provider returns the library m/z, library intensity, observed intensity and correlation first
FRAGMENT_ARRAYS = 4
# The classifier fit that ranks the candidates for the removal uses at most this many rows: on a 5 ng file it picks
# the same winners and removes the same false targets as a fit on all rows (300 k rows did not, a logistic regression
# found no confident targets).
PRESCORE_ROWS = 1_000_000
PPM = 1e-6

# other charge states of the same peptide share its fragments by construction, so they never explain each other
SEQUENCE_ID_COLUMN = "_sequence_id"
REQUIRED_COLUMNS = (
    "precursor_idx",
    SEQUENCE_ID_COLUMN,
    "rank",
    "elution_group_idx",
    "fwhm_rt",
    CalibCols.RT_OBSERVED,
    CalibCols.MZ_OBSERVED,
)


def fragment_evidence(
    observed: np.ndarray, correlation: np.ndarray, library: np.ndarray
) -> np.ndarray:
    """Evidence of each fragment (rows: candidates, columns: top library fragments): the observed intensity, capped at
    what the candidate's library pattern predicts, weighted by the fragment's co-elution with the candidate.

    The library scale is the least-squares fit of the observed to the library intensities over the coherent
    fragments; a fragment that also carries a foreign peak cannot claim more than the prediction.
    """
    coherent = (correlation > COHERENT_CORRELATION) & (library > 0)
    weighted_library = np.where(coherent, library, 0.0)
    scale = (weighted_library * observed).sum(1) / np.maximum(
        (weighted_library * library).sum(1), 1e-12
    )
    return np.minimum(observed, scale[:, None] * library) * np.clip(
        correlation, 0, None
    )


@nb.njit(parallel=True, cache=True)
def _riding(  # noqa: PLR0913, C901, PLR0912 # one array per candidate attribute; the partner loop is one kernel
    window: np.ndarray,
    rt: np.ndarray,
    fwhm: np.ndarray,
    elution_group: np.ndarray,
    sequence: np.ndarray,
    decoy: np.ndarray,
    proba: np.ndarray,
    mz: np.ndarray,
    evidence: np.ndarray,
    library: np.ndarray,
    correlation: np.ndarray,
    partner_rows: np.ndarray,
    partner_start: np.ndarray,
    partner_stop: np.ndarray,
    tolerance_ppm: float,
    class_blind: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray]:
    n, n_frag = evidence.shape
    riding = np.zeros(n, dtype=np.float32)
    own_support = np.ones(n, dtype=np.float32)
    for r in nb.prange(n):
        w = window[r]
        if w < 0:
            continue
        total = 0.0
        for f in range(n_frag):
            total += evidence[r, f]
        if total <= 0:
            continue
        claim = np.zeros(n_frag)
        any_shared = np.zeros(n_frag, dtype=np.bool_)
        match = np.full(n_frag, -1, dtype=np.int64)
        for j in range(partner_start[w], partner_stop[w]):
            p = partner_rows[j]
            if (
                p == r
                or elution_group[p] == elution_group[r]
                or sequence[p] == sequence[r]
                or proba[p] >= proba[r]
                or (not class_blind and decoy[p] != decoy[r])
            ):
                continue
            if abs(rt[p] - rt[r]) > CO_ELUTION_FWHM * max(fwhm[p], fwhm[r]):
                continue
            n_shared = 0
            for f in range(n_frag):
                match[f] = -1
                if mz[r, f] <= 0:
                    continue
                for g in range(n_frag):
                    if (
                        mz[p, g] > 0
                        and abs(mz[p, g] - mz[r, f]) <= tolerance_ppm * PPM * mz[r, f]
                    ):
                        match[f] = g
                        n_shared += 1
                        break
            if n_shared == 0:
                continue
            # the partner counts only if its own (unshared) fragments carry more evidence than the candidate's
            own_r = 0.0
            for f in range(n_frag):
                if match[f] < 0:
                    own_r += evidence[r, f]
            own_p = 0.0
            for g in range(n_frag):
                used = False
                for f in range(n_frag):
                    if match[f] == g:
                        used = True
                        break
                if not used:
                    own_p += evidence[p, g]
            if own_p <= own_r:
                continue
            for f in range(n_frag):
                if match[f] >= 0:
                    claim[f] += evidence[p, match[f]]
                    any_shared[f] = True
        explained = 0.0
        own_library = 0.0
        own_coherent = 0.0
        for f in range(n_frag):
            explained += min(claim[f], evidence[r, f])
            if not any_shared[f] and mz[r, f] > 0:
                own_library += library[r, f]
                if correlation[r, f] > COHERENT_CORRELATION:
                    own_coherent += library[r, f]
        riding[r] = explained / total
        own_support[r] = own_coherent / own_library if own_library > 0 else 0.0
    return riding, own_support


def isolation_window(mz: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Index of the first MS2 isolation window of the DIA cycle that holds each m/z, -1 if none."""
    bounds = cycle[0, :, 0, :]
    lower, upper = bounds[:, 0], bounds[:, 1]
    ms2 = np.flatnonzero(lower > 0)
    order = ms2[np.argsort(lower[ms2], kind="stable")]
    pos = np.searchsorted(lower[order], mz, side="right") - 1
    inside = (pos >= 0) & (mz <= upper[order][np.clip(pos, 0, None)])
    return np.where(inside, pos, -1)


def find_riding(  # noqa: PLR0913 # the dense fragment arrays come as four arrays
    rows: pd.DataFrame,
    fragments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    partner: np.ndarray,
    cycle: np.ndarray,
    ms2_tolerance_ppm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Riding share and own support of every row of `rows`.

    `rows` needs REQUIRED_COLUMNS, `proba` (P(decoy)) and `_decoy`; `fragments` are the (len(rows), TOP_K_FRAGMENTS)
    library m/z (0: empty slot), library intensity, observed intensity and correlation of each row; `partner` marks the rows that can explain others. A row's
    riding share is the part of its fragment evidence that co-eluting partners with a lower proba and more evidence on
    their own fragments explain; its own support is the coherent share of the library intensity of the fragments no
    such partner shares.
    """
    mz, library, observed, correlation = fragments
    evidence = fragment_evidence(observed, correlation, library)
    window = isolation_window(rows[CalibCols.MZ_OBSERVED].to_numpy(np.float64), cycle)
    rt = rows[CalibCols.RT_OBSERVED].to_numpy(np.float64)
    partner_rows = np.flatnonzero(partner & (window >= 0))
    partner_rows = partner_rows[np.lexsort((rt[partner_rows], window[partner_rows]))]
    n_windows = int(window.max()) + 1 if len(window) else 0
    partner_window = window[partner_rows]
    return _riding(
        window.astype(np.int64),
        rt,
        rows["fwhm_rt"].to_numpy(np.float64),
        rows["elution_group_idx"].to_numpy(np.int64),
        rows[SEQUENCE_ID_COLUMN].to_numpy(np.int64),
        rows["_decoy"].to_numpy(np.int64),
        rows["proba"].to_numpy(np.float64),
        mz,
        evidence,
        library,
        correlation,
        partner_rows.astype(np.int64),
        np.searchsorted(partner_window, np.arange(n_windows), "left"),
        np.searchsorted(partner_window, np.arange(n_windows), "right"),
        SHARED_TOLERANCE_FACTOR * ms2_tolerance_ppm,
        CLASS_BLIND,
    )

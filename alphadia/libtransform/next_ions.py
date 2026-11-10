"""The library fragments ranked just below the flattened top fragments ("next ions") of every precursor.

Flattening keeps the top-k fragments by predicted intensity. The rust next-ion features count how many of the next
NEXT_ION_SLOTS b/y ions are seen at a candidate's apex scan: at low input a faint real peptide still shows some of
them, a chance match does not. They are kept as precursor columns so that they follow every subset of the library.
"""

import re

import numba as nb
import numpy as np
import pandas as pd

NEXT_ION_SLOTS = 12
NEXT_ION_COLUMNS = [f"next_ion_mz_{i}" for i in range(NEXT_ION_SLOTS)]
# b1 / y1 and b_(n-1) / y_(n-1) are left out: y1 of K/R is in almost every tryptic spectrum, and the replay that
# measured the features counted cleavages 2 .. n-2 only
MIN_ION_NUMBER = 2
MAX_ION_NUMBER_FROM_END = 2
BY_COLUMN = re.compile(r"([by])_z\d+")


@nb.njit(parallel=True, cache=True)
def _next_ion_mz(
    mz: np.ndarray,
    intensity: np.ndarray,
    frag_start: np.ndarray,
    frag_stop: np.ndarray,
    n_aa: np.ndarray,
    column_is_by: np.ndarray,
    column_is_y: np.ndarray,
    top_k: int,
    min_intensity: float,
    out: np.ndarray,
) -> None:
    n_columns = mz.shape[1]
    for p in nb.prange(len(frag_start)):
        block_intensity = intensity[frag_start[p] : frag_stop[p]].ravel()
        block_mz = mz[frag_start[p] : frag_stop[p]].ravel()
        order = np.argsort(-block_intensity, kind="mergesort")
        slot = 0
        for rank in range(top_k, min(len(order), top_k + out.shape[1])):
            j = order[rank]
            if block_intensity[j] < min_intensity:
                break
            column = j % n_columns
            if block_mz[j] <= 0 or not column_is_by[column]:
                continue
            row = j // n_columns
            number = n_aa[p] - 1 - row if column_is_y[column] else row + 1
            if number < MIN_ION_NUMBER or number > n_aa[p] - MAX_ION_NUMBER_FROM_END:
                continue
            out[p, slot] = block_mz[j]
            slot += 1


def next_ion_mz(
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    top_k_fragments: int,
    min_fragment_intensity: float,
) -> np.ndarray:
    """[precursor, NEXT_ION_SLOTS] library m/z of the b/y ions at intensity ranks top_k .. top_k + NEXT_ION_SLOTS - 1
    (at least `min_fragment_intensity`, cleavages 2 .. n-2) in `precursor_df` row order, 0 = empty slot.

    Ranks are over all fragment columns as in flattening, so the top-k ions are the flattened ones.
    """
    columns = [str(c) for c in fragment_mz_df.columns]
    out = np.zeros((len(precursor_df), NEXT_ION_SLOTS), dtype=np.float32)
    _next_ion_mz(
        # no dtype conversion: a copy of the dense fragment tables of a large library costs several GB
        np.ascontiguousarray(fragment_mz_df.to_numpy()),
        np.ascontiguousarray(fragment_intensity_df.to_numpy()),
        precursor_df["frag_start_idx"].to_numpy(np.int64),
        precursor_df["frag_stop_idx"].to_numpy(np.int64),
        precursor_df["nAA"].to_numpy(np.int64),
        np.array([BY_COLUMN.fullmatch(c) is not None for c in columns]),
        np.array([c.startswith("y_") for c in columns]),
        top_k_fragments,
        min_fragment_intensity,
        out,
    )
    return out

"""Utility methods for fragment competition."""

import logging

import numba as nb
import numpy as np
import pandas as pd

from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger(__name__)


def add_frag_start_stop_idx(
    psm_df: pd.DataFrame, frag_df: pd.DataFrame
) -> pd.DataFrame:
    """The fragment dataframe is indexed by the precursor index.

    This function adds the start and stop indices of the fragments to the PSM dataframe.

    Parameters
    ----------
    psm_df: pd.DataFrame
        The PSM dataframe.

    frag_df: pd.DataFrame
        The fragment dataframe.

    Returns
    -------
    pd.DataFrame
        The PSM dataframe with the start and stop indices of the fragments.

    """
    if "_frag_start_idx" in psm_df.columns and "_frag_stop_idx" in psm_df.columns:
        logger.warning(
            "Fragment start and stop indices already present in PSM dataframe. Skipping."
        )
        return psm_df

    frag_df["frag_idx"] = np.arange(len(frag_df))
    index_df = frag_df.groupby("_candidate_idx", as_index=False).agg(
        _frag_start_idx=pd.NamedAgg("frag_idx", min),
        _frag_stop_idx=pd.NamedAgg("frag_idx", max),
    )
    index_df["_frag_stop_idx"] += 1

    return psm_df.merge(index_df, "inner", on="_candidate_idx")


def candidate_hash(precursor_idx: int, rank: int) -> int:
    """Create a 64 bit hash from the precursor_idx, and rank.

    The precursor_idx is the lower 32 bits.
    The rank is the next 8 bits.
    """
    errors = []

    # simple = precursor_idx + (rank << 32)
    original = _candidate_hash(precursor_idx, rank)
    # castuint64 = (precursor_idx + (rank << 32)).astype(np.uint64)
    castint64 = (precursor_idx + (rank << 32)).astype(np.int64)

    # for key, hash_values in {
    #     "simple": simple,
    #     "original": original,
    #     "castuint64": castuint64,
    #     "castint64": castint64,
    # }.items():
    #     if len(hash_values) != len(set(hash_values)):
    #         unique_vals, counts = np.unique(hash_values, return_counts=True)
    #         duplicates = unique_vals[counts > 1]
    # num_diff = np.sum(castint64 != original)
    if any(castint64 != original):
        key = "castint64!=original"
        errors.append(
            f"Hash value {key} {type(original)} {type(original[0])} {type(castint64)} {type(castint64[0])}: hash_values {original[:10]}..{original[-10:]} not unique"
        )

        num_diff = np.sum(castint64 != original)
        mismatches = np.where(castint64 != original)[0]
        mismatch_values = [(original[i], castint64[i]) for i in mismatches]
        key = "castint64!=original"
        errors.append(
            f"Hash value {key} {type(original)} {type(original[0])} {type(castint64)} {type(castint64[0])}: {num_diff} differing values; "
        )
        errors.append(
            f"mismatches: {mismatch_values}; hash_values {original[:10]}..{original[-10:]} not unique"
        )
        errors.append(f"original {original[:10]}..{original[-10:]}  ")
        errors.append(f"castint64 {castint64[:10]}..{castint64[-10:]}  ")

    for error in errors:
        logger.error(error)

    if errors:
        raise RuntimeError(errors)

    return castint64


@nb.njit(cache=USE_NUMBA_CACHING)
def _candidate_hash(precursor_idx: int, rank: int) -> int:
    """Create a 64 bit hash from the precursor_idx, and rank.

    The precursor_idx is the lower 32 bits.
    The rank is the next 8 bits.
    """
    return precursor_idx + (rank << 32)

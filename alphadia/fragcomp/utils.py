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

    simple = precursor_idx + (rank << 32)
    original = _candidate_hash(precursor_idx, rank)
    castuint64 = (precursor_idx + (rank << 32)).astype(np.uint64)
    castint64 = (precursor_idx + (rank << 32)).astype(np.int64)

    for key, hash_values in {
        "simple": simple,
        "original": original,
        "castuint64": castuint64,
        "castint64": castint64,
    }.items():
        if len(hash_values) != len(set(hash_values)):
            duplicates = set([x for x in hash_values if hash_values.count(x) > 1])  # noqa: C403
            errors.append(
                f"Hash value {key} {type(hash_values)}: {hash_values} not unique: {len(hash_values)=} {len(set(hash_values))=} {duplicates=}"
            )

    for error in errors:
        logger.error(error)

    if errors:
        raise RuntimeError(errors)

    return castuint64


@nb.njit(cache=USE_NUMBA_CACHING)
def _candidate_hash(precursor_idx: int, rank: int) -> int:
    """Create a 64 bit hash from the precursor_idx, and rank.

    The precursor_idx is the lower 32 bits.
    The rank is the next 8 bits.
    """
    return precursor_idx + (rank << 32)

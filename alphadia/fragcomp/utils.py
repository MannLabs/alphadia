"""Utility methods for fragment competition."""

import logging

import numpy as np
import pandas as pd

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
        _frag_start_idx=pd.NamedAgg("frag_idx", "min"),
        _frag_stop_idx=pd.NamedAgg("frag_idx", "max"),
    )
    index_df["_frag_stop_idx"] += 1

    return psm_df.merge(index_df, "inner", on="_candidate_idx")


def candidate_hash(precursor_idx: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Create a 64 bit hash (np.ndarray[np.uint64]) from precursor_idx and rank.

    The precursor_idx is the lower 32 bits.
    The rank is the next 8 bits.

    Note: the explicit casting is important to ensure it returns a 64-bit integer.
    """
    return (precursor_idx.astype(np.int64) + (rank.astype(np.int64) << 32)).astype(
        np.uint64
    )

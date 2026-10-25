"""The fragment competition module contains functionality to maintain the exclusive assignment of signal to identifications.

The sweep itself lives in Rust (`alphadia_search_rs.FragmentCompetition`); what is
left here is the dataframe bookkeeping it needs.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from alphadia_search_rs import FragmentCompetition
from pandas.errors import SettingWithCopyWarning

from alphadia.constants.keys import CalibCols
from alphadia.fragcomp.utils import candidate_hash

logger = logging.getLogger(__name__)

DEFAULT_RT_TOLERANCE_SECONDS = 3.0
DEFAULT_MASS_TOLERANCE_PPM = 15.0
# the isolation window a fragment was extracted from (its centre m/z) and the offset of the extracted m/z from the
# monoisotopic fragment m/z, as the rust quantification reports them
WINDOW_MZ_COLUMN = "window_mz"
MZ_SHIFT_COLUMN = "mz_shift"
CANDIDATE_COLUMN = "_candidate_idx"


def compete_for_fragments(
    psm_df: pd.DataFrame,
    frag_df: pd.DataFrame,
    cycle: np.ndarray,
    rt_tol_seconds: float = DEFAULT_RT_TOLERANCE_SECONDS,
    mass_tol_ppm: float = DEFAULT_MASS_TOLERANCE_PPM,
) -> pd.DataFrame:
    """Remove PSMs that share fragments with other PSMs.

    A PSM competes in every isolation window its fragments were extracted from, with the
    fragments extracted there at the m/z they were extracted at, against the PSMs that
    extracted fragments from the same window within `rt_tol_seconds`; the lower `proba`
    wins, and a PSM that loses in any window is removed. Row order does not matter and is
    preserved.

    Parameters
    ----------
    psm_df: pd.DataFrame
        The PSM dataframe.

    frag_df: pd.DataFrame
        The quantified fragment dataframe, with the isolation window and m/z shift of each fragment.

    cycle: np.ndarray
        DIA cycle.

    rt_tol_seconds: float
        The retention time tolerance in seconds.

    mass_tol_ppm: float
        The mass tolerance in ppm.

    Returns
    -------
    pd.DataFrame
        The PSM dataframe, reduced to the PSMs that won their fragments.

    """
    # TODO: this method raises SettingWithCopyWarning. Resolve without increasing memory usage.

    warnings.simplefilter(action="ignore", category=(SettingWithCopyWarning))

    psm_df[CANDIDATE_COLUMN] = candidate_hash(
        psm_df["precursor_idx"].values, psm_df["rank"].values
    )
    frag_df[CANDIDATE_COLUMN] = candidate_hash(
        frag_df["precursor_idx"].values, frag_df["rank"].values
    )

    parts, fragment_mz = window_parts(psm_df, frag_df)
    valid_parts = FragmentCompetition(rt_tol_seconds, mass_tol_ppm).compete(
        parts[WINDOW_MZ_COLUMN].to_numpy(dtype=np.float32),
        parts["precursor_idx"].to_numpy(dtype=np.int64),
        parts["proba"].to_numpy(dtype=np.float64),
        parts[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
        parts["_frag_start_idx"].to_numpy(dtype=np.int64),
        parts["_frag_stop_idx"].to_numpy(dtype=np.int64),
        fragment_mz,
        np.ascontiguousarray(cycle, dtype=np.float32),
    )
    lost = set(parts.loc[~valid_parts, CANDIDATE_COLUMN])
    valid = ~psm_df[CANDIDATE_COLUMN].isin(lost).to_numpy()

    warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
    return psm_df[valid]


def window_parts(
    psm_df: pd.DataFrame, frag_df: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per PSM and isolation window its fragments were extracted from, with the PSM's precursor_idx, proba
    and retention time and the range of its fragments in the returned flat array of extracted fragment m/z.

    A PSM without quantified fragments has no part and competes with nobody.
    """
    fragments = frag_df[
        [CANDIDATE_COLUMN, WINDOW_MZ_COLUMN, MZ_SHIFT_COLUMN, CalibCols.MZ_OBSERVED]
    ].sort_values([CANDIDATE_COLUMN, WINDOW_MZ_COLUMN], kind="stable")
    key = fragments[[CANDIDATE_COLUMN, WINDOW_MZ_COLUMN]].to_numpy()
    first = np.flatnonzero(np.r_[True, (key[1:] != key[:-1]).any(axis=1)])
    parts = fragments.iloc[first][[CANDIDATE_COLUMN, WINDOW_MZ_COLUMN]].reset_index(
        drop=True
    )
    parts["_frag_start_idx"] = first
    parts["_frag_stop_idx"] = np.r_[first[1:], len(fragments)]
    parts = parts.merge(
        psm_df[[CANDIDATE_COLUMN, "precursor_idx", "proba", CalibCols.RT_OBSERVED]],
        on=CANDIDATE_COLUMN,
        how="inner",
    )
    fragment_mz = fragments[CalibCols.MZ_OBSERVED].to_numpy(
        dtype=np.float32
    ) + fragments[MZ_SHIFT_COLUMN].to_numpy(dtype=np.float32)
    return parts, np.ascontiguousarray(fragment_mz)

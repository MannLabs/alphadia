"""The fragment competition module contains functionality to maintain the exclusive assignment of signal to identifications.

The numeric work — DIA-window assignment, priority ranking and the fragment-overlap
sweep — lives in Rust (`alphadia_search_rs.FragmentCompetition`). This module only
prepares the candidate/fragment index bookkeeping around it.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from alphadia_search_rs import FragmentCompetition
from pandas.errors import SettingWithCopyWarning

from alphadia.constants.keys import CalibCols
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash

logger = logging.getLogger(__name__)

DEFAULT_RT_TOLERANCE_SECONDS = 3.0
DEFAULT_MASS_TOLERANCE_PPM = 15.0


def compete_for_fragments(
    psm_df: pd.DataFrame,
    frag_df: pd.DataFrame,
    cycle: np.ndarray,
    rt_tol_seconds: float = DEFAULT_RT_TOLERANCE_SECONDS,
    mass_tol_ppm: float = DEFAULT_MASS_TOLERANCE_PPM,
) -> pd.DataFrame:
    """Remove PSMs that share fragments with other PSMs.

    PSMs compete only against others in the same DIA isolation window and within
    `rt_tol_seconds`; of two PSMs sharing fragments the one with the lower `proba`
    wins. Row order is irrelevant and is preserved in the result.

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

    valid = FragmentCompetition(rt_tol_seconds, mass_tol_ppm).compete(
        psm_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
        psm_df["precursor_idx"].to_numpy(dtype=np.int64),
        psm_df["proba"].to_numpy(dtype=np.float64),
        psm_df[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
        psm_df["_frag_start_idx"].to_numpy(dtype=np.int64),
        psm_df["_frag_stop_idx"].to_numpy(dtype=np.int64),
        frag_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
        np.ascontiguousarray(cycle, dtype=np.float32),
    )

    # clean up
    psm_df.drop(columns=["_frag_start_idx", "_frag_stop_idx"], inplace=True)

    warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
    return psm_df[valid]

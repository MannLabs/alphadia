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
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash

logger = logging.getLogger(__name__)

DEFAULT_RT_TOLERANCE_SECONDS = 3.0
DEFAULT_MASS_TOLERANCE_PPM = 15.0
# about the width of a peptide's mobility peak on a timsTOF (1/K0)
DEFAULT_MOBILITY_TOLERANCE = 0.02


def compete_for_fragments(
    psm_df: pd.DataFrame,
    frag_df: pd.DataFrame,
    cycle: np.ndarray,
    rt_tol_seconds: float = DEFAULT_RT_TOLERANCE_SECONDS,
    mass_tol_ppm: float = DEFAULT_MASS_TOLERANCE_PPM,
    scan_mobility: np.ndarray | None = None,
    mobility_tol: float = DEFAULT_MOBILITY_TOLERANCE,
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

    scan_mobility: np.ndarray, optional
        The 1/K0 of every scan of `cycle`, for ion mobility data: a PSM's window is the one that isolated it on the
        scan nearest its observed mobility, and PSMs compete only if their mobilities are closer than `mobility_tol`.

    mobility_tol: float
        The mobility tolerance in 1/K0.

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

    competition = FragmentCompetition(rt_tol_seconds, mass_tol_ppm)
    if scan_mobility is None:
        valid = competition.compete(
            psm_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
            psm_df["precursor_idx"].to_numpy(dtype=np.int64),
            psm_df["proba"].to_numpy(dtype=np.float64),
            psm_df[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
            psm_df["_frag_start_idx"].to_numpy(dtype=np.int64),
            psm_df["_frag_stop_idx"].to_numpy(dtype=np.int64),
            frag_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
            np.ascontiguousarray(cycle, dtype=np.float32),
        )
    else:
        valid = competition.compete_with_mobility(
            psm_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
            psm_df["precursor_idx"].to_numpy(dtype=np.int64),
            psm_df["proba"].to_numpy(dtype=np.float64),
            psm_df[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
            psm_df[CalibCols.MOBILITY_OBSERVED].to_numpy(dtype=np.float32),
            psm_df["_frag_start_idx"].to_numpy(dtype=np.int64),
            psm_df["_frag_stop_idx"].to_numpy(dtype=np.int64),
            frag_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
            np.ascontiguousarray(cycle, dtype=np.float32),
            np.ascontiguousarray(scan_mobility, dtype=np.float32),
            mobility_tol,
        )

    # these only exist to point the Rust sweep at each PSM's fragments
    psm_df.drop(columns=["_frag_start_idx", "_frag_stop_idx"], inplace=True)

    warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
    return psm_df[valid]

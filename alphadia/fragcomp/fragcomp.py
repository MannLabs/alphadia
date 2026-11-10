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


def compete_for_fragments(
    psm_df: pd.DataFrame,
    frag_df: pd.DataFrame,
    cycle: np.ndarray,
    rt_tol_seconds: float = DEFAULT_RT_TOLERANCE_SECONDS,
    mass_tol_ppm: float = DEFAULT_MASS_TOLERANCE_PPM,
    decoy_column: str = "decoy",
) -> pd.DataFrame:
    """Remove PSMs that share fragments with other PSMs.

    PSMs only compete against others of their own class (targets against targets,
    decoys against decoys), in the same DIA window and within `rt_tol_seconds`, and
    the lower `proba` wins. Row order does not matter and is preserved.

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

    decoy_column: str
        The column that is non-zero for decoys.

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
    fragment_mz = frag_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32)
    cycle = np.ascontiguousarray(cycle, dtype=np.float32)
    valid = np.zeros(len(psm_df), dtype=bool)
    is_decoy = psm_df[decoy_column].to_numpy() != 0
    # Same rule as the riding removal: a decoy is never removed for sharing ions with a
    # target. Class-blind, decoys riding on confident targets lost twice as often as
    # false targets (whose riders the riding removal had taken out before the final
    # fit), which loosened the q-value cutoff and let more false targets through.
    for members in (np.flatnonzero(~is_decoy), np.flatnonzero(is_decoy)):
        rows = psm_df.iloc[members]
        valid[members] = competition.compete(
            rows[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
            rows["precursor_idx"].to_numpy(dtype=np.int64),
            rows["proba"].to_numpy(dtype=np.float64),
            rows[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
            rows["_frag_start_idx"].to_numpy(dtype=np.int64),
            rows["_frag_stop_idx"].to_numpy(dtype=np.int64),
            fragment_mz,
            cycle,
        )

    # these only exist to point the Rust sweep at each PSM's fragments
    psm_df.drop(columns=["_frag_start_idx", "_frag_stop_idx"], inplace=True)

    warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
    return psm_df[valid]

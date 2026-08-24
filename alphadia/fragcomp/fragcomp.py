"""The fragment competition module contains functionality to maintain the exclusive assignment of signal to identifications.

Thin Python wrapper around the Rust fragment-competition kernel
(`alphadia_search_rs.FragmentCompetition`). The numeric sweep (per-DIA-window
fragment-overlap comparison) lives in Rust; this wrapper only prepares the
candidate/fragment index bookkeeping and dataframe I/O.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from alphadia_search_rs import FragmentCompetition as _RustFragmentCompetition
from pandas.errors import SettingWithCopyWarning

from alphadia.constants.keys import CalibCols
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash

logger = logging.getLogger(__name__)


class FragmentCompetition:
    """Fragment competition class to remove PSMs that share fragments with other PSMs."""

    def __init__(self, rt_tol_seconds: float = 3, mass_tol_ppm: float = 15):
        """Remove PSMs that share fragments with other PSMs.

        Parameters
        ----------
        rt_tol_seconds: float
            The retention time tolerance in seconds.

        mass_tol_ppm: float
            The mass tolerance in ppm.

        """
        self.rt_tol_seconds = rt_tol_seconds
        self.mass_tol_ppm = mass_tol_ppm
        self._fragment_competition = _RustFragmentCompetition(
            float(rt_tol_seconds), float(mass_tol_ppm)
        )

    @staticmethod
    def _add_window_idx(psm_df: pd.DataFrame, cycle: np.ndarray) -> pd.DataFrame:
        """Add the window index to the PSM dataframe.

        Parameters
        ----------
        psm_df: pd.DataFrame
            The PSM dataframe.

        cycle: np.ndarray
            The cycle array.

        Returns
        -------
        pd.DataFrame
            The PSM dataframe with the window index.

        """
        if "window_idx" in psm_df.columns:
            logger.warning("Window index already present in PSM dataframe. Skipping.")
            return psm_df

        lower_limit = np.min(cycle[0, :, :, 0], axis=1, keepdims=True).T
        upper_limit = np.max(cycle[0, :, :, 1], axis=1, keepdims=True).T

        idx = (
            np.expand_dims(psm_df[CalibCols.MZ_OBSERVED].values, axis=-1) >= lower_limit
        ) & (
            np.expand_dims(psm_df[CalibCols.MZ_OBSERVED].values, axis=-1) < upper_limit
        )

        psm_df["window_idx"] = np.argmax(idx, axis=1)
        return psm_df

    def __call__(
        self, psm_df: pd.DataFrame, frag_df: pd.DataFrame, cycle: np.ndarray
    ) -> pd.DataFrame:
        """Remove PSMs that share fragments with other PSMs.

        Parameters
        ----------
        psm_df: pd.DataFrame
            The PSM dataframe.

        frag_df: pd.DataFrame
            The fragment dataframe.

        cycle: np.ndarray
            DIA cycle.

        Returns
        -------
        pd.DataFrame
            The PSM dataframe with the valid column.

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
        psm_df = self._add_window_idx(psm_df, cycle)

        # important to sort by window_idx and proba: the Rust sweep processes
        # candidates in this order, and within a conflicting pair the earlier one wins
        psm_df.sort_values(
            by=["window_idx", "proba", "precursor_idx"], inplace=True
        )  # last sort to break ties

        valid = self._fragment_competition.compete(
            psm_df["window_idx"].to_numpy(dtype=np.int64),
            psm_df[CalibCols.RT_OBSERVED].to_numpy(dtype=np.float32),
            psm_df["_frag_start_idx"].to_numpy(dtype=np.int64),
            psm_df["_frag_stop_idx"].to_numpy(dtype=np.int64),
            frag_df[CalibCols.MZ_OBSERVED].to_numpy(dtype=np.float32),
        )

        psm_df["valid"] = valid

        # clean up
        psm_df.drop(
            columns=["_frag_start_idx", "_frag_stop_idx", "window_idx"], inplace=True
        )

        warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
        return psm_df[psm_df["valid"]]

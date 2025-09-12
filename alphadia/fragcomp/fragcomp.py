"""The fragment competition module contains functionality to maintain the exclusive assignment of signal to identifications."""

import logging
import warnings

import numba as nb
import numpy as np
import pandas as pd
from alphatims import utils as timsutils
from pandas.errors import SettingWithCopyWarning

from alphadia.constants.keys import CalibCols
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash
from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger(__name__)


@nb.njit(cache=USE_NUMBA_CACHING)
def _get_fragment_overlap(
    frag_mz_1: np.ndarray,
    frag_mz_2: np.ndarray,
    mass_tol_ppm: float = 10,
) -> int:
    """Get the number of overlapping fragments between two spectra.

    Parameters
    ----------
    frag_mz_1: np.ndarray
        The m/z values of the first spectrum.

    frag_mz_2: np.ndarray
        The m/z values of the second spectrum.

    mass_tol_ppm: float
        The mass tolerance in ppm.

    Returns
    -------
    int
        The number of overlapping fragments.

    """
    frag_mz_1 = frag_mz_1.reshape(-1, 1)
    frag_mz_2 = frag_mz_2.reshape(1, -1)
    delta_mz = np.abs(frag_mz_1 - frag_mz_2)
    ppm_delta_mz = delta_mz / frag_mz_1 * 1e6
    return np.sum(ppm_delta_mz < mass_tol_ppm)


@timsutils.pjit(cache=USE_NUMBA_CACHING)
def _compete_for_fragments(  # noqa: PLR0913 # Too many arguments
    thread_idx: int,  # pjit decorator changes the passed argument from an iterable to single index
    precursor_start_idxs: np.ndarray,
    precursor_stop_idxs: np.ndarray,
    rt: np.ndarray,
    frag_start_idx: np.ndarray,
    frag_stop_idx: np.ndarray,
    fragment_mz: np.ndarray,
    rt_tol_seconds: float,
    mass_tol_ppm: float,
    valid: np.ndarray,
) -> None:
    """Remove PSMs that share fragments with other PSMs.

    The function is applied on a dia window basis.

    The pjit decorator thread-parallelizes over the first argument index and additionally wraps with numba.njit(nogil=True).
    Make sure to read and understand the pjit decorator, especially how it changes the type of the first argument.

    Parameters
    ----------
    thread_idx: int
        The thread index. Each thread will handle one dia window.
        The pjit decorator effectively changes the type of this argument to `np.ndarray` and thread-parallelizes
        over it.

    precursor_start_idxs: np.ndarray
        Array of length n_windows. The start indices of the precursors in the PSM dataframe.

    precursor_stop_idxs: np.ndarray
        Array of length n_windows. The stop indices of the precursors in the PSM dataframe.

    rt: np.ndarray
        The retention times of the precursors.

    frag_start_idx: np.ndarray
        Array of length n_psms. The start indices of the fragments in the fragment dataframe.

    frag_stop_idx: np.ndarray
        Array of length n_psms. The stop indices of the fragments in the fragment dataframe.

    fragment_mz: np.ndarray
        The m/z values of the fragments.

    rt_tol_seconds: float
        The retention time tolerance in seconds.

    mass_tol_ppm: float
        The mass tolerance in ppm.

    valid: np.ndarray
        Array of length n_psms. The validity of each PSM. This is where the method output will be stored.

    Returns
    -------
        None, but modifies the `valid` array in place.

    """
    precursor_start_idx = precursor_start_idxs[thread_idx]
    precursor_stop_idx = precursor_stop_idxs[thread_idx]

    rt_window = rt[precursor_start_idx:precursor_stop_idx]
    valid_window = valid[precursor_start_idx:precursor_stop_idx]

    for i, i_rt in enumerate(rt_window):
        if not valid_window[i]:
            continue
        for j, j_rt in enumerate(rt_window):
            if i == j:
                continue
            if not valid_window[j]:
                continue

            delta_rt = abs(i_rt - j_rt)
            if delta_rt < rt_tol_seconds:
                fragment_overlap = _get_fragment_overlap(
                    fragment_mz[
                        frag_start_idx[precursor_start_idx + i] : frag_stop_idx[
                            precursor_start_idx + i
                        ]
                    ],
                    fragment_mz[
                        frag_start_idx[precursor_start_idx + j] : frag_stop_idx[
                            precursor_start_idx + j
                        ]
                    ],
                    mass_tol_ppm=mass_tol_ppm,
                )
                if fragment_overlap >= 3:  # noqa: PLR2004
                    valid_window[j] = False

    valid[precursor_start_idx:precursor_stop_idx] = valid_window


class FragmentCompetition:
    """Fragment competition class to remove PSMs that share fragments with other PSMs."""

    def __init__(
        self, rt_tol_seconds: int = 3, mass_tol_ppm: int = 15, thread_count: int = 8
    ):
        """Remove PSMs that share fragments with other PSMs.

        Parameters
        ----------
        rt_tol_seconds: int
            The retention time tolerance in seconds.

        mass_tol_ppm: int
            The mass tolerance in ppm.

        thread_count: int
            The number of threads to use.

        """
        self.rt_tol_seconds = rt_tol_seconds
        self.mass_tol_ppm = mass_tol_ppm
        self.thread_count = thread_count

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

    @staticmethod
    def _get_thread_plan_df(psm_df: pd.DataFrame) -> pd.DataFrame:
        """Expects a dataframe sorted by window idxs and qvals.

        Returns a dataframe with start and stop indices of the threads.

        Parameters
        ----------
        psm_df: pd.DataFrame
            The PSM dataframe.

        Returns
        -------
        pd.DataFrame
            The thread plan dataframe.

        """
        psm_df["_thread_idx"] = np.arange(len(psm_df))
        index_df = psm_df.groupby("window_idx", as_index=False).agg(
            start_idx=pd.NamedAgg("_thread_idx", "min"),
            stop_idx=pd.NamedAgg("_thread_idx", "max"),
        )
        index_df["stop_idx"] += 1

        psm_df.drop(columns=["_thread_idx"], inplace=True)
        return index_df

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
            DIA cycle as provided by alphatims.

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

        # important to sort by window_idx and proba
        psm_df.sort_values(
            by=["window_idx", "proba", "precursor_idx"], inplace=True
        )  # last sort to break ties

        valid = np.ones(len(psm_df)).astype(bool)
        # psm_df["valid"] = True

        timsutils.set_threads(self.thread_count)
        thread_plan_df = self._get_thread_plan_df(psm_df)

        _compete_for_fragments(
            np.arange(len(thread_plan_df)),  # type: ignore  # noqa: PGH003  # function is wrapped by pjit -> will be turned into single index and passed to the method
            thread_plan_df["start_idx"].values,
            thread_plan_df["stop_idx"].values,
            psm_df[CalibCols.RT_OBSERVED].values,
            psm_df["_frag_start_idx"].values,
            psm_df["_frag_stop_idx"].values,
            frag_df[CalibCols.MZ_OBSERVED].values,
            self.rt_tol_seconds,
            self.mass_tol_ppm,
            valid,
        )

        psm_df["valid"] = valid

        # clean up
        psm_df.drop(
            columns=["_frag_start_idx", "_frag_stop_idx", "window_idx"], inplace=True
        )

        warnings.simplefilter(action="default", category=(SettingWithCopyWarning))
        return psm_df[psm_df["valid"]]

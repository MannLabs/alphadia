"""Module to determine the DIA cycle from a spectrum dataframe.

This could be moved to AlphaRaw.
"""

import logging

import numba as nb
import numpy as np
import pandas as pd

from alphadia.exceptions import NotValidDiaDataError
from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger()


def determine_dia_cycle(
    spectrum_df: pd.DataFrame,
    subset_for_cycle_detection: int = 10000,
) -> tuple[np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]], int, int]:
    """Determine the repeating DIA cycle from a spectrum dataframe.

    Detects the cycle length (number of spectra per cycle) via autocorrelation,
    finds where the first complete cycle begins, validates consistency, and returns
    the isolation window boundaries for one cycle.

    Parameters
    ----------
    spectrum_df : pandas.DataFrame
        AlphaRaw compatible spectrum dataframe with columns
        ``isolation_lower_mz``, ``isolation_upper_mz``, and ``rt``.

    subset_for_cycle_detection : int, default = 10000
        The number of spectra to use for cycle detection.

    Returns
    -------
    cycle : np.ndarray
        Array of shape ``(1, cycle_length, 1, 2)`` containing the lower and upper
        isolation m/z boundaries for each scan position in the cycle.

    cycle_start : int
        The spectrum index where the first complete cycle begins.

    cycle_length : int
        The number of spectra per cycle.

    """
    logger.info("Determining DIA cycle")

    cycle_signature = (
        spectrum_df["isolation_lower_mz"].to_numpy()[:subset_for_cycle_detection]
        + spectrum_df["isolation_upper_mz"].to_numpy()[:subset_for_cycle_detection]
    )

    if (cycle_length := _get_cycle_length(cycle_signature)) == -1:
        raise NotValidDiaDataError("Failed to determine length of DIA cycle.")

    if (cycle_start := _get_cycle_start(cycle_signature, cycle_length)) == -1:
        raise NotValidDiaDataError("Failed to determine start of DIA cycle.")

    cycle_start_rt = spectrum_df["rt"].to_numpy()[cycle_start]
    if not _is_valid_cycle(cycle_signature, cycle_length, cycle_start):
        raise NotValidDiaDataError(
            f"Cycle with start {cycle_start_rt:.2f} min and length {cycle_length} detected, but is not consistent."
        )

    logger.info(
        f"Found cycle with start {cycle_start_rt:.2f} min and length {cycle_length}."
    )

    cycle = _build_cycle_array(spectrum_df, cycle_start, cycle_length)

    return cycle, cycle_start, cycle_length


def _get_cycle_length(cycle_signature: np.ndarray) -> int:
    """Determine the number of spectra in one complete DIA cycle.

    Uses normalized autocorrelation to find the periodicity of the cycle signature.
    The highest autocorrelation peak corresponds to the cycle length.

    Parameters
    ----------
    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    Returns
    -------
    cycle_length: int
        The number of spectra per cycle (e.g. 301 = 1 MS1 + 300 MS2 scans), -1 if it could not be determined.

    """
    corr = _normed_auto_correlation(cycle_signature)

    is_peak = (corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:])

    peak_index = is_peak.nonzero()[0] + 1

    if len(peak_index) == 0:
        return -1

    argmax = np.argmax(corr[peak_index])

    return peak_index[argmax]


def _normed_auto_correlation(x: np.ndarray) -> np.ndarray:
    """Calculate the normalized auto correlation of a 1D array.

    Parameters
    ----------
    x : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The normalized auto correlation of the input array.

    """
    x = x - x.mean()
    result = np.correlate(x, x, mode="full")
    result = result[len(result) // 2 :]
    result /= result[0]
    return result


@nb.njit(cache=USE_NUMBA_CACHING)
def _get_cycle_start(
    cycle_signature: np.ndarray,
    cycle_length: int,
) -> int:
    """Find the index of the first complete DIA cycle in the spectrum sequence.

    Some raw files have an incomplete or irregular prefix before the regular cycling pattern
    begins. This function scans forward to find the first position where two consecutive
    windows of length `cycle_length` have identical signatures.

    Parameters
    ----------
    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    cycle_length: int
        The number of spectra per cycle.

    Returns
    -------
    cycle_start: int
        The spectrum index where the first complete cycle begins, -1 if it could not be determined.

    """
    for i in range(len(cycle_signature) - (2 * cycle_length)):
        if np.all(cycle_signature[i : i + cycle_length] == cycle_signature[i]):
            continue

        if np.all(
            cycle_signature[i : i + cycle_length]
            == cycle_signature[i + cycle_length : i + 2 * cycle_length]
        ):
            return i

    return -1


@nb.njit(cache=USE_NUMBA_CACHING)
def _is_valid_cycle(
    cycle_signature: np.ndarray, cycle_length: int, cycle_start: int
) -> bool:
    """Validate that the DIA cycle repeats consistently throughout the signature.

    Checks that every pair of consecutive cycle-length windows (starting from `cycle_start`)
    has identical signatures. Returns False if any mismatch is found.

    Parameters
    ----------
    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    cycle_length: int
        The number of spectra per cycle.

    cycle_start: int
        The spectrum index where the first complete cycle begins.

    Returns
    -------
    cycle_valid: bool
        True if the cycle pattern is consistent across the entire signature.

    """
    for i in range(len(cycle_signature) - (2 * cycle_length) - cycle_start):
        if not np.all(
            cycle_signature[i + cycle_start : i + cycle_start + cycle_length]
            == cycle_signature[
                i + cycle_start + cycle_length : i + cycle_start + 2 * cycle_length
            ]
        ):
            return False
    return True


def _build_cycle_array(
    spectrum_df: pd.DataFrame,
    cycle_start: int,
    cycle_length: int,
) -> np.ndarray:
    """Build the cycle array containing isolation window boundaries for one complete cycle.

    Parameters
    ----------
    spectrum_df : pandas.DataFrame
        AlphaRaw compatible spectrum dataframe with columns
        ``isolation_lower_mz`` and ``isolation_upper_mz``.

    cycle_start : int
        The spectrum index where the first complete cycle begins.

    cycle_length : int
        The number of spectra per cycle.

    Returns
    -------
    cycle : np.ndarray
        Array of shape ``(1, cycle_length, 1, 2)`` containing the lower and upper
        isolation m/z boundaries for each scan position in the cycle.

    """
    cycle = np.zeros((1, cycle_length, 1, 2), dtype=np.float64)
    cycle[0, :, 0, 0] = spectrum_df["isolation_lower_mz"].to_numpy()[
        cycle_start : cycle_start + cycle_length
    ]
    cycle[0, :, 0, 1] = spectrum_df["isolation_upper_mz"].to_numpy()[
        cycle_start : cycle_start + cycle_length
    ]
    return cycle

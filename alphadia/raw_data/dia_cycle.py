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


def _get_cycle_length(cycle_signature: np.ndarray) -> int:
    """Get the cycle length from the cycle signature.

    Parameters
    ----------
    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    Returns
    -------
    cycle_length: int
        The length of the DIA cycle, -1 in case it could not be determined.

    """
    corr = _normed_auto_correlation(cycle_signature)

    is_peak = (corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:])

    peak_index = is_peak.nonzero()[0] + 1

    if len(peak_index) == 0:
        return -1

    argmax = np.argmax(corr[peak_index])

    return peak_index[argmax]


@nb.njit(cache=USE_NUMBA_CACHING)
def _get_cycle_start(
    cycle_signature: np.ndarray,
    cycle_length: int,
) -> int:
    """Get the cycle start from the cycle signature.

    Parameters
    ----------
    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    cycle_length: int
        The length of the DIA cycle.

    Returns
    -------
    cycle_start: int
        The index of the first cycle in the signature.

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
    """Return whether the found DIA cycle is valid.

    Parameters
    ----------
    cycle_signature: np.ndarray
        The signature of the DIA cycle. This will usually be the sum of the isolation windows.

    cycle_length: int
        The length of the DIA cycle.

    cycle_start: int
        The index of the first cycle in the signature.

    Returns
    -------
    cycle_valid: bool
        True if the cycle is valid, False otherwise.

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


def determine_dia_cycle(
    spectrum_df: pd.DataFrame,
    subset_for_cycle_detection: int = 10000,
) -> tuple[np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]], int, int]:
    """Determine the DIA cycle.

    Parameters
    ----------
    spectrum_df : pandas.DataFrame
        AlphaRaw compatible spectrum dataframe.

    subset_for_cycle_detection : int, default = 10000
        The number of spectra to use for cycle detection.

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

    cycle = np.zeros((1, cycle_length, 1, 2), dtype=np.float64)
    cycle[0, :, 0, 0] = spectrum_df["isolation_lower_mz"].to_numpy()[
        cycle_start : cycle_start + cycle_length
    ]
    cycle[0, :, 0, 1] = spectrum_df["isolation_upper_mz"].to_numpy()[
        cycle_start : cycle_start + cycle_length
    ]

    return cycle, cycle_start, cycle_length

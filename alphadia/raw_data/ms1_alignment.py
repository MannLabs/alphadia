"""Place MS1 survey scans that are acquired on their own clock into fixed slots of the MS2 cycle.

alphaDIA and its backends index every spectrum by (cycle, position in cycle), so MS1 scans must sit at a fixed
position of every cycle. Methods with a time-based MS1 interval (e.g. Astral: MS1 every x s, MS2 cycle of 300
windows) put a varying number of MS2 scans between two MS1 scans, so no fixed position exists.

Here the MS2 cycle is determined on the MS2 scans alone, and each cycle gets k MS1 slots, k = the MS1 scans per
cycle rounded, at evenly spaced times within the cycle; each slot holds the MS1 scan acquired nearest to its time.
The backends sum the MS1 observations of a cycle, so the cycle's MS1 trace is the sum of about its own MS1 scans.
"""

import logging

import numpy as np
import pandas as pd

from alphadia.raw_data.dia_cycle import determine_dia_cycle

logger = logging.getLogger()

MS1_LEVEL = 1
# a slot farther than this many MS1 intervals from every MS1 scan lies in an acquisition gap and stays empty
MAX_SLOT_DISTANCE_IN_INTERVALS = 1.0
MIN_MS1_SCANS = 2
MIN_CYCLES = 2


def align_ms1_to_ms2_cycle(spectrum_df: pd.DataFrame) -> pd.DataFrame | None:
    """Spectrum table with the MS1 scans moved into k fixed slots at the start of every MS2 cycle.

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        AlphaRaw spectrum table (``ms_level``, ``rt``, ``isolation_lower_mz``, ``isolation_upper_mz``,
        ``peak_start_idx``, ``peak_stop_idx``, ``spec_idx``), MS1 and MS2 scans in acquisition order.

    Returns
    -------
    pd.DataFrame | None
        The aligned table, starting at the first complete MS2 cycle and ending with the last one, or None when there
        are too few MS1 scans or cycles to align. A slot's spectrum keeps the MS1 scan's peaks and gets the RT of its
        cycle's first MS2 scan; an empty slot has no peaks (``peak_stop_idx == peak_start_idx``).

    """
    is_ms1 = spectrum_df["ms_level"].to_numpy() == MS1_LEVEL
    ms1 = spectrum_df[is_ms1]
    ms2 = spectrum_df[~is_ms1]
    if len(ms1) < MIN_MS1_SCANS:
        return None

    _, cycle_start, cycle_length = determine_dia_cycle(ms2)
    n_cycles = (len(ms2) - cycle_start) // cycle_length
    if n_cycles < MIN_CYCLES:
        return None
    ms2 = ms2.iloc[cycle_start : cycle_start + n_cycles * cycle_length]

    cycle_rt = ms2["rt"].to_numpy()[::cycle_length]
    cycle_time = float(np.median(np.diff(cycle_rt)))
    ms1_rt = ms1["rt"].to_numpy()
    ms1_interval = float(np.median(np.diff(ms1_rt)))
    slots = max(1, round(cycle_time / ms1_interval))

    slot_rt = (
        cycle_rt[:, None] + (np.arange(slots)[None, :] + 0.5) * cycle_time / slots
    ).ravel()
    right = np.clip(np.searchsorted(ms1_rt, slot_rt), 1, len(ms1_rt) - 1)
    left = right - 1
    nearest = np.where(
        np.abs(ms1_rt[left] - slot_rt) <= np.abs(ms1_rt[right] - slot_rt), left, right
    )
    empty = (
        np.abs(ms1_rt[nearest] - slot_rt)
        > MAX_SLOT_DISTANCE_IN_INTERVALS * ms1_interval
    )

    ms1_slots = ms1.iloc[nearest].copy()
    ms1_slots["rt"] = np.repeat(cycle_rt, slots)
    peak_start = ms1_slots["peak_start_idx"].to_numpy()
    ms1_slots["peak_stop_idx"] = np.where(
        empty, peak_start, ms1_slots["peak_stop_idx"].to_numpy()
    )

    # each cycle: its MS1 slots, then its MS2 scans
    order = np.concatenate(
        [
            np.repeat(np.arange(n_cycles), slots) * (slots + cycle_length)
            + np.tile(np.arange(slots), n_cycles),
            np.repeat(np.arange(n_cycles), cycle_length) * (slots + cycle_length)
            + slots
            + np.tile(np.arange(cycle_length), n_cycles),
        ]
    )
    aligned = pd.concat([ms1_slots, ms2], ignore_index=True)
    aligned = aligned.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)
    aligned["spec_idx"] = np.arange(len(aligned))

    used = len(np.unique(nearest[~empty]))
    logger.info(
        f"MS1 scans are not at a fixed position of the DIA cycle: placed {slots} MS1 slot(s) in each of {n_cycles} "
        f"MS2 cycles ({cycle_time:.2f} s, MS1 interval {ms1_interval:.2f} s); {used} of {len(ms1)} MS1 scans used, "
        f"{int(empty.sum())} slot(s) left empty"
    )
    return aligned

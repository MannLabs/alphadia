import logging

import numpy as np

logger = logging.getLogger()


def log_stats(rt_values: np.array, cycle: np.array):
    """Log raw file statistics

    Parameters
    ----------

    rt_values: np.ndarray
            retention time values in seconds for all frames

    cycle: np.ndarray
            DIA cycle object describing the msms pattern
    """

    logger.info("============ Raw file stats ============")

    rt_limits = rt_values.min() / 60, rt_values.max() / 60
    rt_duration_sec = rt_values.max() - rt_values.min()
    rt_duration_min = rt_duration_sec / 60

    logger.info(f"{'RT (min)':<20}: {rt_limits[0]:.1f} - {rt_limits[1]:.1f}")
    logger.info(f"{'RT duration (sec)':<20}: {rt_duration_sec:.1f}")
    logger.info(f"{'RT duration (min)':<20}: {rt_duration_min:.1f}")

    cycle_length = cycle.shape[1]
    cycle_duration = np.diff(rt_values[::cycle_length]).mean()
    cycle_number = len(rt_values) // cycle_length

    logger.info(f"{'Cycle len (scans)':<20}: {cycle_length:.0f}")
    logger.info(f"{'Cycle len (sec)':<20}: {cycle_duration:.2f}")
    logger.info(f"{'Number of cycles':<20}: {cycle_number:.0f}")

    flat_cycle = cycle.flatten()
    flat_cycle = flat_cycle[flat_cycle > 0]
    msms_range = flat_cycle.min(), flat_cycle.max()

    logger.info(f"{'MS2 range (m/z)':<20}: {msms_range[0]:.1f} - {msms_range[1]:.1f}")

    logger.info("========================================")

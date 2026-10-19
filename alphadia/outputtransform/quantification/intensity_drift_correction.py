"""Retention time (RT) dependent intensity drift correction of the fragment table.

Converts the dataframes to the run-major matrices of
`alphadia_search_rs.correct_intensity_drift`, which defines the correction.
"""

import logging

import numpy as np
import pandas as pd
from alphadia_search_rs import correct_intensity_drift as _correct_intensity_drift

from alphadia.constants.keys import CalibCols
from alphadia.outputtransform.quantification.quant_builder import (
    PRECURSOR_IDX_COLUMN,
    RUN_COLUMN,
    get_run_columns,
)

logger = logging.getLogger(__name__)

PSM_COLUMNS = [PRECURSOR_IDX_COLUMN, RUN_COLUMN, CalibCols.RT_OBSERVED]


def correct_intensity_drift(
    intensity_df: pd.DataFrame, correlation_df: pd.DataFrame, psm_df: pd.DataFrame
) -> None:
    """Correct the RT-dependent intensity drift of every run column of `intensity_df`, in place.

    In place, so that no second fragment table is held during quantification.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Linear fragment intensities, one row per ion, with precursor_idx, ion, the
        quantification level columns and one column per run. 0 marks a missing value.
    correlation_df : pd.DataFrame
        Cross-run fragment correlations with the rows and run columns of intensity_df,
        in the same order.
    psm_df : pd.DataFrame
        PSM table with precursor_idx, run and rt_observed.

    Raises
    ------
    ValueError
        If the psm table lacks one of the columns needed to place the precursors in RT.
    """
    missing_columns = set(PSM_COLUMNS) - set(psm_df.columns)
    if missing_columns:
        raise ValueError(
            f"Intensity drift correction needs the psm columns {sorted(missing_columns)}"
        )

    logger.info("Apply RT dependent intensity normalisation")

    run_columns = get_run_columns(intensity_df)
    precursor, precursors = pd.factorize(
        intensity_df[PRECURSOR_IDX_COLUMN].to_numpy(np.int64)
    )
    intensity = _run_major(intensity_df[run_columns])
    correlation = _run_major(correlation_df[run_columns])
    rt = _run_major(_rt_by_run(psm_df, precursors, run_columns))

    _correct_intensity_drift(intensity, correlation, rt, precursor.astype(np.uintp))
    # free the correlation copy before the write-back allocates
    del correlation

    intensity_df[run_columns] = intensity.T


def _run_major(df: pd.DataFrame) -> np.ndarray:
    """Return `df` as a C-contiguous float64 matrix of shape `(n_columns, n_rows)`.

    `to_numpy` of a single-dtype frame is F-contiguous, so its transpose is C-contiguous
    without a copy.
    """
    return np.ascontiguousarray(df.to_numpy(dtype=np.float64).T)


def _rt_by_run(
    psm_df: pd.DataFrame, precursors: np.ndarray, run_columns: list[str]
) -> pd.DataFrame:
    """Observed RT with one row per entry of `precursors` and one column per run, NaN where the run has no psm entry."""
    return (
        psm_df[PSM_COLUMNS]
        .astype({PRECURSOR_IDX_COLUMN: np.int64})
        .drop_duplicates([PRECURSOR_IDX_COLUMN, RUN_COLUMN])
        .pivot(
            index=PRECURSOR_IDX_COLUMN,
            columns=RUN_COLUMN,
            values=CalibCols.RT_OBSERVED,
        )
        .reindex(index=precursors, columns=run_columns)
    )

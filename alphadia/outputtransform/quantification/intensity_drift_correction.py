"""Removal of each run's retention-time dependent intensity drift before LFQ.

The estimator lives in `alphadia_search_rs.correct_intensity_drift`, this module only
converts the dataframes for it.
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
    """Divide the RT-dependent drift out of every run column of `intensity_df`, in place.

    Correcting in place avoids holding a second full fragment matrix through the rest of
    the quantification.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Linear fragment intensities with columns precursor_idx, ion, run1, run2, ...
        and the quantification level columns. Zero marks a missing value.
    correlation_df : pd.DataFrame
        Fragment correlations with the same rows, in the same order, and run columns as
        intensity_df.
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
    """Return `df` as a C-contiguous `(n_columns, n_rows)` matrix.

    The estimator borrows the matrices run-major. pandas hands a single-dtype frame out
    F-contiguous, so the transpose is a view rather than a copy.
    """
    return np.ascontiguousarray(df.to_numpy(dtype=np.float64).T)


def _rt_by_run(
    psm_df: pd.DataFrame, precursors: np.ndarray, run_columns: list[str]
) -> pd.DataFrame:
    """Observed RT of every precursor in every run, NaN where a run has no psm entry."""
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

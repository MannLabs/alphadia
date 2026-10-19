"""Correction of the retention-time dependent (shape-only) intensity drift of every run.

Intensities can drift along the gradient in a run-specific way (e.g. one condition being
0.3-0.6 log2 low after 1000 s). Per run, the log2 deviation from the across-run level of
every ion is binned along RT over well-measured ions, the per-bin location is interpolated
to every ion and divided out.

The curve is re-centred to zero mean over the run's ions before it is applied, so that
only the *shape* along RT is removed. The global level of a run is deliberately left to the
downstream LFQ normalization; removing it here would silently force the majority species
to ratio 1.

The estimator lives in Rust (`alphadia_search_rs.correct_intensity_drift`). What is left
here is the dataframe bookkeeping it needs: placing every precursor in retention time per
run, handing the matrices over run-major and writing the corrected intensities back.
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
        PSM table with precursor_idx, run and rt_observed, used to place every precursor
        in RT for each run.

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
    del correlation

    intensity_df[run_columns] = intensity.T


def _run_major(df: pd.DataFrame) -> np.ndarray:
    """Return `df` as a C-contiguous `(n_columns, n_rows)` matrix.

    The estimator works per run and therefore takes the matrices run-major. pandas hands
    the columns of a single-dtype frame out as one F-contiguous block, which makes the
    transpose a view rather than a copy.
    """
    return np.ascontiguousarray(df.to_numpy(dtype=np.float64).T)


def _rt_by_run(
    psm_df: pd.DataFrame, precursors: np.ndarray, run_columns: list[str]
) -> pd.DataFrame:
    """Observed RT of every precursor in every run, NaN where a run has no psm entry.

    The estimator places a precursor the run has no entry for at its mean RT over the
    other runs, so a cell with an intensity but no psm entry is still corrected.
    """
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

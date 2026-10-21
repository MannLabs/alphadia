"""Retention time (RT) dependent intensity drift correction of the fragment table.

Converts the dataframes to the one-row-per-run matrices of
`alphadia_search_rs.IntensityDriftCorrection`, which defines the correction and its
thresholds.
"""

import logging

import numpy as np
import pandas as pd
from alphadia_search_rs import IntensityDriftCorrection

from alphadia.constants.keys import CalibCols
from alphadia.outputtransform.quantification.quant_builder import (
    PRECURSOR_IDX_COLUMN,
    get_run_columns,
)

logger = logging.getLogger(__name__)

_RUN_COLUMN = "run"
_REQUIRED_PSM_COLUMNS = [PRECURSOR_IDX_COLUMN, _RUN_COLUMN, CalibCols.RT_OBSERVED]


def validate_psm_columns(psm_df: pd.DataFrame) -> None:
    """Raise a ValueError if `psm_df` lacks a column needed to place the precursors in RT."""
    missing_columns = set(_REQUIRED_PSM_COLUMNS) - set(psm_df.columns)
    if missing_columns:
        raise ValueError(
            f"Intensity drift correction needs the psm columns {sorted(missing_columns)}"
        )


def correct_intensity_drift(
    intensity_df: pd.DataFrame, correlation_df: pd.DataFrame, psm_df: pd.DataFrame
) -> pd.DataFrame:
    """Return `intensity_df` with the RT-dependent intensity drift of every run column removed.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Linear fragment intensities, one row per ion, with precursor_idx, ion, the
        quantification level columns and one column per run. 0 marks a missing value.
    correlation_df : pd.DataFrame
        Cross-run fragment correlations with the rows and run columns of intensity_df,
        in the same order.
    psm_df : pd.DataFrame
        PSM table with the columns checked by `validate_psm_columns`.

    Returns
    -------
    pd.DataFrame
        Copy of intensity_df with corrected float32 run columns. The inputs are not changed.
    """
    run_columns = get_run_columns(intensity_df)
    ion_precursor_codes, unique_precursor_idx = pd.factorize(
        intensity_df[PRECURSOR_IDX_COLUMN].to_numpy(np.int64)
    )
    intensity = _runs_as_rows(intensity_df[run_columns])

    summary = IntensityDriftCorrection().correct(
        intensity,
        _runs_as_rows(correlation_df[run_columns]),
        _rt_by_run(psm_df, unique_precursor_idx, run_columns),
        ion_precursor_codes.astype(np.uintp),
    )
    _log_summary(run_columns, summary)

    corrected_df = intensity_df.copy(deep=False)
    corrected_df[run_columns] = intensity.T
    return corrected_df


def _runs_as_rows(df: pd.DataFrame) -> np.ndarray:
    """Return `df` as a C-contiguous float32 matrix of shape `(n_columns, n_rows)`."""
    return np.ascontiguousarray(df.to_numpy(dtype=np.float32).T)


def _rt_by_run(
    psm_df: pd.DataFrame, unique_precursor_idx: np.ndarray, run_columns: list[str]
) -> np.ndarray:
    """Observed RT of shape `(n_runs, n_precursors)`, NaN where the run has no psm of the precursor."""
    rt = np.full(
        (len(run_columns), len(unique_precursor_idx)), np.nan, dtype=np.float32
    )
    psm_run_codes = pd.Index(run_columns).get_indexer(psm_df[_RUN_COLUMN])
    psm_precursor_codes = pd.Index(unique_precursor_idx).get_indexer(
        psm_df[PRECURSOR_IDX_COLUMN].to_numpy(np.int64)
    )
    # psms of runs or precursors outside the fragment table have no cell
    found_mask = (psm_run_codes >= 0) & (psm_precursor_codes >= 0)
    rt[psm_run_codes[found_mask], psm_precursor_codes[found_mask]] = psm_df[
        CalibCols.RT_OBSERVED
    ].to_numpy(np.float32)[found_mask]
    return rt


def _log_summary(run_columns: list[str], summary: dict[str, np.ndarray]) -> None:
    corrected = summary["corrected"]
    max_abs_correction = summary["max_abs_correction"]
    logger.info(
        f"Corrected the RT dependent intensity drift of {corrected.sum()} of "
        f"{len(run_columns)} runs, largest correction {max_abs_correction.max(initial=0.0):.3f} log2"
    )
    for run, run_corrected, run_max in zip(
        run_columns, corrected, max_abs_correction, strict=True
    ):
        if run_corrected:
            logger.debug(f"Run {run}: largest correction {run_max:.3f} log2")
        else:
            logger.debug(f"Run {run}: not corrected, too few usable ions for a curve")

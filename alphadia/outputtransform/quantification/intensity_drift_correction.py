"""Retention time (RT) dependent intensity drift correction of the variance-model precursor table.

Converts the dataframes to the one-row-per-run matrices of
`alphadia_search_rs.IntensityDriftCorrection`, which defines the correction and its
thresholds.
"""

import logging

import numpy as np
import pandas as pd
from alphadia_search_rs import IntensityDriftCorrection

from alphadia.constants.keys import CalibCols, QuantificationLevelKey
from alphadia.outputtransform.quantification.quant_builder import get_run_columns

logger = logging.getLogger(__name__)

_RUN_COLUMN = "run"
_PRECURSOR_COLUMN = QuantificationLevelKey.PRECURSOR
_REQUIRED_PSM_COLUMNS = [_PRECURSOR_COLUMN, _RUN_COLUMN, CalibCols.RT_OBSERVED]

# IntensityDriftCorrection selects ions by fragment correlation, which a precursor table
# does not have. A correlation of one lets every precursor pass that filter.
_PRECURSOR_CORRELATION = 1.0

# keys of the per-run summary returned by IntensityDriftCorrection.correct
_SUMMARY_CORRECTED = "corrected"
_SUMMARY_MAX_ABS_CORRECTION = "max_abs_correction"


def _validate_psm_columns(psm_df: pd.DataFrame) -> None:
    """Raise a ValueError if `psm_df` lacks a column needed to place the precursors in RT."""
    missing_columns = set(_REQUIRED_PSM_COLUMNS) - set(psm_df.columns)
    if missing_columns:
        raise ValueError(
            f"Intensity drift correction needs the psm columns {sorted(missing_columns)}"
        )


def correct_intensity_drift(
    precursor_df: pd.DataFrame, psm_df: pd.DataFrame
) -> pd.DataFrame:
    """Return `precursor_df` with the RT-dependent intensity drift of every run column removed.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Linear precursor quantities of the variance model, one row per precursor, with
        mod_seq_charge_hash, the other quantification level columns and one column per run.
        0 marks a missing value.
    psm_df : pd.DataFrame
        PSM table with mod_seq_charge_hash, run and rt_observed.

    Returns
    -------
    pd.DataFrame
        Copy of precursor_df with corrected float32 run columns. The inputs are not changed.

    Raises
    ------
    ValueError
        If `psm_df` lacks one of the required columns.
    """
    _validate_psm_columns(psm_df)
    run_columns = get_run_columns(precursor_df)
    intensity_matrix = np.ascontiguousarray(
        precursor_df[run_columns].to_numpy(dtype=np.float32).T
    )

    # correct() overwrites intensity_matrix in place, the corrected values are read back from it
    summary = IntensityDriftCorrection().correct(
        intensity_matrix,
        np.full_like(intensity_matrix, _PRECURSOR_CORRELATION),
        _rt_by_run(psm_df, precursor_df[_PRECURSOR_COLUMN].to_numpy(), run_columns),
        np.arange(len(precursor_df), dtype=np.uintp),
    )
    _log_summary(run_columns, summary)

    corrected_df = precursor_df.copy(deep=False)
    corrected_df[run_columns] = intensity_matrix.T
    return corrected_df


def _rt_by_run(
    psm_df: pd.DataFrame, precursor_hashes: np.ndarray, run_columns: list[str]
) -> np.ndarray:
    """Observed RT of shape `(n_runs, n_precursors)`, NaN where the run has no psm of the precursor.

    A precursor with more than one psm in a run takes the RT of the first one.
    """
    psm_df = psm_df.drop_duplicates([_PRECURSOR_COLUMN, _RUN_COLUMN])
    rt = np.full((len(run_columns), len(precursor_hashes)), np.nan, dtype=np.float32)
    psm_run_codes = pd.Index(run_columns).get_indexer(psm_df[_RUN_COLUMN])
    psm_precursor_codes = pd.Index(precursor_hashes).get_indexer(
        psm_df[_PRECURSOR_COLUMN]
    )
    # psms of runs or precursors outside the precursor table have no cell
    found_mask = (psm_run_codes >= 0) & (psm_precursor_codes >= 0)
    rt[psm_run_codes[found_mask], psm_precursor_codes[found_mask]] = psm_df[
        CalibCols.RT_OBSERVED
    ].to_numpy(np.float32)[found_mask]
    return rt


def _log_summary(run_columns: list[str], summary: dict[str, np.ndarray]) -> None:
    corrected_mask = summary[_SUMMARY_CORRECTED]
    max_abs_correction = summary[_SUMMARY_MAX_ABS_CORRECTION]
    logger.info(
        f"Corrected the RT dependent intensity drift of {corrected_mask.sum()} of "
        f"{len(run_columns)} runs, largest correction {max_abs_correction.max(initial=0.0):.3f} log2"
    )
    for run, is_run_corrected, run_max_abs_correction in zip(
        run_columns, corrected_mask, max_abs_correction, strict=True
    ):
        if is_run_corrected:
            logger.debug(
                f"Run {run}: largest correction {run_max_abs_correction:.3f} log2"
            )
        else:
            logger.debug(
                f"Run {run}: not corrected, too few usable precursors for a curve"
            )

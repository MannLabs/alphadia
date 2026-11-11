"""Retention time (RT) dependent intensity drift correction of the fragment or the precursor table.

Converts the dataframes to the one-row-per-run matrices of
`alphadia_search_rs.IntensityDriftCorrection` (fragments, along the RT) and
`alphadia_search_rs.PrecursorDriftCorrection` (precursors, along the RT and the precursor
mass), which define the corrections and their thresholds.
"""

import logging

import numpy as np
import pandas as pd
from alphadia_search_rs import IntensityDriftCorrection, PrecursorDriftCorrection

from alphadia.constants.keys import CalibCols, QuantificationLevelKey
from alphadia.outputtransform.quantification.quant_builder import (
    PRECURSOR_IDX_COLUMN,
    get_run_columns,
)

logger = logging.getLogger(__name__)

_RUN_COLUMN = "run"
_CHARGE_COLUMN = "charge"
_REQUIRED_PSM_COLUMNS = [PRECURSOR_IDX_COLUMN, _RUN_COLUMN, CalibCols.RT_OBSERVED]
_PRECURSOR_REQUIRED_PSM_COLUMNS = [
    QuantificationLevelKey.PRECURSOR,
    _RUN_COLUMN,
    CalibCols.RT_OBSERVED,
    CalibCols.MZ_LIBRARY,
    _CHARGE_COLUMN,
]

# keys of the summary returned by IntensityDriftCorrection.correct and PrecursorDriftCorrection.correct
_SUMMARY_CORRECTED = "corrected"
_SUMMARY_MAX_ABS_CORRECTION = "max_abs_correction"
_SUMMARY_N_STRATA = "n_strata"


def _validate_psm_columns(psm_df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a ValueError if `psm_df` lacks one of `required_columns`."""
    missing_columns = set(required_columns) - set(psm_df.columns)
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
        PSM table with precursor_idx, run and rt_observed.

    Returns
    -------
    pd.DataFrame
        Copy of intensity_df with corrected float32 run columns. The inputs are not changed.

    Raises
    ------
    ValueError
        If `psm_df` lacks one of the required columns.
    """
    _validate_psm_columns(psm_df, _REQUIRED_PSM_COLUMNS)
    run_columns = get_run_columns(intensity_df)
    ion_precursor_codes, unique_precursor_idx = pd.factorize(
        intensity_df[PRECURSOR_IDX_COLUMN].to_numpy(np.int64)
    )
    intensity_matrix = _runs_as_rows(intensity_df[run_columns])

    # correct() overwrites intensity_matrix in place, the corrected values are read back from it
    summary = IntensityDriftCorrection().correct(
        intensity_matrix,
        _runs_as_rows(correlation_df[run_columns]),
        _rt_by_run(
            psm_df, PRECURSOR_IDX_COLUMN, unique_precursor_idx, run_columns, np.float32
        ),
        ion_precursor_codes.astype(np.uintp),
    )
    _log_summary(run_columns, summary)

    corrected_df = intensity_df.copy(deep=False)
    corrected_df[run_columns] = intensity_matrix.T
    return corrected_df


def correct_precursor_intensity_drift(
    precursor_df: pd.DataFrame, psm_df: pd.DataFrame
) -> pd.DataFrame:
    """Return `precursor_df` with the RT and precursor mass dependent intensity drift of every run column removed.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Linear precursor quantities, one row per precursor, with mod_seq_charge_hash, the other
        quantification level columns and one column per run. 0 or NaN marks a missing value.
    psm_df : pd.DataFrame
        PSM table with mod_seq_charge_hash, run, rt_observed, mz_library and charge.

    Returns
    -------
    pd.DataFrame
        Copy of precursor_df with corrected float64 run columns. The inputs are not changed.

    Raises
    ------
    ValueError
        If `psm_df` lacks one of the required columns.
    """
    _validate_psm_columns(psm_df, _PRECURSOR_REQUIRED_PSM_COLUMNS)
    run_columns = get_run_columns(precursor_df)
    precursor_keys = precursor_df[QuantificationLevelKey.PRECURSOR].to_numpy()
    intensity_matrix = np.ascontiguousarray(
        precursor_df[run_columns].to_numpy(dtype=np.float64).T
    )

    # correct() overwrites intensity_matrix in place, the corrected values are read back from it
    summary = PrecursorDriftCorrection().correct(
        intensity_matrix,
        _rt_by_run(
            psm_df,
            QuantificationLevelKey.PRECURSOR,
            precursor_keys,
            run_columns,
            np.float64,
        ),
        _precursor_mass(psm_df, precursor_keys),
    )
    _log_precursor_summary(run_columns, summary)

    corrected_df = precursor_df.copy(deep=False)
    corrected_df[run_columns] = intensity_matrix.T
    return corrected_df


def _precursor_mass(psm_df: pd.DataFrame, precursor_keys: np.ndarray) -> np.ndarray:
    """Library m/z times charge of every precursor key, NaN for a key without a psm."""
    psm_mass = psm_df[CalibCols.MZ_LIBRARY].to_numpy(np.float64) * psm_df[
        _CHARGE_COLUMN
    ].to_numpy(np.float64)
    mass_by_key = (
        pd.Series(psm_mass, index=psm_df[QuantificationLevelKey.PRECURSOR].to_numpy())
        .groupby(level=0)
        .first()
    )
    return np.ascontiguousarray(
        mass_by_key.reindex(precursor_keys).to_numpy(np.float64)
    )


def _runs_as_rows(df: pd.DataFrame) -> np.ndarray:
    """Return `df` as a C-contiguous float32 matrix of shape `(n_columns, n_rows)`."""
    return np.ascontiguousarray(df.to_numpy(dtype=np.float32).T)


def _rt_by_run(
    psm_df: pd.DataFrame,
    key_column: str,
    unique_keys: np.ndarray,
    run_columns: list[str],
    dtype: type,
) -> np.ndarray:
    """Observed RT of shape `(n_runs, len(unique_keys))`, NaN where the run has no psm of the key."""
    rt = np.full((len(run_columns), len(unique_keys)), np.nan, dtype=dtype)
    psm_run_codes = pd.Index(run_columns).get_indexer(psm_df[_RUN_COLUMN])
    psm_key_codes = pd.Index(unique_keys).get_indexer(psm_df[key_column].to_numpy())
    # psms of runs or keys outside the quantity table have no cell
    found_mask = (psm_run_codes >= 0) & (psm_key_codes >= 0)
    rt[psm_run_codes[found_mask], psm_key_codes[found_mask]] = psm_df[
        CalibCols.RT_OBSERVED
    ].to_numpy(dtype)[found_mask]
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
            logger.debug(f"Run {run}: not corrected, too few usable ions for a curve")


def _log_precursor_summary(run_columns: list[str], summary: dict) -> None:
    corrected_mask = summary[_SUMMARY_CORRECTED]
    max_abs_correction = summary[_SUMMARY_MAX_ABS_CORRECTION]
    logger.info(
        f"Corrected the RT and precursor mass dependent intensity drift of {corrected_mask.sum()} of "
        f"{len(run_columns)} runs in {summary[_SUMMARY_N_STRATA]} mass strata, "
        f"largest correction {max_abs_correction.max(initial=0.0):.3f} log2"
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
                f"Run {run}: not corrected, a single run or too few precursors quantified in every run"
            )

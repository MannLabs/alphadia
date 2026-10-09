"""Correction of the retention-time dependent (shape-only) intensity drift of every run.

The estimator lives in Rust (`alphadia_search_rs.IntensityDriftCorrection`). What is left
here is the dataframe bookkeeping it needs: placing every ion in retention time per run,
handing the matrices over run-major and writing the corrected intensities back.
"""

import logging

import numpy as np
import pandas as pd
from alphadia_search_rs import DriftCurve
from alphadia_search_rs import IntensityDriftCorrection as _RustIntensityDriftCorrection

from alphadia.constants.keys import CalibCols, ConstantsClass
from alphadia.outputtransform.quantification.quant_builder import (
    ION_COLUMN,
    PRECURSOR_IDX_COLUMN,
    get_run_columns,
)

logger = logging.getLogger()

RUN_COLUMN = "run"


class IntensityDriftStatistic(metaclass=ConstantsClass):
    """Per-bin location estimators of the intensity drift correction."""

    MODE: str = "mode"
    MEDIAN: str = "median"


def _run_major(df: pd.DataFrame) -> np.ndarray:
    """Return the run columns of `df` as a C-contiguous `(n_runs, n_ions)` matrix.

    The estimator works per run and therefore takes the matrices run-major. pandas hands
    the run columns of a single-dtype frame out as one F-contiguous block, which makes the
    transpose a view rather than a copy.
    """
    return np.ascontiguousarray(df.to_numpy(dtype=np.float64).T)


class IntensityDriftCorrector:
    """Remove the retention-time dependent part of every run's intensity deviation.

    Intensities can drift along the gradient in a run-specific way (e.g. one condition
    being 0.3-0.6 log2 low after 1000 s). Per run, the log2 deviation from the across-run
    level of every ion is binned along RT over well-measured ions, the per-bin location is
    interpolated to every ion and divided out.

    The curve is re-centred to zero mean over the run's ions before it is applied, so that
    only the *shape* along RT is removed. The global level of a run is deliberately left
    to the downstream LFQ normalization; removing it here would silently force the majority
    species to ratio 1.

    Parameters
    ----------
    psm_df : pd.DataFrame
        PSM table with `precursor_idx`, `run` and `rt_observed`, used to place every
        fragment ion in RT for each run.
    bin_seconds : float
        Width of the fixed RT bins the curve is estimated in.
    min_correlation : float
        Minimum mean cross-run fragment correlation for an ion to contribute to the curve.
    statistic : str
        Per-bin location estimator, see `IntensityDriftStatistic`.
    min_observations : int
        Minimum number of runs an ion must be observed in to contribute to the curve.
    min_ions_per_bin : int
        Consecutive sparse bins are merged until they hold at least this many ions.
    min_ions_per_run : int
        Runs with fewer usable ions are left unchanged.
    """

    DEFAULT_BIN_SECONDS = 30.0
    DEFAULT_MIN_CORRELATION = 0.8
    DEFAULT_MIN_OBSERVATIONS = 3
    DEFAULT_MIN_IONS_PER_BIN = 100
    DEFAULT_MIN_IONS_PER_RUN = 500

    N_LOGGED_CURVE_POINTS = 10

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        psm_df: pd.DataFrame,
        bin_seconds: float = DEFAULT_BIN_SECONDS,
        min_correlation: float = DEFAULT_MIN_CORRELATION,
        statistic: str = IntensityDriftStatistic.MODE,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        min_ions_per_bin: int = DEFAULT_MIN_IONS_PER_BIN,
        min_ions_per_run: int = DEFAULT_MIN_IONS_PER_RUN,
    ):
        self._psm_df = psm_df
        self._correction = _RustIntensityDriftCorrection(
            bin_seconds=bin_seconds,
            min_correlation=min_correlation,
            min_observations=min_observations,
            min_ions_per_bin=min_ions_per_bin,
            min_ions_per_run=min_ions_per_run,
            statistic=statistic,
        )

    def correct(
        self, intensity_df: pd.DataFrame, correlation_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Divide the RT-dependent drift out of every run column.

        Parameters
        ----------
        intensity_df : pd.DataFrame
            Linear fragment intensities with columns precursor_idx, ion, run1, run2, ...
            and the quantification level columns. Zero marks a missing value.
        correlation_df : pd.DataFrame
            Fragment correlations with the same ions and run columns as intensity_df.

        Returns
        -------
        pd.DataFrame
            Copy of intensity_df with corrected run columns. Returned unchanged if the
            psm table cannot place the ions in RT.
        """
        missing_columns = {
            PRECURSOR_IDX_COLUMN,
            RUN_COLUMN,
            CalibCols.RT_OBSERVED,
        } - set(self._psm_df.columns)
        if missing_columns:
            logger.warning(
                f"Skipping intensity drift correction, the psm table lacks the columns {sorted(missing_columns)}"
            )
            return intensity_df

        logger.info("Correcting RT-dependent intensity drift per run")

        run_columns = get_run_columns(intensity_df)
        intensity = _run_major(intensity_df[run_columns])
        correlation = _run_major(
            self._correlation_by_run(
                correlation_df, intensity_df[ION_COLUMN], run_columns
            )
        )
        rt = _run_major(
            self._rt_by_run(intensity_df[PRECURSOR_IDX_COLUMN], run_columns)
        )

        corrected, curves = self._correction.correct(intensity, correlation, rt)
        for run, curve in zip(run_columns, curves, strict=True):
            self._log_curve(run, curve)

        result_df = intensity_df.copy()
        result_df[run_columns] = corrected.T
        return result_df

    @staticmethod
    def _correlation_by_run(
        correlation_df: pd.DataFrame, ion: pd.Series, run_columns: list[str]
    ) -> pd.DataFrame:
        """Cross-run correlation of every ion of the intensity table, in its row order."""
        return correlation_df.set_index(ION_COLUMN)[run_columns].reindex(ion.to_numpy())

    def _rt_by_run(
        self, precursor_idx: pd.Series, run_columns: list[str]
    ) -> pd.DataFrame:
        """Observed RT of every ion's precursor in every run, NaN where a run has no psm entry.

        The estimator places an ion the run has no entry for at its mean RT over the other
        runs, so a cell with an intensity but no psm entry is still corrected.
        """
        return (
            self._psm_df[[PRECURSOR_IDX_COLUMN, RUN_COLUMN, CalibCols.RT_OBSERVED]]
            .astype({PRECURSOR_IDX_COLUMN: np.int64})
            .drop_duplicates([PRECURSOR_IDX_COLUMN, RUN_COLUMN])
            .pivot(
                index=PRECURSOR_IDX_COLUMN,
                columns=RUN_COLUMN,
                values=CalibCols.RT_OBSERVED,
            )
            .reindex(index=precursor_idx.to_numpy(np.int64), columns=run_columns)
        )

    def _log_curve(self, run: str, curve: DriftCurve | None) -> None:
        """Log the correction of one run, sub-sampled to a readable number of bins."""
        if curve is None:
            logger.info(
                f"Intensity drift correction: leaving {run} unchanged, too few usable ions "
                "or a single RT bin"
            )
            return

        n_bins = len(curve.rt)
        logged_bins = (
            np.linspace(0, n_bins - 1, min(n_bins, self.N_LOGGED_CURVE_POINTS))
            .round()
            .astype(int)
        )
        points = ", ".join(
            f"{curve.rt[index]:.0f}s: {curve.correction[index]:+.3f}"
            for index in logged_bins
        )
        logger.info(
            f"Intensity drift correction {run}: {curve.n_ions} ions in {n_bins} bins, "
            f"log2 correction [{points}]"
        )

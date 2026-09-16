"""Correction of the retention-time dependent (shape-only) intensity drift of every run."""

import logging

import numpy as np
import pandas as pd

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


class IntensityDriftCorrector:
    """Remove the retention-time dependent part of every run's intensity deviation.

    Intensities can drift along the gradient in a run-specific way (e.g. one condition
    being 0.3-0.6 log2 low after 1000 s). Per run, the log2 deviation from the
    across-run median is binned along RT over well-measured ions, the per-bin location
    is interpolated to every ion and divided out.

    The curve is re-centred to zero mean over the run's ions before it is applied, so
    only the *shape* along RT is removed. The global level of a run is deliberately left
    to the downstream LFQ normalization; removing it here would silently force the
    majority species to ratio 1.

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
    min_obs : int
        Minimum number of runs an ion must be observed in to contribute to the curve.
    min_ions_per_bin : int
        Consecutive sparse bins are merged until they hold at least this many ions.
    min_ions_per_run : int
        Runs with fewer usable ions are left unchanged.
    """

    DEFAULT_BIN_SECONDS = 30.0
    DEFAULT_MIN_CORRELATION = 0.8
    DEFAULT_MIN_OBS = 3
    DEFAULT_MIN_IONS_PER_BIN = 100
    DEFAULT_MIN_IONS_PER_RUN = 500

    # histogram the mode is read from: bin width in log2, moving-average window and the
    # percentile range that keeps outliers from stretching the histogram
    MODE_BIN_WIDTH = 0.02
    MODE_SMOOTHING_BINS = 11
    MODE_PERCENTILE_LOW = 2
    MODE_PERCENTILE_HIGH = 98
    CURVE_SMOOTHING_BINS = 3
    N_LOGGED_CURVE_POINTS = 10

    def __init__(
        self,
        psm_df: pd.DataFrame,
        bin_seconds: float = DEFAULT_BIN_SECONDS,
        min_correlation: float = DEFAULT_MIN_CORRELATION,
        statistic: str = IntensityDriftStatistic.MODE,
        min_obs: int = DEFAULT_MIN_OBS,
        min_ions_per_bin: int = DEFAULT_MIN_IONS_PER_BIN,
        min_ions_per_run: int = DEFAULT_MIN_IONS_PER_RUN,
    ):
        if statistic not in IntensityDriftStatistic.get_values():
            raise ValueError(
                f"Unknown intensity drift statistic '{statistic}', "
                f"expected one of {IntensityDriftStatistic.get_values()}"
            )
        self.psm_df = psm_df
        self.bin_seconds = bin_seconds
        self.min_correlation = min_correlation
        self.statistic = statistic
        # the across-run reference needs at least one observation
        self.min_obs = max(min_obs, 1)
        self.min_ions_per_bin = min_ions_per_bin
        self.min_ions_per_run = min_ions_per_run

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
        } - set(self.psm_df.columns)
        if missing_columns:
            logger.warning(
                f"Skipping intensity drift correction, the psm table lacks the columns {sorted(missing_columns)}"
            )
            return intensity_df

        logger.info("Correcting RT-dependent intensity drift per run")

        run_columns = get_run_columns(intensity_df)
        values = intensity_df[run_columns].to_numpy(dtype=np.float64, copy=True)
        observed = values > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = np.where(observed, np.log2(values), np.nan)

        quality = self._ion_quality(correlation_df, intensity_df, run_columns, observed)
        curve_ions = (observed.sum(axis=1) >= self.min_obs) & (
            quality >= self.min_correlation
        )
        reference = np.full(len(intensity_df), np.nan)
        reference[curve_ions] = np.nanmedian(log_values[curve_ions], axis=1)

        rt_matrix = self._rt_matrix(
            intensity_df[PRECURSOR_IDX_COLUMN].to_numpy(), run_columns
        )

        for j, run in enumerate(run_columns):
            curve = self._run_curve(
                run, log_values[:, j] - reference, rt_matrix[:, j], observed[:, j]
            )
            if curve is None:
                continue
            values[observed[:, j], j] /= 2 ** curve[observed[:, j]]

        result_df = intensity_df.copy()
        result_df[run_columns] = values
        return result_df

    @staticmethod
    def _ion_quality(
        correlation_df: pd.DataFrame,
        intensity_df: pd.DataFrame,
        run_columns: list[str],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Mean cross-run correlation of every ion over the runs it was observed in."""
        correlation = (
            correlation_df.set_index(ION_COLUMN)[run_columns]
            .reindex(intensity_df[ION_COLUMN].to_numpy())
            .to_numpy(dtype=np.float64)
        )
        correlation = np.where(observed & np.isfinite(correlation), correlation, 0.0)
        n_obs = observed.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            quality = correlation.sum(axis=1) / n_obs
        return np.where(n_obs > 0, quality, 0.0)

    def _rt_matrix(
        self, precursor_idx: np.ndarray, run_columns: list[str]
    ) -> np.ndarray:
        """Observed RT of every ion's precursor in every run (ions x runs).

        A precursor without an identification in a run is placed at its mean RT over
        the other runs, so a cell with an intensity but no psm entry is still corrected.
        """
        rt_by_run = (
            self.psm_df[[PRECURSOR_IDX_COLUMN, RUN_COLUMN, CalibCols.RT_OBSERVED]]
            .astype({PRECURSOR_IDX_COLUMN: np.int64})
            .drop_duplicates([PRECURSOR_IDX_COLUMN, RUN_COLUMN])
            .pivot(
                index=PRECURSOR_IDX_COLUMN,
                columns=RUN_COLUMN,
                values=CalibCols.RT_OBSERVED,
            )
            .reindex(index=precursor_idx.astype(np.int64), columns=run_columns)
            .to_numpy(dtype=np.float64)
        )
        n_finite = np.isfinite(rt_by_run).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_rt = np.nansum(rt_by_run, axis=1, keepdims=True) / n_finite
        return np.where(np.isfinite(rt_by_run), rt_by_run, mean_rt)

    def _run_curve(
        self,
        run: str,
        deviation: np.ndarray,
        rt: np.ndarray,
        observed_in_run: np.ndarray,
    ) -> np.ndarray | None:
        """Re-centred log2 correction of one run at every ion's RT, None if the run is skipped."""
        usable = np.isfinite(deviation) & np.isfinite(rt)
        n_usable = int(usable.sum())
        if n_usable < self.min_ions_per_run:
            logger.info(
                f"Intensity drift correction: skipping {run}, {n_usable} usable ions "
                f"(minimum {self.min_ions_per_run})"
            )
            return None

        bin_rt, bin_location = self._binned_location(rt[usable], deviation[usable])
        if len(bin_rt) < 2:
            logger.info(
                f"Intensity drift correction: skipping {run}, ions fall into a single RT bin"
            )
            return None
        bin_location = self._smooth(bin_location, self.CURVE_SMOOTHING_BINS)

        # np.interp holds the end values constant outside the binned RT range
        curve = np.interp(rt, bin_rt, bin_location)
        # shape-only: the run's level is left to the LFQ normalization
        centre = curve[observed_in_run & np.isfinite(curve)].mean()
        curve = np.where(np.isfinite(curve), curve - centre, 0.0)

        self._log_curve(run, n_usable, bin_rt, bin_location - centre)
        return curve

    def _binned_location(
        self, rt: np.ndarray, deviation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Location of the deviation per RT bin.

        Fixed-width bins are merged with their successors until they hold at least
        `min_ions_per_bin` ions; trailing ions that do not fill a bin join the last one.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Mean RT and deviation location of every merged bin.
        """
        order = np.argsort(rt, kind="stable")
        rt, deviation = rt[order], deviation[order]

        bin_index = np.floor(rt / self.bin_seconds).astype(np.int64)
        fixed_bin_stops = np.append(np.flatnonzero(np.diff(bin_index)) + 1, len(rt))

        edges = [0]
        for stop in fixed_bin_stops:
            if stop - edges[-1] >= self.min_ions_per_bin:
                edges.append(int(stop))
        if edges[-1] != len(rt):
            if len(edges) == 1:
                edges.append(len(rt))
            else:
                edges[-1] = len(rt)

        bin_rt = np.array([rt[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        bin_location = np.array(
            [self._location(deviation[a:b]) for a, b in zip(edges[:-1], edges[1:])]
        )
        return bin_rt, bin_location

    def _location(self, deviation: np.ndarray) -> float:
        if self.statistic == IntensityDriftStatistic.MEDIAN:
            return float(np.median(deviation))
        return self._mode(deviation)

    def _mode(self, deviation: np.ndarray) -> float:
        """Mode from a smoothed histogram between the 2nd and 98th percentile."""
        low = float(np.percentile(deviation, self.MODE_PERCENTILE_LOW))
        high = float(np.percentile(deviation, self.MODE_PERCENTILE_HIGH))
        n_bins = int(np.ceil((high - low) / self.MODE_BIN_WIDTH))
        # a distribution narrower than the smoothing window has no resolvable mode
        if n_bins < self.MODE_SMOOTHING_BINS:
            return float(np.median(deviation))

        counts, edges = np.histogram(deviation, bins=n_bins, range=(low, high))
        # zero padding pulls the ends of the histogram down so the peak is not read off
        # the trimmed edges
        smoothed = np.convolve(counts, np.ones(self.MODE_SMOOTHING_BINS), mode="same")
        centres = (edges[:-1] + edges[1:]) / 2
        return float(centres[np.argmax(smoothed)])

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        """Moving average that only averages over the neighbours inside the array.

        Unlike zero padding this keeps the ends of the curve unbiased, which matters at
        the start and end of the gradient where the drift is usually largest.
        """
        kernel = np.ones(window)
        weights = np.convolve(np.ones(len(values)), kernel, mode="same")
        return np.convolve(values, kernel, mode="same") / weights

    def _log_curve(
        self, run: str, n_ions: int, bin_rt: np.ndarray, bin_curve: np.ndarray
    ) -> None:
        n_points = min(len(bin_rt), self.N_LOGGED_CURVE_POINTS)
        idx = np.linspace(0, len(bin_rt) - 1, n_points).round().astype(int)
        points = ", ".join(
            f"{rt:.0f}s: {value:+.3f}" for rt, value in zip(bin_rt[idx], bin_curve[idx])
        )
        logger.info(
            f"Intensity drift correction {run}: {n_ions} ions in {len(bin_rt)} bins, "
            f"log2 correction [{points}]"
        )

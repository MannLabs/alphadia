from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import CalibrationPhase, SearchTolerance
from alphadia.constants.settings import (
    MAX_FRAGMENT_MZ_TOLERANCE,
    CalibrationLoop,
)
from alphadia.exceptions import NoOptimizationLockTargetError
from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import (
    CalibrationEstimators,
    CalibrationGroups,
    CalibrationManager,
)
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.optimizers.optimization_lock import OptimizationLock
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.recalibration_handler import RecalibrationHandler
from alphadia.workflow.peptidecentric.utils import log_precursor_df

ExtractionHandlerT = TypeVar("ExtractionHandlerT", bound=ExtractionHandler)

# tolerance name -> (calibration group, estimator, config key of the target tolerance in `search`)
_TOLERANCES = {
    SearchTolerance.MS2_ERROR: (
        CalibrationGroups.FRAGMENT,
        CalibrationEstimators.MZ,
        "target_ms2_tolerance",
    ),
    SearchTolerance.MS1_ERROR: (
        CalibrationGroups.PRECURSOR,
        CalibrationEstimators.MZ,
        "target_ms1_tolerance",
    ),
    SearchTolerance.RT_ERROR: (
        CalibrationGroups.PRECURSOR,
        CalibrationEstimators.RT,
        "target_rt_tolerance",
    ),
    SearchTolerance.MOBILITY_ERROR: (
        CalibrationGroups.PRECURSOR,
        CalibrationEstimators.MOBILITY,
        "target_mobility_tolerance",
    ),
}


@dataclass
class CalibrationCheckpoint:
    """The tolerances and classifier a calibrated round searched with, and the identification rate it reached."""

    id_rate: float
    tolerances: dict[str, float]
    classifier_version: int


@dataclass
class ExtractedBatch:
    """Extraction results for one batch of the optimization lock."""

    candidates_df: pd.DataFrame
    precursors_df: pd.DataFrame
    fragments_df: pd.DataFrame


class OptimizationHandler(ABC, Generic[ExtractionHandlerT]):
    """
    Handles the optimization of peptide-centric workflows.

    Each extraction backend has its own subclass, which extracts the batches of the optimization lock and performs FDR on them.
    """

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: Pipeline,
        spectral_library: SpecLibFlat,
        dia_data: "DiaData | DiaDataRust",  # noqa: F821
        create_extraction_handler: Callable[[SpecLibFlat], ExtractionHandlerT],
        figure_path: str | None = None,
    ):
        self._config = config
        self._optimization_manager = optimization_manager
        self._calibration_manager = calibration_manager
        self._fdr_manager = fdr_manager

        self._reporter = reporter
        self._spectral_library = spectral_library
        self._dia_data: DiaData | DiaDataRust = dia_data  # noqa: F821
        self._figure_path = figure_path
        self._create_extraction_handler = create_extraction_handler

        self._optlock: OptimizationLock = OptimizationLock(
            self._spectral_library, self._config
        )

        self._column_name_handler = ColumnNameHandler(
            self._calibration_manager,
            dia_data_has_ms1=self._dia_data.has_ms1,
            dia_data_has_mobility=self._dia_data.has_mobility,
        )

    def _active_tolerances(self) -> list[str]:
        """Names of the tolerances the raw file has data for."""
        names = [SearchTolerance.MS2_ERROR, SearchTolerance.RT_ERROR]
        if self._dia_data.has_ms1:
            names.append(SearchTolerance.MS1_ERROR)
        if self._dia_data.has_mobility:
            names.append(SearchTolerance.MOBILITY_ERROR)
        return names

    def _target_tolerance(self, name: str) -> float | None:
        """The fixed tolerance of a targeted parameter, None if the parameter is set from the data."""
        target = self._config["search"][_TOLERANCES[name][2]]
        if target <= 0:
            return None
        if name == SearchTolerance.RT_ERROR and target <= 1:
            # an RT target up to 1 is a share of the gradient
            return target * self._dia_data.rt_values.max()
        return target

    def _propose_tolerance(
        self, name: str, precursor_df: pd.DataFrame, fragments_df: pd.DataFrame
    ) -> float | None:
        """The tolerance that covers the calibrated residuals of the confident identifications.

        Targeted parameters take their target. Otherwise the tolerance is `TOLERANCE_FACTOR` times the
        `TOLERANCE_QUANTILE` interval of the residuals, so that it follows the spread of the true identifications
        rather than the tail of chance matches inside the current window. None if the estimator could not be fitted.
        """
        target = self._target_tolerance(name)
        if target is not None:
            return target
        group, estimator_name, _ = _TOLERANCES[name]
        df = fragments_df if group == CalibrationGroups.FRAGMENT else precursor_df
        estimator = self._calibration_manager.get_estimator(group, estimator_name)
        interval = estimator.ci(df, CalibrationLoop.TOLERANCE_QUANTILE)
        if interval is None:
            return None
        return CalibrationLoop.TOLERANCE_FACTOR[name] * interval

    def _get_phase(self, n_identified: int) -> str:
        """The calibration phase of the current round, see `search_parameter_optimization`."""
        if self._optlock.has_target_num_precursors:
            return CalibrationPhase.CONVERGE
        if n_identified < CalibrationLoop.WARM_UP_MIN_PRECURSORS:
            return CalibrationPhase.EXPAND
        return CalibrationPhase.WARM_UP

    def _propose_tolerances(
        self, names: list[str], precursor_df: pd.DataFrame, fragments_df: pd.DataFrame
    ) -> dict[str, float]:
        """Propose every tolerance; a tolerance whose estimator could not be fitted keeps its current value."""
        proposal = {}
        for name in names:
            tolerance = self._propose_tolerance(name, precursor_df, fragments_df)
            if tolerance is None:
                tolerance = self._optimization_manager.__dict__[name]
                self._reporter.log_string(
                    f"Calibration of {name} could not be fitted, keeping {tolerance:.2f}.",
                    verbosity="warning",
                )
            proposal[name] = tolerance
        return proposal

    def search_parameter_optimization(self) -> None:
        """Recalibrate and set the search tolerances in one loop.

        Each round searches the optimization lock, scores the result and then runs one of three phases, chosen by the
        number of precursors the lock holds at 1% FDR:

        - Expand (fewer than `WARM_UP_MIN_PRECURSORS`): too few to calibrate. The lock is extended by the next batch of
          the plan, the earlier batches are kept.
        - Warm-up (fewer than `optimization_lock_target`): recalibrate and set every tolerance from the residuals
          (see `_propose_tolerance`), shrinking each by at most `WARM_UP_MAX_SHRINK`. The lock grows to the next batch
          and is searched again from its first elution group.
        - Converge (at least `optimization_lock_target`): recalibrate and set every tolerance from the residuals. The
          lock is resized to the smallest batch expected to hold the target and searched again.

        Every round searched with a calibration is a checkpoint, scored by its identifications per extracted elution
        group, which compares across lock sizes. The loop stops when no tolerance changes by more than
        `CONVERGENCE_THRESHOLD` in converge, when a round falls more than `MAX_ID_LOSS` below the best checkpoint,
        after `max_steps` calibrations, or when the plan has no batches left before the target. In every case the
        main search takes the tolerances and classifier of the best checkpoint, with the latest calibration.
        """
        log_string = self._reporter.log_string
        max_calibrations = self._config["calibration"]["max_steps"]
        names = self._active_tolerances()

        recalibration_handler = RecalibrationHandler(
            self._config,
            self._optimization_manager,
            self._calibration_manager,
            self._reporter,
            self._figure_path,
            self._dia_data.has_ms1,
        )

        log_string("Starting initial search for precursors.", verbosity="progress")
        precursor_df = self._process_batch()
        best: CalibrationCheckpoint | None = None
        n_calibrations = 0
        while True:
            precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                precursor_df, self._optlock.fragments_df
            )
            n_identified = len(precursor_df_filtered)
            current = {
                name: self._optimization_manager.__dict__[name] for name in names
            }

            # score what was searched before deciding anything, so that no search goes unchecked
            searched_calibrated = n_calibrations > 0
            if searched_calibrated:
                checkpoint = CalibrationCheckpoint(
                    id_rate=n_identified / self._optlock.total_elution_groups,
                    tolerances=current,
                    classifier_version=self._fdr_manager.current_version,
                )
                if best is not None and checkpoint.id_rate < (
                    (1 - CalibrationLoop.MAX_ID_LOSS) * best.id_rate
                ):
                    log_string(
                        f"Identification rate dropped from {best.id_rate:.4f} to {checkpoint.id_rate:.4f}.",
                        verbosity="progress",
                    )
                    break
                if best is None or checkpoint.id_rate >= best.id_rate:
                    best = checkpoint
                if n_calibrations >= max_calibrations:
                    log_string(
                        f"Search parameters did not converge within {max_calibrations} calibrations.",
                        verbosity="warning",
                    )
                    break

            if (
                not self._optlock.has_target_num_precursors
                and not self._optlock.batches_remaining()
            ):
                if self._optlock.total_elution_groups == 0:
                    raise NoOptimizationLockTargetError(
                        "The search extracted no candidate from the raw file."
                    )
                log_string(
                    "Insufficient number of precursors to continue optimization.",
                    verbosity="warning",
                )
                break

            phase = self._get_phase(n_identified)

            if phase == CalibrationPhase.EXPAND:
                log_string(
                    f"Calibration round [{phase}]: {n_identified} identifications "
                    f"on {self._optlock.total_elution_groups} elution groups, too few to calibrate."
                )
                # after a calibration the new batch needs the calibrated columns of the earlier rounds
                self._optlock.extend(
                    self._calibration_manager if searched_calibrated else None
                )
                precursor_df = self._process_batch()
                continue

            recalibration_handler.recalibrate(
                precursor_df_filtered, fragments_df_filtered
            )
            n_calibrations += 1
            proposal = self._propose_tolerances(
                names, precursor_df_filtered, fragments_df_filtered
            )
            if phase == CalibrationPhase.WARM_UP:
                # few identifications give a noisy interval
                proposal = {
                    name: max(
                        proposal[name],
                        current[name] / CalibrationLoop.WARM_UP_MAX_SHRINK,
                    )
                    for name in names
                }
            log_string(
                f"Calibration round {n_calibrations} [{phase}]: {n_identified} identifications "
                f"on {self._optlock.total_elution_groups} elution groups; "
                + ", ".join(
                    f"{name} {current[name]:.2f} -> {proposal[name]:.2f}"
                    for name in names
                ),
                verbosity="progress",
            )

            # only tolerances that were searched with a calibration can have converged
            if (
                phase == CalibrationPhase.CONVERGE
                and searched_calibrated
                and all(
                    abs(proposal[name] - current[name]) / max(current[name], 1e-9)
                    <= CalibrationLoop.CONVERGENCE_THRESHOLD
                    for name in names
                )
            ):
                log_string("Search parameters converged.", verbosity="progress")
                break

            self._optimization_manager.update(**proposal)
            if phase == CalibrationPhase.CONVERGE:
                self._optlock.resize(self._calibration_manager)
            else:
                self._optlock.grow(self._calibration_manager)
            precursor_df = self._process_batch()

        if best is None:
            log_string(
                "No calibrated round to take forward, keeping the initial search parameters.",
                verbosity="warning",
            )
        else:
            self._optimization_manager.update(
                **best.tolerances, classifier_version=best.classifier_version
            )
        log_string(
            "Search parameter optimization finished. Values taken forward for search are:",
            verbosity="progress",
        )
        for name in names:
            log_string(
                f"{name:<15}: {self._optimization_manager.__dict__[name]:.4f}",
                verbosity="progress",
            )

    def _process_batch(self):
        """Extracts precursors and fragments from the spectral library, performs FDR correction and logs the precursor dataframe."""
        self._reporter.log_string(
            f"=== Extracting elution groups {self._optlock.start_idx} to {self._optlock.stop_idx} ===",
            verbosity="progress",
        )

        extraction_handler = self._create_extraction_handler(
            self._optlock.batch_library
        )

        batch = self._extract_batch(extraction_handler)
        self._optlock.update_with_extraction(batch.precursors_df, batch.fragments_df)

        self._reporter.log_string(
            f"=== Extracted {len(self._optlock.features_df)} precursors and {len(self._optlock.fragments_df)} fragments ===",
            verbosity="progress",
        )

        precursor_df = self._perform_fdr(extraction_handler, batch)

        self._optlock.update_with_fdr(precursor_df)

        self._reporter.log_string(
            f"=== FDR correction performed with classifier version {self._optimization_manager.classifier_version} ===",
        )

        log_precursor_df(self._reporter, precursor_df)

        return precursor_df

    @abstractmethod
    def _extract_batch(self, extraction_handler: ExtractionHandlerT) -> ExtractedBatch:
        """Select, score and quantify the candidates of the current batch library."""

    @abstractmethod
    def _perform_fdr(
        self, extraction_handler: ExtractionHandlerT, batch: ExtractedBatch
    ) -> pd.DataFrame:
        """Perform FDR on all precursors the optimization lock has collected so far.

        Returns
        -------
        pd.DataFrame
            Precursors with FDR results
        """

    def _filter_dfs(self, precursor_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """Filters precursor and fragment dataframes to extract the most reliable examples for calibration.

        Parameters
        ----------
        precursor_df : pd.DataFrame
            Precursor dataframe after FDR correction.

        fragments_df : pd.DataFrame
            Fragment dataframe.

        Returns
        -------
        precursor_df_filtered : pd.DataFrame
            Filtered precursor dataframe. Decoy precursors and those found at worse than 1% FDR are removed from the precursor dataframe.

        fragments_df_filtered : pd.DataFrame
            Filtered fragment dataframe. Retained fragments must either:
                1) have a correlation greater than 0.7 and belong to the top 5000 fragments sorted by correlation, if there are more than 500 with a correlation greater than 0.7, or
                2) belong to the top 500 fragments sorted by correlation otherwise.
            Fragments with abs(mass_error) greater than MAX_FRAGMENT_MZ_TOLERANCE (200) are removed.
        """
        qval_mask = precursor_df["qval"] < 0.01
        decoy_mask = precursor_df["decoy"] == 0
        precursor_df_filtered = precursor_df[qval_mask & decoy_mask]

        precursor_idx_mask = fragments_df["precursor_idx"].isin(
            precursor_df_filtered["precursor_idx"]
        )
        mass_error_mask = (
            np.abs(fragments_df["mass_error"]) <= MAX_FRAGMENT_MZ_TOLERANCE
        )

        fragments_df_filtered = fragments_df[
            precursor_idx_mask & mass_error_mask
        ].sort_values(
            by=["correlation", "precursor_idx"], ascending=False
        )  # last sort to break ties

        # Determine the number of fragments to keep
        high_corr_count = (
            fragments_df_filtered["correlation"]
            > self._config["calibration"]["min_correlation"]
        ).sum()
        stop_rank = min(
            high_corr_count,
            self._config["calibration"]["max_fragments"],
        )

        # Select top fragments
        fragments_df_filtered = fragments_df_filtered.head(stop_rank)

        self._reporter.log_string(
            f"fragments_df: keeping {len(fragments_df_filtered)} of {len(fragments_df)} [{sum(precursor_idx_mask)=} {sum(mass_error_mask)=} {stop_rank=}"
        )

        return precursor_df_filtered, fragments_df_filtered

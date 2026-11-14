from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.calibration.tolerance import (
    posterior_error_probability,
    true_match_interval,
)
from alphadia.constants.keys import SearchTolerance
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

# Maps each tolerance name to its calibration group, its estimator and the config key of its target in `search`.
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
        """Return the names of the tolerances that the raw file has data for."""
        names = [SearchTolerance.MS2_ERROR, SearchTolerance.RT_ERROR]
        if self._dia_data.has_ms1:
            names.append(SearchTolerance.MS1_ERROR)
        if self._dia_data.has_mobility:
            names.append(SearchTolerance.MOBILITY_ERROR)
        return names

    def _target_tolerance(self, name: str) -> float | None:
        """Return the target of a targeted parameter, or None for an automatic parameter."""
        target = self._config["search"][_TOLERANCES[name][2]]
        if target <= 0:
            return None
        if name == SearchTolerance.RT_ERROR and target <= 1:
            # The config defines an RT target of 1 or less as a share of the gradient length.
            return target * self._dia_data.rt_values.max()
        return target

    def _propose_tolerance(
        self, name: str, precursor_df: pd.DataFrame, fragments_df: pd.DataFrame
    ) -> float:
        """Propose a tolerance that covers the calibrated residuals of the identifications.

        A targeted parameter takes its target. An automatic parameter gets `TOLERANCE_FACTOR` times the
        `TOLERANCE_QUANTILE` interval of the residuals. The tolerance thus follows the spread of the true
        identifications and not the tail of chance matches in the current window.
        """
        target = self._target_tolerance(name)
        if target is not None:
            return target
        group, estimator_name, _ = _TOLERANCES[name]
        df = fragments_df if group == CalibrationGroups.FRAGMENT else precursor_df
        estimator = self._calibration_manager.get_estimator(group, estimator_name)
        return CalibrationLoop.TOLERANCE_FACTOR[name] * estimator.ci(
            df, CalibrationLoop.TOLERANCE_QUANTILE
        )

    def search_parameter_optimization(self) -> None:
        """Calibrate the library and set the tolerances in one loop.

        An identification is a precursor at 1% FDR.

        1. Extend the optimization lock until it holds `optimization_lock_target` identifications.
        2. Calibrate on the identifications and set every tolerance from their residuals, see `_propose_tolerance`.
           Search the lock again with the new calibration and tolerances.
        3. Stop when no tolerance changes by more than `CONVERGENCE_THRESHOLD`. When a round on the same lock finds
           more than `MAX_IDENTIFICATION_LOSS` fewer identifications than the best checkpoint, restore that checkpoint.
        4. After the tolerances converged, refit the classifier on the last search while it identifies more
           precursors and calibrate on the best refit, see `_converge_classifier`.
        5. An automatic RT tolerance for the main search holds `TOLERANCE_QUANTILE` of the true matches of the last
           round, weighted by the classifier, see `_true_match_tolerance`.
        """
        log_string = self._reporter.log_string
        config = self._config["calibration"]
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
        best_checkpoint = None
        calibrated = False
        restored = False
        for round_idx in range(config["max_steps"] + len(self._optlock.batch_plan)):
            precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                precursor_df, self._optlock.fragments_df
            )
            n_identified = len(precursor_df_filtered)
            at_target = self._optlock.has_target_num_precursors

            if not at_target:
                if self._optlock.batches_remaining():
                    log_string("Target number of precursors not reached yet.")
                    self._optlock.update()
                    if calibrated:
                        # The new batch needs the calibrated columns that the earlier rounds use.
                        self._optlock.update_with_calibration(self._calibration_manager)
                    precursor_df = self._process_batch()
                    continue
                if self._optlock.total_elution_groups == 0:
                    raise NoOptimizationLockTargetError(
                        "The search extracted no candidate from the raw file."
                    )
                log_string(
                    "Insufficient number of precursors to continue optimization.",
                    verbosity="warning",
                )
                if n_identified >= CalibrationLoop.MIN_PRECURSORS_TO_RECALIBRATE:
                    recalibration_handler.recalibrate(
                        precursor_df_filtered, fragments_df_filtered
                    )
                    self._optimization_manager.update(
                        **{
                            name: self._propose_tolerance(
                                name, precursor_df_filtered, fragments_df_filtered
                            )
                            for name in names
                        }
                    )
                break

            current_tolerances = {
                name: self._optimization_manager.__dict__[name] for name in names
            }
            # Only rounds on the same lock can compare identifications.
            # A smaller lock finds fewer identifications at the same tolerances.
            lock_range = (self._optlock.start_idx, self._optlock.stop_idx)
            if (
                best_checkpoint is not None
                and best_checkpoint["lock_range"] == lock_range
                and n_identified
                < (1 - CalibrationLoop.MAX_IDENTIFICATION_LOSS)
                * best_checkpoint["n_identified"]
            ):
                log_string(
                    f"Identifications dropped from {best_checkpoint['n_identified']} to {n_identified}, "
                    "restoring the best checkpoint.",
                    verbosity="progress",
                )
                recalibration_handler.recalibrate(*best_checkpoint["identifications"])
                self._optimization_manager.update(**best_checkpoint["parameters"])
                restored = True
                break
            if (
                best_checkpoint is None
                or best_checkpoint["lock_range"] != lock_range
                or n_identified >= best_checkpoint["n_identified"]
            ):
                best_checkpoint = {
                    "lock_range": lock_range,
                    "n_identified": n_identified,
                    "parameters": {
                        **current_tolerances,
                        "classifier_version": self._fdr_manager.current_version,
                    },
                    "identifications": (precursor_df_filtered, fragments_df_filtered),
                }

            recalibration_handler.recalibrate(
                precursor_df_filtered, fragments_df_filtered
            )
            calibrated = True
            next_tolerances = {
                name: self._propose_tolerance(
                    name, precursor_df_filtered, fragments_df_filtered
                )
                for name in names
            }
            log_string(
                f"Calibration round {round_idx}: {n_identified} identifications "
                f"on {self._optlock.total_elution_groups} elution groups; "
                + ", ".join(
                    f"{name} {current_tolerances[name]:.2f} -> {next_tolerances[name]:.2f}"
                    for name in names
                ),
                verbosity="progress",
            )

            if all(
                abs(next_tolerances[name] - current_tolerances[name])
                / max(current_tolerances[name], 1e-9)
                <= CalibrationLoop.CONVERGENCE_THRESHOLD
                for name in names
            ):
                log_string("Tolerances converged.", verbosity="progress")
                refit_identifications = self._converge_classifier(n_identified)
                if refit_identifications is not None:
                    recalibration_handler.recalibrate(*refit_identifications)
                break

            self._optimization_manager.update(**next_tolerances)
            self._optlock.update()
            self._optlock.update_with_calibration(self._calibration_manager)
            precursor_df = self._process_batch()

        else:
            log_string(
                f"Tolerances did not converge within {config['max_steps']} rounds, "
                "using the last calibration.",
                verbosity="warning",
            )

        if not restored:
            self._optimization_manager.update(
                classifier_version=self._fdr_manager.current_version
            )
        if calibrated and self._target_tolerance(SearchTolerance.RT_ERROR) is None:
            rt_tolerance = self._true_match_tolerance(SearchTolerance.RT_ERROR)
            if rt_tolerance is not None:
                rt_tolerance = max(
                    rt_tolerance,
                    CalibrationLoop.MIN_RT_TOLERANCE_CYCLES * self._cycle_duration(),
                )
                self._optimization_manager.update(
                    **{SearchTolerance.RT_ERROR: rt_tolerance}
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

    def _converge_classifier(
        self, n_identified: int
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Refit the classifier on the last search while it identifies more precursors.

        Each fit continues the training of the previous classifier, so the identifications of a round depend on how
        many rounds came before it. The score cutoff and the calibration of the main search come from the
        identifications of the last calibration. Converging the classifier on the last search makes them independent
        of the number of rounds the tolerances needed.

        Returns the filtered identifications of the best refit, or None when no refit identified more precursors.
        """
        identifications = None
        for _ in range(self._config["calibration"]["max_steps"]):
            precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                self._refit_classifier(), self._optlock.fragments_df
            )
            self._reporter.log_string(
                f"Classifier refit: {len(precursor_df_filtered)} identifications, before {n_identified}.",
                verbosity="progress",
            )
            if len(precursor_df_filtered) <= n_identified:
                break
            n_identified = len(precursor_df_filtered)
            identifications = (precursor_df_filtered, fragments_df_filtered)
        return identifications

    def _cycle_duration(self) -> float:
        """Return the mean duration of a DIA cycle in seconds."""
        cycle_length = self._dia_data.cycle.shape[1]
        return float(np.diff(self._dia_data.rt_values[::cycle_length]).mean())

    def _true_match_tolerance(self, name: str) -> float | None:
        """Estimate the tolerance that holds `TOLERANCE_QUANTILE` of the true matches of the last calibration round.

        The calibration rounds search wider than the residuals of the true matches, so the last round shows their
        tail. Its candidates are weighted by their posterior error probability, see `true_match_interval`. Returns
        None when the estimator is not fitted or the round holds too few likely true matches.
        """
        group, estimator_name, _ = _TOLERANCES[name]
        estimator = self._calibration_manager.get_estimator(group, estimator_name)
        if not estimator.is_fitted:
            return None
        precursor_df, window = self._last_fdr
        residuals = estimator.calc_deviation(precursor_df)[:, 2]
        is_decoy = precursor_df["decoy"].to_numpy() == 1
        # The classifier probability `proba` is the probability of a decoy.
        pep = posterior_error_probability(
            1 - precursor_df["proba"].to_numpy(), is_decoy
        )
        tolerance = true_match_interval(
            residuals[~is_decoy],
            pep[~is_decoy],
            window,
            CalibrationLoop.TOLERANCE_QUANTILE,
        )
        if tolerance is not None:
            self._reporter.log_string(
                f"{name} of the true matches: {tolerance:.2f} (searched {window:.2f}).",
                verbosity="progress",
            )
        return tolerance

    def _refit_classifier(self) -> pd.DataFrame:
        """Perform FDR on the last search again, with the classifier trained further."""
        precursor_df = self._perform_fdr(*self._last_search)
        self._last_fdr = (precursor_df, self._last_fdr[1])
        self._optlock.update_with_fdr(precursor_df)
        log_precursor_df(self._reporter, precursor_df)
        return precursor_df

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
        self._last_search = (extraction_handler, batch)
        self._optlock.update_with_extraction(batch.precursors_df, batch.fragments_df)

        self._reporter.log_string(
            f"=== Extracted {len(self._optlock.features_df)} precursors and {len(self._optlock.fragments_df)} fragments ===",
            verbosity="progress",
        )

        precursor_df = self._perform_fdr(extraction_handler, batch)
        self._last_fdr = (precursor_df, self._optimization_manager.rt_error)

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

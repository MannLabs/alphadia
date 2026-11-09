from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import CalibrationStrategy
from alphadia.constants.settings import MAX_FRAGMENT_MZ_TOLERANCE
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
from alphadia.workflow.optimizers.automatic import (
    AutomaticMobilityOptimizer,
    AutomaticMS1Optimizer,
    AutomaticMS2Optimizer,
    AutomaticOptimizer,
    AutomaticRTOptimizer,
)
from alphadia.workflow.optimizers.base import BaseOptimizer
from alphadia.workflow.optimizers.optimization_lock import OptimizationLock
from alphadia.workflow.optimizers.targeted import (
    TargetedMobilityOptimizer,
    TargetedMS1Optimizer,
    TargetedMS2Optimizer,
    TargetedOptimizer,
    TargetedRTOptimizer,
)
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.recalibration_handler import RecalibrationHandler
from alphadia.workflow.peptidecentric.utils import log_precursor_df

ExtractionHandlerT = TypeVar("ExtractionHandlerT", bound=ExtractionHandler)

# fewer identifications than this give no usable calibration fit
MIN_PRECURSORS_TO_RECALIBRATE = 6

# tolerance name -> (calibration group, estimator, config key of the target tolerance in `search`)
_TOLERANCES = {
    "ms2_error": (
        CalibrationGroups.FRAGMENT,
        CalibrationEstimators.MZ,
        "target_ms2_tolerance",
    ),
    "ms1_error": (
        CalibrationGroups.PRECURSOR,
        CalibrationEstimators.MZ,
        "target_ms1_tolerance",
    ),
    "rt_error": (
        CalibrationGroups.PRECURSOR,
        CalibrationEstimators.RT,
        "target_rt_tolerance",
    ),
    "mobility_error": (
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

    def _init_optimizer(
        self,
        clazz: type[AutomaticOptimizer | TargetedOptimizer],
        initial_parameter: float,
        target_parameter: float | None = None,
    ):
        """Returns an instance of the specified optimizer class with the given initial and target parameters."""
        if issubclass(clazz, TargetedOptimizer):
            return clazz(
                initial_parameter,
                target_parameter,
                self._config,
                self._optimization_manager,
                self._calibration_manager,
                self._fdr_manager,
                self._reporter,
            )
        if issubclass(clazz, AutomaticOptimizer):
            return clazz(
                initial_parameter,
                self._config,
                self._optimization_manager,
                self._calibration_manager,
                self._fdr_manager,
                self._optlock,
                self._reporter,
            )

        raise ValueError(f"Unknown Optimiser type: {clazz}")

    def _get_ordered_optimizers(self) -> list[list[BaseOptimizer]]:
        """Select appropriate optimizers. Targeted optimization is used if a valid target value (i.e. a number greater than 0) is specified in the config;
        if a value less than or equal to 0 is supplied, automatic optimization is used.
        Targeted optimizers are run simultaneously; automatic optimizers are run separately in the order MS2, RT, MS1, mobility.
        This order is built into the structure of the returned list of lists, ordered_optimizers.
        For MS1 and mobility, the relevant optimizer will be excluded from the returned list of lists if it is not present in the data.

        Returns
        -------
        ordered_optimizers : list
            List of lists of optimizers

        """
        config_search = self._config["search"]

        if config_search["target_ms2_tolerance"] > 0:
            ms2_optimizer = self._init_optimizer(
                TargetedMS2Optimizer,
                self._optimization_manager.ms2_error,
                config_search["target_ms2_tolerance"],
            )
        else:
            ms2_optimizer = self._init_optimizer(
                AutomaticMS2Optimizer, self._optimization_manager.ms2_error
            )

        if config_search["target_rt_tolerance"] > 0:
            gradient_length = self._dia_data.rt_values.max()
            target_rt_error = (
                config_search["target_rt_tolerance"]
                if config_search["target_rt_tolerance"] > 1
                else config_search["target_rt_tolerance"] * gradient_length
            )
            rt_optimizer = self._init_optimizer(
                TargetedRTOptimizer,
                self._optimization_manager.rt_error,
                target_rt_error,
            )
        else:
            rt_optimizer = self._init_optimizer(
                AutomaticRTOptimizer, self._optimization_manager.rt_error
            )

        if self._dia_data.has_ms1:
            if config_search["target_ms1_tolerance"] > 0:
                ms1_optimizer = self._init_optimizer(
                    TargetedMS1Optimizer,
                    self._optimization_manager.ms1_error,
                    config_search["target_ms1_tolerance"],
                )
            else:
                ms1_optimizer = self._init_optimizer(
                    AutomaticMS1Optimizer, self._optimization_manager.ms1_error
                )
        else:
            ms1_optimizer = None

        if self._dia_data.has_mobility:
            if config_search["target_mobility_tolerance"] > 0:
                mobility_optimizer = self._init_optimizer(
                    TargetedMobilityOptimizer,
                    self._optimization_manager.mobility_error,
                    config_search["target_mobility_tolerance"],
                )
            else:
                mobility_optimizer = self._init_optimizer(
                    AutomaticMobilityOptimizer,
                    self._optimization_manager.mobility_error,
                )
        else:
            mobility_optimizer = None

        if self._config["optimization"]["order_of_optimization"] is None:
            optimizers = [
                ms2_optimizer,
                rt_optimizer,
                ms1_optimizer,
                mobility_optimizer,
            ]
            targeted_optimizers = [
                [
                    optimizer
                    for optimizer in optimizers
                    if isinstance(optimizer, TargetedOptimizer)
                ]
            ]
            automatic_optimizers = [
                [optimizer]
                for optimizer in optimizers
                if isinstance(optimizer, AutomaticOptimizer)
            ]

            ordered_optimizers = (
                targeted_optimizers + automatic_optimizers
                if any(
                    targeted_optimizers
                )  # This line is required so no empty list is added to the ordered_optimizers list
                else automatic_optimizers
            )
        else:
            opt_mapping = {
                "ms2_error": ms2_optimizer,
                "rt_error": rt_optimizer,
                "ms1_error": ms1_optimizer,
                "mobility_error": mobility_optimizer,
            }
            ordered_optimizers = []
            for optimizers_in_ordering in self._config["optimization"][
                "order_of_optimization"
            ]:
                ordered_optimizers += [
                    [
                        opt_mapping[opt]
                        for opt in optimizers_in_ordering
                        if opt_mapping[opt] is not None
                    ]
                ]

        return ordered_optimizers

    def search_parameter_optimization(self):
        """Performs optimization of the search parameters.

        This occurs in two stages:
        1) Optimization lock: the data are searched to acquire a locked set of precursors which is used for search parameter optimization. The classifier is also trained during this stage.
        2) Optimization loop: the search parameters are optimized iteratively using the locked set of precursors.
            In each iteration, the data are searched with the locked library from stage 1, and the properties -- m/z for both precursors and fragments (i.e. MS1 and MS2), RT and mobility -- are recalibrated.
            The optimization loop is repeated for each list of optimizers in ordered_optimizers.

        """
        if self._config["calibration"]["strategy"] == CalibrationStrategy.CONVERGE:
            self._converge_search_parameters()
            return

        log_string = self._reporter.log_string

        ordered_optimizers = self._get_ordered_optimizers()

        log_string(
            "Starting initial search for precursors.",
            verbosity="progress",
        )

        recalibration_handler = RecalibrationHandler(
            self._config,
            self._optimization_manager,
            self._calibration_manager,
            self._reporter,
            self._figure_path,
            self._dia_data.has_ms1,
        )

        insufficient_precursors_to_optimize = False
        # Start of optimization/recalibration loop
        for optimizers in ordered_optimizers:
            if insufficient_precursors_to_optimize:
                break
            for current_step in range(
                self._config["calibration"]["max_steps"]
            ):  # Note current_step here refers to a different step than the attribute of the same name in the optimizer -- this should be rectified
                if np.all([optimizer.has_converged for optimizer in optimizers]):
                    log_string(
                        f"Optimization finished for {', '.join([optimizer.parameter_name for optimizer in optimizers])}.",
                        verbosity="progress",
                    )

                    self._optlock.reset_after_convergence(self._calibration_manager)

                    # TODO: re-enable, but save to disk
                    # currently yields `UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown`
                    # for optimizer in optimizers:
                    #     optimizer.plot()

                    break

                log_string(f"Starting optimization step {current_step}.")

                precursor_df = self._process_batch()

                if not self._optlock.has_target_num_precursors:
                    log_string("Target number of precursors not reached yet.")
                    if not self._optlock.batches_remaining():
                        if self._optlock.total_elution_groups == 0:
                            # The recovery path below measures the last round. Without
                            # a single elution group there is nothing to measure.
                            raise NoOptimizationLockTargetError(
                                "The search extracted no candidate from the raw file."
                            )
                        log_string(
                            "Insufficient number of precursors to continue optimization."
                        )
                        insufficient_precursors_to_optimize = True
                        break

                    self._optlock.update()

                    if self._optlock.previously_calibrated:
                        self._optlock.update_with_calibration(
                            self._calibration_manager
                        )  # This is needed so that the addition to the batch libary has the most recent calibration

                        self._skip_all_optimizers(optimizers)

                else:
                    log_string("Target number of precursors reached.")
                    precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                        precursor_df, self._optlock.fragments_df
                    )

                    self._optlock.update()
                    recalibration_handler.recalibrate(
                        precursor_df_filtered, fragments_df_filtered
                    )
                    self._optlock.update_with_calibration(self._calibration_manager)

                    if not self._optlock.previously_calibrated:  # Updates classifier but does not optimize the first time the target is reached.
                        # Optimization is more stable when done with calibrated values.
                        self._initiate_search_parameter_optimization()
                        continue

                    self._step_all_optimizers(
                        optimizers, precursor_df_filtered, fragments_df_filtered
                    )

            else:
                log_string(
                    f"Optimization did not converge within the maximum number of steps, which is {self._config['calibration']['max_steps']}.",
                    verbosity="warning",
                )

        log_string(
            "Search parameter optimization finished. Values taken forward for search are:",
            verbosity="progress",
        )
        log_string(
            "==============================================", verbosity="progress"
        )

        if insufficient_precursors_to_optimize:
            log_string("Handling insufficient precursors to optimize...")
            precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                precursor_df, self._optlock.fragments_df
            )
            if precursor_df_filtered.shape[0] >= MIN_PRECURSORS_TO_RECALIBRATE:
                recalibration_handler.recalibrate(
                    precursor_df_filtered, fragments_df_filtered
                )

            for optimizers in ordered_optimizers:
                for optimizer in optimizers:
                    optimizer.proceed_with_insufficient_precursors(
                        precursor_df_filtered, self._optlock.fragments_df
                    )

        for optimizers in ordered_optimizers:
            for optimizer in optimizers:
                log_string(
                    f"{optimizer.parameter_name:<15}: {self._optimization_manager.__dict__[optimizer.parameter_name]:.4f}",
                    verbosity="progress",
                )
        log_string(
            "==============================================", verbosity="progress"
        )

    def _active_tolerances(self) -> list[str]:
        """Names of the tolerances the raw file has data for."""
        names = ["ms2_error", "rt_error"]
        if self._dia_data.has_ms1:
            names.append("ms1_error")
        if self._dia_data.has_mobility:
            names.append("mobility_error")
        return names

    def _target_tolerance(self, name: str) -> float | None:
        """The fixed tolerance of a targeted parameter, None if the parameter is set from the data."""
        target = self._config["search"][_TOLERANCES[name][2]]
        if target <= 0:
            return None
        if name == "rt_error" and target <= 1:
            # an RT target up to 1 is a share of the gradient
            return target * self._dia_data.rt_values.max()
        return target

    def _propose_tolerance(
        self, name: str, precursor_df: pd.DataFrame, fragments_df: pd.DataFrame
    ) -> float:
        """The tolerance that covers the calibrated residuals of the confident identifications.

        Targeted parameters take their target. Otherwise the tolerance is `tolerance_factor` times the
        `tolerance_quantile` interval of the residuals, so that it follows the spread of the true identifications
        rather than the tail of chance matches inside the current window.
        """
        target = self._target_tolerance(name)
        if target is not None:
            return target
        group, estimator_name, _ = _TOLERANCES[name]
        config = self._config["calibration"]
        df = fragments_df if group == CalibrationGroups.FRAGMENT else precursor_df
        estimator = self._calibration_manager.get_estimator(group, estimator_name)
        return config["tolerance_factor"][name] * estimator.ci(
            df, config["tolerance_quantile"]
        )

    def _converge_search_parameters(self) -> None:
        """Recalibrate and set the search tolerances in one loop.

        1) Grow the optimization lock until it holds `min_calibration_precursors` precursors at 1% FDR.
        2) Recalibrate on the confident identifications and set every tolerance from their residuals
           (see `_propose_tolerance`); below `optimization_lock_target`, grow the lock and search it again from its
           first elution group with the new calibration and tolerances, otherwise search the same lock again.
        3) Stop when no tolerance changes by more than `convergence_threshold`, or restore the best round when a round
           on the same lock finds more than `max_id_loss` fewer identifications.
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
        best = None
        calibrated = False
        restored = False
        for step in range(config["max_steps"] + len(self._optlock.batch_plan)):
            precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                precursor_df, self._optlock.fragments_df
            )
            n_identified = len(precursor_df_filtered)
            at_target = self._optlock.has_target_num_precursors
            can_grow = self._optlock.batches_remaining()

            if (
                not at_target
                and can_grow
                and n_identified < config["min_calibration_precursors"]
            ):
                log_string("Too few precursors to calibrate yet.")
                self._optlock.update()
                if calibrated:
                    # the new batch needs the calibrated columns of the earlier rounds
                    self._optlock.update_with_calibration(self._calibration_manager)
                precursor_df = self._process_batch()
                continue
            if not at_target and not can_grow:
                if self._optlock.total_elution_groups == 0:
                    raise NoOptimizationLockTargetError(
                        "The search extracted no candidate from the raw file."
                    )
                log_string(
                    "Insufficient number of precursors to continue optimization.",
                    verbosity="warning",
                )
                if n_identified >= MIN_PRECURSORS_TO_RECALIBRATE:
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

            current = {
                name: self._optimization_manager.__dict__[name] for name in names
            }
            # identifications only compare on the same lock: a smaller lock finds fewer at the same tolerances
            lock_range = (self._optlock.start_idx, self._optlock.stop_idx)
            if (
                best is not None
                and best["lock_range"] == lock_range
                and n_identified < (1 - config["max_id_loss"]) * best["n_identified"]
            ):
                log_string(
                    f"Identifications dropped from {best['n_identified']} to {n_identified}, "
                    "restoring the best round.",
                    verbosity="progress",
                )
                recalibration_handler.recalibrate(*best["identifications"])
                self._optimization_manager.update(**best["parameters"])
                restored = True
                break
            if (
                best is None
                or best["lock_range"] != lock_range
                or n_identified >= best["n_identified"]
            ):
                best = {
                    "lock_range": lock_range,
                    "n_identified": n_identified,
                    "parameters": {
                        **current,
                        "classifier_version": self._fdr_manager.current_version,
                    },
                    "identifications": (precursor_df_filtered, fragments_df_filtered),
                }

            recalibration_handler.recalibrate(
                precursor_df_filtered, fragments_df_filtered
            )
            calibrated = True
            proposal = {
                name: self._propose_tolerance(
                    name, precursor_df_filtered, fragments_df_filtered
                )
                for name in names
            }
            if not at_target:
                # few identifications give a noisy interval: shrink by at most half per round until the target
                proposal = {
                    name: max(proposal[name], current[name] / 2) for name in names
                }
            log_string(
                f"Calibration round {step}: {n_identified} identifications "
                f"on {self._optlock.total_elution_groups} elution groups; "
                + ", ".join(
                    f"{name} {current[name]:.2f} -> {proposal[name]:.2f}"
                    for name in names
                ),
                verbosity="progress",
            )

            if at_target and all(
                abs(proposal[name] - current[name]) / max(current[name], 1e-9)
                <= config["convergence_threshold"]
                for name in names
            ):
                log_string("Search parameters converged.", verbosity="progress")
                break

            self._optimization_manager.update(**proposal)
            if at_target:
                self._optlock.update()
                self._optlock.update_with_calibration(self._calibration_manager)
            else:
                self._optlock.grow(self._calibration_manager)
            precursor_df = self._process_batch()

        else:
            log_string(
                f"Search parameters did not converge within {config['max_steps']} rounds, "
                "using the last calibration.",
                verbosity="warning",
            )

        if not restored:
            self._optimization_manager.update(
                classifier_version=self._fdr_manager.current_version
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

    def _initiate_search_parameter_optimization(self):
        """Saves the classifier version just before search parameter optimization begins and updates the optimization lock to show that calibration has been performed."""
        self._optlock.previously_calibrated = True
        self._optimization_manager.update(
            classifier_version=self._fdr_manager.current_version
        )
        self._reporter.log_string(
            "Required number of precursors found. Starting search parameter optimization.",
            verbosity="progress",
        )

    def _step_all_optimizers(
        self,
        optimizers: list[BaseOptimizer],
        precursor_df_filtered: pd.DataFrame,
        fragments_df_filtered: pd.DataFrame,
    ):
        """All optimizers currently in use are stepped and their current state is logged.

        Parameters
        ----------
        optimizers : list
            List of optimizers to be stepped.

        precursor_df_filtered : pd.DataFrame
            Filtered precursor dataframe (see filter_dfs).

        fragments_df_filtered : pd.DataFrame
            Filtered fragment dataframe (see filter_dfs).
        """
        self._reporter.log_string(
            "=== checking if optimization conditions were reached ===",
        )

        for optimizer in optimizers:
            optimizer.step(precursor_df_filtered, fragments_df_filtered)

        self._reporter.log_string(
            "==============================================",
        )

    def _skip_all_optimizers(
        self,
        optimizers: list[BaseOptimizer],
    ):
        """All optimizers currently in use are stepped and their current state is logged.

        Parameters
        ----------
        optimizers : list
            List of optimizers to be stepped.

        """
        self._reporter.log_string(
            "=== skipping optimization until target number of precursors are found ===",
        )

        for optimizer in optimizers:
            optimizer.skip()

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

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia.constants.settings import MAX_FRAGMENT_MZ_TOLERANCE
from alphadia.data.alpharaw_wrapper import AlphaRaw
from alphadia.data.bruker import TimsTOFTranspose
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationManager
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.optimization_lock import OptimizationLock
from alphadia.workflow.optimizers.automatic import (
    AutomaticMobilityOptimizer,
    AutomaticMS1Optimizer,
    AutomaticMS2Optimizer,
    AutomaticOptimizer,
    AutomaticRTOptimizer,
)
from alphadia.workflow.optimizers.base import BaseOptimizer
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
from alphadia.workflow.peptidecentric.utils import fdr_correction, log_precursor_df


class OptimizationHandler:
    """
    Handles the optimization of peptide-centric workflows.
    """

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: Pipeline,
        spectral_library: SpecLibBase,
        dia_data: AlphaRaw | TimsTOFTranspose,
        figure_path: str | None = None,
    ):
        self.config = config
        # TODO decouple optimizers from workflow:
        #  there's an implicit coupling between this class and the Optimizers, which think this is a workflow object!
        self.optimization_manager = optimization_manager
        self.calibration_manager = calibration_manager
        self.fdr_manager = fdr_manager

        self.reporter = reporter
        self.spectral_library = spectral_library
        self.dia_data = dia_data
        self.figure_path = figure_path

        self.optlock: OptimizationLock | None = None

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
        config_search = self.config["search"]

        if config_search["target_ms2_tolerance"] > 0:
            ms2_optimizer = TargetedMS2Optimizer(
                self.optimization_manager.ms2_error,
                config_search["target_ms2_tolerance"],
                self,
            )
        else:
            ms2_optimizer = AutomaticMS2Optimizer(
                self.optimization_manager.ms2_error,
                self,
            )

        if config_search["target_rt_tolerance"] > 0:
            gradient_length = self.dia_data.rt_values.max()
            target_rt_error = (
                config_search["target_rt_tolerance"]
                if config_search["target_rt_tolerance"] > 1
                else config_search["target_rt_tolerance"] * gradient_length
            )
            rt_optimizer = TargetedRTOptimizer(
                self.optimization_manager.rt_error,
                target_rt_error,
                self,
            )
        else:
            rt_optimizer = AutomaticRTOptimizer(
                self.optimization_manager.rt_error,
                self,
            )
        if self.dia_data.has_ms1:
            if config_search["target_ms1_tolerance"] > 0:
                ms1_optimizer = TargetedMS1Optimizer(
                    self.optimization_manager.ms1_error,
                    config_search["target_ms1_tolerance"],
                    self,
                )
            else:
                ms1_optimizer = AutomaticMS1Optimizer(
                    self.optimization_manager.ms1_error,
                    self,
                )
        else:
            ms1_optimizer = None
        if self.dia_data.has_mobility:
            if config_search["target_mobility_tolerance"] > 0:
                mobility_optimizer = TargetedMobilityOptimizer(
                    self.optimization_manager.mobility_error,
                    config_search["target_mobility_tolerance"],
                    self,
                )
            else:
                mobility_optimizer = AutomaticMobilityOptimizer(
                    self.optimization_manager.mobility_error,
                    self,
                )
        else:
            mobility_optimizer = None

        if self.config["optimization"]["order_of_optimization"] is None:
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
            for optimizers_in_ordering in self.config["optimization"][
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
        log_string = self.reporter.log_string

        # Get the order of optimization
        ordered_optimizers = self._get_ordered_optimizers()

        log_string(
            "Starting initial search for precursors.",
            verbosity="progress",
        )

        recalibration_handler = RecalibrationHandler(
            self.config,
            self.optimization_manager,
            self.calibration_manager,
            self.reporter,
            self.figure_path,
            self.dia_data.has_ms1,
        )

        self.optlock = OptimizationLock(self.spectral_library, self.config)

        insufficient_precursors_to_optimize = False
        # Start of optimization/recalibration loop
        for optimizers in ordered_optimizers:
            if insufficient_precursors_to_optimize:
                break
            for current_step in range(
                self.config["calibration"]["max_steps"]
            ):  # Note current_step here refers to a different step than the attribute of the same name in the optimizer -- this should be rectified
                if np.all([optimizer.has_converged for optimizer in optimizers]):
                    log_string(
                        f"Optimization finished for {', '.join([optimizer.parameter_name for optimizer in optimizers])}.",
                        verbosity="progress",
                    )

                    self.optlock.reset_after_convergence(self.calibration_manager)

                    for optimizer in optimizers:
                        optimizer.plot()

                    break

                log_string(f"Starting optimization step {current_step}.")

                precursor_df = self._process_batch()

                if not self.optlock.has_target_num_precursors:
                    if not self.optlock.batches_remaining():
                        insufficient_precursors_to_optimize = True
                        break

                    self.optlock.update()

                    if self.optlock.previously_calibrated:
                        self.optlock.update_with_calibration(
                            self.calibration_manager
                        )  # This is needed so that the addition to the batch libary has the most recent calibration

                        self._skip_all_optimizers(optimizers)

                else:
                    precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                        precursor_df, self.optlock.fragments_df
                    )

                    self.optlock.update()
                    recalibration_handler.recalibrate(
                        precursor_df_filtered, fragments_df_filtered
                    )
                    self.optlock.update_with_calibration(self.calibration_manager)

                    if not self.optlock.previously_calibrated:  # Updates classifier but does not optimize the first time the target is reached.
                        # Optimization is more stable when done with calibrated values.
                        self._initiate_search_parameter_optimization()
                        continue

                    self._step_all_optimizers(
                        optimizers, precursor_df_filtered, fragments_df_filtered
                    )

            else:
                log_string(
                    f"Optimization did not converge within the maximum number of steps, which is {self.config['calibration']['max_steps']}.",
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
            precursor_df_filtered, fragments_df_filtered = self._filter_dfs(
                precursor_df, self.optlock.fragments_df
            )
            if precursor_df_filtered.shape[0] >= 6:
                recalibration_handler.recalibrate(
                    precursor_df_filtered, fragments_df_filtered
                )

            for optimizers in ordered_optimizers:
                for optimizer in optimizers:
                    optimizer.proceed_with_insufficient_precursors(
                        precursor_df_filtered, self.optlock.fragments_df
                    )

        for optimizers in ordered_optimizers:
            for optimizer in optimizers:
                log_string(
                    f"{optimizer.parameter_name:<15}: {self.optimization_manager.__dict__[optimizer.parameter_name]:.4f}",
                    verbosity="progress",
                )
        log_string(
            "==============================================", verbosity="progress"
        )

    def _process_batch(self):
        """Extracts precursors and fragments from the spectral library, performs FDR correction and logs the precursor dataframe."""
        self.reporter.log_string(
            f"=== Extracting elution groups {self.optlock.start_idx} to {self.optlock.stop_idx} ===",
            verbosity="progress",
        )

        extraction_handler = ExtractionHandler(
            self.config,
            self.optimization_manager,
            self.reporter,
            ColumnNameHandler(
                self.optimization_manager,
                dia_data_has_ms1=self.dia_data.has_ms1,
                dia_data_has_mobility=self.dia_data.has_mobility,
            ),
            self.spectral_library,
        )

        feature_df, fragment_df = extraction_handler.extract_batch(
            self.dia_data,
            self.optlock.batch_library.precursor_df,
            self.optlock.batch_library.fragment_df,
        )
        self.optlock.update_with_extraction(feature_df, fragment_df)

        self.reporter.log_string(
            f"=== Extracted {len(self.optlock.features_df)} precursors and {len(self.optlock.fragments_df)} fragments ===",
            verbosity="progress",
        )

        precursor_df = fdr_correction(
            self.fdr_manager,
            self.config,
            self.dia_data.cycle,
            self.optlock.features_df,
            self.optlock.fragments_df,
            self.optimization_manager.classifier_version,
        )

        self.optlock.update_with_fdr(precursor_df)

        self.reporter.log_string(
            f"=== FDR correction performed with classifier version {self.optimization_manager.classifier_version} ===",
        )

        log_precursor_df(self.reporter, precursor_df)

        return precursor_df

    def _initiate_search_parameter_optimization(self):
        """Saves the classifier version just before search parameter optimization begins and updates the optimization lock to show that calibration has been performed."""
        self.optlock.previously_calibrated = True
        self.optimization_manager.update(
            classifier_version=self.fdr_manager.current_version
        )
        self.reporter.log_string(
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
        self.reporter.log_string(
            "=== checking if optimization conditions were reached ===",
        )

        for optimizer in optimizers:
            optimizer.step(precursor_df_filtered, fragments_df_filtered)

        self.reporter.log_string(
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
        self.reporter.log_string(
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
        ].sort_values(by="correlation", ascending=False)

        # Determine the number of fragments to keep
        high_corr_count = (
            fragments_df_filtered["correlation"]
            > self.config["calibration"]["min_correlation"]
        ).sum()
        stop_rank = min(
            high_corr_count,
            self.config["calibration"]["max_fragments"],
        )

        # Select top fragments
        fragments_df_filtered = fragments_df_filtered.head(stop_rank)

        self.reporter.log_string(
            f"fragments_df: keeping {len(fragments_df_filtered)} of {len(fragments_df)} [{sum(precursor_idx_mask)=} {sum(mass_error_mask)=} {stop_rank=}"
        )

        return precursor_df_filtered, fragments_df_filtered

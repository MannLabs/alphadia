import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia._fdrx.models.logistic_regression import LogisticRegressionClassifier
from alphadia._fdrx.models.two_step_classifier import TwoStepClassifier
from alphadia.constants.settings import MAX_FRAGMENT_MZ_TOLERANCE
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.fragcomp.utils import candidate_hash
from alphadia.workflow import base, optimization
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.recalibration_handler import RecalibrationHandler
from alphadia.workflow.peptidecentric.requantification_handler import (
    RequantificationHandler,
)

feature_columns = [
    "reference_intensity_correlation",
    "mean_reference_scan_cosine",
    "top3_reference_scan_cosine",
    "mean_reference_frame_cosine",
    "top3_reference_frame_cosine",
    "mean_reference_template_scan_cosine",
    "mean_reference_template_frame_cosine",
    "mean_reference_template_frame_cosine_rank",
    "mean_reference_template_scan_cosine_rank",
    "mean_reference_frame_cosine_rank",
    "mean_reference_scan_cosine_rank",
    "reference_intensity_correlation_rank",
    "top3_b_ion_correlation_rank",
    "top3_y_ion_correlation_rank",
    "top3_frame_correlation_rank",
    "fragment_frame_correlation_rank",
    "weighted_ms1_intensity_rank",
    "isotope_intensity_correlation_rank",
    "isotope_pattern_correlation_rank",
    "mono_ms1_intensity_rank",
    "weighted_mass_error_rank",
    "base_width_mobility",
    "base_width_rt",
    "rt_observed",
    "delta_rt",
    "mobility_observed",
    "mono_ms1_intensity",
    "top_ms1_intensity",
    "sum_ms1_intensity",
    "weighted_ms1_intensity",
    "weighted_mass_deviation",
    "weighted_mass_error",
    "mz_library",
    "mz_observed",
    "mono_ms1_height",
    "top_ms1_height",
    "sum_ms1_height",
    "weighted_ms1_height",
    "isotope_intensity_correlation",
    "isotope_height_correlation",
    "n_observations",
    "intensity_correlation",
    "height_correlation",
    "intensity_fraction",
    "height_fraction",
    "intensity_fraction_weighted",
    "height_fraction_weighted",
    "mean_observation_score",
    "sum_b_ion_intensity",
    "sum_y_ion_intensity",
    "diff_b_y_ion_intensity",
    "fragment_scan_correlation",
    "top3_scan_correlation",
    "fragment_frame_correlation",
    "top3_frame_correlation",
    "template_scan_correlation",
    "template_frame_correlation",
    "top3_b_ion_correlation",
    "top3_y_ion_correlation",
    "n_b_ions",
    "n_y_ions",
    "f_masked",
    "cycle_fwhm",
    "mobility_fwhm",
    "top_3_ms2_mass_error",
    "mean_ms2_mass_error",
    "n_overlapping",
    "mean_overlapping_intensity",
    "mean_overlapping_mass_error",
]


def _get_classifier_base(
    enable_two_step_classifier: bool = False,
    two_step_classifier_max_iterations: int = 5,
    enable_nn_hyperparameter_tuning: bool = False,
    fdr_cutoff: float = 0.01,
):
    """Creates and returns a classifier base instance.

    Parameters
    ----------
    enable_two_step_classifier : bool, optional
        If True, uses logistic regression + neural network.
        If False (default), uses only neural network.

    two_step_classifier_max_iterations : int
        Maximum number of iterations withtin .fit_predict() of the two-step classifier.

    enable_nn_hyperparameter_tuning: bool, optional
        If True, uses hyperparameter tuning for the neural network.
        If False (default), uses default hyperparameters for the neural network.

    fdr_cutoff : float, optional
        The FDR cutoff threshold used by the second classifier when two-step
        classification is enabled. Default is 0.01.

    Returns
    -------
    BinaryClassifierLegacyNewBatching | TwoStepClassifier
        Neural network or two-step classifier based on enable_two_step_classifier.
    """
    nn_classifier = BinaryClassifierLegacyNewBatching(
        test_size=0.001,
        batch_size=5000,
        learning_rate=0.001,
        epochs=10,
        experimental_hyperparameter_tuning=enable_nn_hyperparameter_tuning,
    )

    if enable_two_step_classifier:
        return TwoStepClassifier(
            first_classifier=LogisticRegressionClassifier(),
            second_classifier=nn_classifier,
            second_fdr_cutoff=fdr_cutoff,
            max_iterations=two_step_classifier_max_iterations,
        )
    else:
        return nn_classifier


class PeptideCentricWorkflow(base.WorkflowBase):
    def __init__(
        self,
        instance_name: str,
        config: Config,
        quant_path: str = None,
    ) -> None:
        super().__init__(
            instance_name,
            config,
            quant_path,
        )
        self.optlock: optimization.OptimizationLock | None = None

        self._extraction_handler: ExtractionHandler | None = None
        self._recalibration_handler: RecalibrationHandler | None = None
        self.requantification_handler: RequantificationHandler | None = None

    def load(
        self,
        dia_data_path: str,
        spectral_library: SpecLibBase,
    ) -> None:
        super().load(
            dia_data_path,
            spectral_library,
        )

        self.reporter.log_string(
            f"Initializing workflow {self._instance_name}", verbosity="progress"
        )

        self._init_fdr_manager()
        self._init_spectral_library()

        self._extraction_handler = ExtractionHandler(
            self.config,
            self.optimization_manager,
            self.reporter,
            self.spectral_library,
            rt_column=self._get_rt_column(),
            mobility_column=self._get_mobility_column(),
            precursor_mz_column=self._get_precursor_mz_column(),
            fragment_mz_column=self._get_fragment_mz_column(),
        )
        self._recalibration_handler = RecalibrationHandler(
            self.config,
            self.optimization_manager,
            self.calibration_manager,
            self.reporter,
            self._figure_path,
            self.dia_data.has_ms1,
        )

    def _init_fdr_manager(self):
        self.fdr_manager = FDRManager(
            feature_columns=feature_columns,
            classifier_base=_get_classifier_base(
                enable_two_step_classifier=self.config["fdr"][
                    "enable_two_step_classifier"
                ],
                two_step_classifier_max_iterations=self.config["fdr"][
                    "two_step_classifier_max_iterations"
                ],
                enable_nn_hyperparameter_tuning=self.config["fdr"][
                    "enable_nn_hyperparameter_tuning"
                ],
                fdr_cutoff=self.config["fdr"]["fdr"],
            ),
            figure_path=self._figure_path,
        )

    def _init_spectral_library(self):
        # apply channel filter
        if self.config["search"]["channel_filter"] == "":
            allowed_channels = self.spectral_library.precursor_df["channel"].unique()
        else:
            allowed_channels = [
                int(c) for c in self.config["search"]["channel_filter"].split(",")
            ]
            self.reporter.log_string(
                f"Applying channel filter using only: {allowed_channels}",
                verbosity="progress",
            )

        # normalize spectral library rt to file specific TIC profile
        self.spectral_library.precursor_df["rt_library"] = self._norm_to_rt(
            self.dia_data, self.spectral_library.precursor_df["rt_library"].values
        )

        # filter based on precursor observability
        lower_mz_limit = self.dia_data.cycle[self.dia_data.cycle > 0].min()
        upper_mz_limit = self.dia_data.cycle[self.dia_data.cycle > 0].max()

        precursor_before = np.sum(self.spectral_library.precursor_df["decoy"] == 0)
        self.spectral_library.precursor_df = self.spectral_library.precursor_df[
            (self.spectral_library.precursor_df["mz_library"] >= lower_mz_limit)
            & (self.spectral_library.precursor_df["mz_library"] <= upper_mz_limit)
        ]
        # self.spectral_library.remove_unused_fragmen
        precursor_after = np.sum(self.spectral_library.precursor_df["decoy"] == 0)
        precursor_removed = precursor_before - precursor_after
        self.reporter.log_string(
            f"{precursor_after:,} target precursors potentially observable ({precursor_removed:,} removed)",
            verbosity="progress",
        )

        # filter spectral library to only contain precursors from allowed channels
        # save original precursor_df for later use
        self.spectral_library.precursor_df_unfiltered = (
            self.spectral_library.precursor_df.copy()
        )
        self.spectral_library.precursor_df = (
            self.spectral_library.precursor_df_unfiltered[
                self.spectral_library.precursor_df_unfiltered["channel"].isin(
                    allowed_channels
                )
            ].copy()
        )

    def _norm_to_rt(
        self,
        dia_data,
        norm_values: np.ndarray,
        active_gradient_start: float | None = None,
        active_gradient_stop: float | None = None,
        mode=None,
    ):
        """Convert normalized retention time values to absolute retention time values.

        Parameters
        ----------
        dia_data : alphatims.bruker.TimsTOF
            TimsTOF object containing the DIA data.

        norm_values : np.ndarray
            Array of normalized retention time values.

        active_gradient_start : float, optional
            Start of the active gradient in seconds, by default None.
            If None, the value from the config is used.
            If not defined in the config, it is set to zero.

        active_gradient_stop : float, optional
            End of the active gradient in seconds, by default None.
            If None, the value from the config is used.
            If not defined in the config, it is set to the last retention time value.

        mode : str, optional
            Mode of the gradient, by default None.
            If None, the value from the config is used which should be 'tic' by default

        """

        # determine if the gradient start and stop are defined in the config
        if active_gradient_start is None:
            if "active_gradient_start" in self.config["calibration"]:
                lower_rt = self.config["calibration"]["active_gradient_start"]
            else:
                lower_rt = (
                    dia_data.rt_values[0]
                    + self.config["search_initial"]["initial_rt_tolerance"] / 2
                )
        else:
            lower_rt = active_gradient_start

        if active_gradient_stop is None:
            if "active_gradient_stop" in self.config["calibration"]:
                upper_rt = self.config["calibration"]["active_gradient_stop"]
            else:
                upper_rt = dia_data.rt_values[-1] - (
                    self.config["search_initial"]["initial_rt_tolerance"] / 2
                )
        else:
            upper_rt = active_gradient_stop

        # make sure values are really norm values
        norm_values = np.interp(
            norm_values, [norm_values.min(), norm_values.max()], [0, 1]
        )

        # determine the mode based on the config or the function parameter
        if mode is None:
            mode = self.config["calibration"].get("norm_rt_mode", "tic")
        else:
            mode = mode.lower()

        if mode == "linear":
            return np.interp(norm_values, [0, 1], [lower_rt, upper_rt])

        elif mode == "tic":
            raise NotImplementedError("tic mode is not implemented yet")

        else:
            raise ValueError(f"Unknown norm_rt_mode {mode}")

    def _get_precursor_mz_column(self):
        """Get the precursor m/z column name.
        This function will return `mz_calibrated` if precursor calibration has happened, otherwise it will return `mz_library`.
        If no MS1 data is present, it will always return `mz_library`.

        Returns
        -------
        str
            Name of the precursor m/z column

        """
        return (
            f"mz_{self.optimization_manager.column_type}"
            if self.dia_data.has_ms1
            else "mz_library"
        )

    def _get_fragment_mz_column(self):
        return f"mz_{self.optimization_manager.column_type}"

    def _get_rt_column(self):
        return f"rt_{self.optimization_manager.column_type}"

    def _get_mobility_column(self):
        return (
            f"mobility_{self.optimization_manager.column_type}"
            if self.dia_data.has_mobility
            else "mobility_library"
        )

    def _get_ordered_optimizers(self):
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
            ms2_optimizer = optimization.TargetedMS2Optimizer(
                self.optimization_manager.ms2_error,
                config_search["target_ms2_tolerance"],
                self,
            )
        else:
            ms2_optimizer = optimization.AutomaticMS2Optimizer(
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
            rt_optimizer = optimization.TargetedRTOptimizer(
                self.optimization_manager.rt_error,
                target_rt_error,
                self,
            )
        else:
            rt_optimizer = optimization.AutomaticRTOptimizer(
                self.optimization_manager.rt_error,
                self,
            )
        if self.dia_data.has_ms1:
            if config_search["target_ms1_tolerance"] > 0:
                ms1_optimizer = optimization.TargetedMS1Optimizer(
                    self.optimization_manager.ms1_error,
                    config_search["target_ms1_tolerance"],
                    self,
                )
            else:
                ms1_optimizer = optimization.AutomaticMS1Optimizer(
                    self.optimization_manager.ms1_error,
                    self,
                )
        else:
            ms1_optimizer = None
        if self.dia_data.has_mobility:
            if config_search["target_mobility_tolerance"] > 0:
                mobility_optimizer = optimization.TargetedMobilityOptimizer(
                    self.optimization_manager.mobility_error,
                    config_search["target_mobility_tolerance"],
                    self,
                )
            else:
                mobility_optimizer = optimization.AutomaticMobilityOptimizer(
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
                    if isinstance(optimizer, optimization.TargetedOptimizer)
                ]
            ]
            automatic_optimizers = [
                [optimizer]
                for optimizer in optimizers
                if isinstance(optimizer, optimization.AutomaticOptimizer)
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
        """Performs optimization of the search parameters. This occurs in two stages:
        1) Optimization lock: the data are searched to acquire a locked set of precursors which is used for search parameter optimization. The classifier is also trained during this stage.
        2) Optimization loop: the search parameters are optimized iteratively using the locked set of precursors.
            In each iteration, the data are searched with the locked library from stage 1, and the properties -- m/z for both precursors and fragments (i.e. MS1 and MS2), RT and mobility -- are recalibrated.
            The optimization loop is repeated for each list of optimizers in ordered_optimizers.

        """
        log_string = self.reporter.log_string
        # First check to see if the calibration has already been performed. Return if so.
        if (
            self.calibration_manager.is_fitted
            and self.calibration_manager.is_loaded_from_file
        ):
            log_string(
                "Skipping calibration as existing calibration was found",
                verbosity="progress",
            )
            return

        # Get the order of optimization
        ordered_optimizers = self._get_ordered_optimizers()

        log_string(
            "Starting initial search for precursors.",
            verbosity="progress",
        )

        self.optlock = optimization.OptimizationLock(self.spectral_library, self.config)

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
                    self._recalibration_handler.recalibrate(
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
                self._recalibration_handler.recalibrate(
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

        self._save_managers()

    def _process_batch(self):
        """Extracts precursors and fragments from the spectral library, performs FDR correction and logs the precursor dataframe."""
        self.reporter.log_string(
            f"=== Extracting elution groups {self.optlock.start_idx} to {self.optlock.stop_idx} ===",
            verbosity="progress",
        )

        feature_df, fragment_df = self._extraction_handler.extract_batch(
            self.dia_data,
            self.optlock.batch_library.precursor_df,
            self.optlock.batch_library.fragment_df,
        )
        self.optlock.update_with_extraction(feature_df, fragment_df)

        self.reporter.log_string(
            f"=== Extracted {len(self.optlock.features_df)} precursors and {len(self.optlock.fragments_df)} fragments ===",
            verbosity="progress",
        )

        precursor_df = self._fdr_correction(
            self.optlock.features_df,
            self.optlock.fragments_df,
            self.optimization_manager.classifier_version,
        )

        self.optlock.update_with_fdr(precursor_df)

        self.reporter.log_string(
            f"=== FDR correction performed with classifier version {self.optimization_manager.classifier_version} ===",
        )

        self.log_precursor_df(precursor_df)

        return precursor_df

    def _initiate_search_parameter_optimization(self):
        """Saves the classifier version just before search parameter optimization begins and updates the optimization lock to show that calibration has been performed."""
        self.optlock.previously_calibrated = True
        self.optimization_manager.fit(
            {"classifier_version": self.fdr_manager.current_version}
        )
        self.reporter.log_string(
            "Required number of precursors found. Starting search parameter optimization.",
            verbosity="progress",
        )

    def _step_all_optimizers(
        self,
        optimizers: list[optimization.BaseOptimizer],
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
        optimizers: list[optimization.BaseOptimizer],
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

    def _fdr_correction(self, features_df, df_fragments, version=-1):
        return self.fdr_manager.fit_predict(
            features_df,
            decoy_strategy="precursor_channel_wise"
            if self.config["fdr"]["channel_wise_fdr"]
            else "precursor",
            competetive=self.config["fdr"]["competetive_scoring"],
            df_fragments=df_fragments
            if self.config["search"]["compete_for_fragments"]
            else None,
            dia_cycle=self.dia_data.cycle,
            version=version,
        )

    def _save_managers(self):
        """Saves the calibration, optimization and FDR managers to disk so that they can be reused if needed.
        Note the timing manager is not saved at this point as it is saved with every call to it.
        The FDR manager is not saved because it is not used in subsequent parts of the workflow.
        """
        self.calibration_manager.save()
        self.optimization_manager.save()  # this replaces the .save() call when the optimization manager is fitted, since there seems little point in saving an intermediate optimization manager.

    def extraction(self):
        self.calibration_manager.predict(
            self.spectral_library.precursor_df, "precursor"
        )
        self.calibration_manager.predict(self.spectral_library._fragment_df, "fragment")

        features_df, fragments_df = self._extraction_handler.extract_batch(
            self.dia_data,
            self.spectral_library.precursor_df,
            self._spectral_library._fragment_df,
            apply_cutoff=True,
        )

        self.reporter.log_string(
            f"=== Performing FDR correction with classifier version {self.optimization_manager.classifier_version} ===",
        )

        precursor_df = self._fdr_correction(
            features_df, fragments_df, self.optimization_manager.classifier_version
        )

        precursor_df = precursor_df[precursor_df["qval"] <= self.config["fdr"]["fdr"]]

        self.reporter.log_string("Removing fragments below FDR threshold")

        # to be optimized later
        fragments_df["candidate_idx"] = candidate_hash(
            fragments_df["precursor_idx"].values, fragments_df["rank"].values
        )
        precursor_df["candidate_idx"] = candidate_hash(
            precursor_df["precursor_idx"].values, precursor_df["rank"].values
        )

        fragments_df = fragments_df[
            fragments_df["candidate_idx"].isin(precursor_df["candidate_idx"])
        ]

        self.log_precursor_df(precursor_df)

        return precursor_df, fragments_df

    def log_precursor_df(self, precursor_df):
        total_precursors = len(precursor_df)

        total_precursors_denom = max(
            float(total_precursors), 1e-6
        )  # avoid division by zero

        target_precursors = len(precursor_df[precursor_df["decoy"] == 0])
        target_precursors_percentages = target_precursors / total_precursors_denom * 100
        decoy_precursors = len(precursor_df[precursor_df["decoy"] == 1])
        decoy_precursors_percentages = decoy_precursors / total_precursors_denom * 100

        self.reporter.log_string(
            "============================= Precursor FDR =============================",
            verbosity="progress",
        )
        self.reporter.log_string(
            f"Total precursors accumulated: {total_precursors:,}", verbosity="progress"
        )
        self.reporter.log_string(
            f"Target precursors: {target_precursors:,} ({target_precursors_percentages:.2f}%)",
            verbosity="progress",
        )
        self.reporter.log_string(
            f"Decoy precursors: {decoy_precursors:,} ({decoy_precursors_percentages:.2f}%)",
            verbosity="progress",
        )

        self.reporter.log_string("", verbosity="progress")
        self.reporter.log_string("Precursor Summary:", verbosity="progress")

        for channel in precursor_df["channel"].unique():
            precursor_05fdr = len(
                precursor_df[
                    (precursor_df["qval"] < 0.05)
                    & (precursor_df["decoy"] == 0)
                    & (precursor_df["channel"] == channel)
                ]
            )
            precursor_01fdr = len(
                precursor_df[
                    (precursor_df["qval"] < 0.01)
                    & (precursor_df["decoy"] == 0)
                    & (precursor_df["channel"] == channel)
                ]
            )
            precursor_001fdr = len(
                precursor_df[
                    (precursor_df["qval"] < 0.001)
                    & (precursor_df["decoy"] == 0)
                    & (precursor_df["channel"] == channel)
                ]
            )
            self.reporter.log_string(
                f"Channel {channel:>3}:\t 0.05 FDR: {precursor_05fdr:>5,}; 0.01 FDR: {precursor_01fdr:>5,}; 0.001 FDR: {precursor_001fdr:>5,}",
                verbosity="progress",
            )

        self.reporter.log_string("", verbosity="progress")
        self.reporter.log_string("Protein Summary:", verbosity="progress")

        for channel in precursor_df["channel"].unique():
            proteins_05fdr = precursor_df[
                (precursor_df["qval"] < 0.05)
                & (precursor_df["decoy"] == 0)
                & (precursor_df["channel"] == channel)
            ]["proteins"].nunique()
            proteins_01fdr = precursor_df[
                (precursor_df["qval"] < 0.01)
                & (precursor_df["decoy"] == 0)
                & (precursor_df["channel"] == channel)
            ]["proteins"].nunique()
            proteins_001fdr = precursor_df[
                (precursor_df["qval"] < 0.001)
                & (precursor_df["decoy"] == 0)
                & (precursor_df["channel"] == channel)
            ]["proteins"].nunique()
            self.reporter.log_string(
                f"Channel {channel:>3}:\t 0.05 FDR: {proteins_05fdr:>5,}; 0.01 FDR: {proteins_01fdr:>5,}; 0.001 FDR: {proteins_001fdr:>5,}",
                verbosity="progress",
            )

        self.reporter.log_string(
            "=========================================================================",
            verbosity="progress",
        )

    def _lazy_init_requantification_handler(self):
        """Initializes the requantification handler if it is not already initialized."""
        if not self.requantification_handler:
            self.requantification_handler = RequantificationHandler(
                self.config,
                self.optimization_manager,
                self.calibration_manager,
                self.fdr_manager,
                self.reporter,
                self.spectral_library,
                rt_column=self._get_rt_column(),
                mobility_column=self._get_mobility_column(),
                precursor_mz_column=self._get_precursor_mz_column(),
                fragment_mz_column=self._get_fragment_mz_column(),
            )

    def requantify(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        self._lazy_init_requantification_handler()

        psm_df = self.requantification_handler.requantify(self.dia_data, psm_df)

        self.log_precursor_df(psm_df)

        return psm_df

    def requantify_fragments(
        self, psm_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._lazy_init_requantification_handler()

        return self.requantification_handler.requantify_fragments(self.dia_data, psm_df)

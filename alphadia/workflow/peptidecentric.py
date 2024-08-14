# native imports
import logging

# third party imports
import numpy as np
import pandas as pd
import seaborn as sns
from alphabase.peptide.fragment import get_charged_frag_types

# alpha family imports
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia import fdrexperimental as fdrx

# alphadia imports
from alphadia import fragcomp, plexscoring, utils
from alphadia.peakgroup import search
from alphadia.workflow import base, manager, optimization

logger = logging.getLogger()


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

classifier_base = fdrx.BinaryClassifierLegacyNewBatching(
    test_size=0.001, batch_size=5000, learning_rate=0.001, epochs=10
)


class PeptideCentricWorkflow(base.WorkflowBase):
    def __init__(
        self,
        instance_name: str,
        config: dict,
    ) -> None:
        super().__init__(
            instance_name,
            config,
        )

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
            f"Initializing workflow {self.instance_name}", verbosity="progress"
        )

        self.init_fdr_manager()
        self.init_spectral_library()

    def init_fdr_manager(self):
        self.fdr_manager = manager.FDRManager(
            feature_columns=feature_columns,
            classifier_base=classifier_base,
        )

    def init_spectral_library(self):
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
        self.spectral_library._precursor_df["rt_library"] = self.norm_to_rt(
            self.dia_data, self.spectral_library._precursor_df["rt_library"].values
        )

        # filter based on precursor observability
        lower_mz_limit = self.dia_data.cycle[self.dia_data.cycle > 0].min()
        upper_mz_limit = self.dia_data.cycle[self.dia_data.cycle > 0].max()

        precursor_before = np.sum(self.spectral_library._precursor_df["decoy"] == 0)
        self.spectral_library._precursor_df = self.spectral_library._precursor_df[
            (self.spectral_library._precursor_df["mz_library"] >= lower_mz_limit)
            & (self.spectral_library._precursor_df["mz_library"] <= upper_mz_limit)
        ]
        # self.spectral_library.remove_unused_fragmen
        precursor_after = np.sum(self.spectral_library._precursor_df["decoy"] == 0)
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
        self.spectral_library._precursor_df = (
            self.spectral_library.precursor_df_unfiltered[
                self.spectral_library.precursor_df_unfiltered["channel"].isin(
                    allowed_channels
                )
            ].copy()
        )

    def norm_to_rt(
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

    def get_exponential_batches(self, step):
        """Get the number of batches for a given step
        This plan has the shape:
        1, 2, 4, 8, 16, 32, 64, ...
        """
        return int(2**step)

    def get_batch_plan(self):
        """Gets an exponential batch plan based on the batch_size value in the config."""
        n_eg = self.spectral_library._precursor_df["elution_group_idx"].nunique()

        plan = []

        batch_size = self.config["calibration"]["batch_size"]
        step = 0
        start_index = 0

        while start_index < n_eg:
            n_batches = self.get_exponential_batches(step)
            stop_index = min(start_index + n_batches * batch_size, n_eg)
            plan.append((start_index, stop_index))
            step += 1
            start_index = stop_index

        return plan

    def get_optimization_lock(self):
        """Search parameter optimization (i.e. refinement of tolerances for RT, MS2, etc.) is performed on a subset of the elution groups in the spectral library.
        This subset is termed the optimization lock.
        The number of elution groups which must be searched to get a sufficiently large number for robust calibration varies depending the library used and the data.
        This function searches an increasing number of elution groups until a sufficient number of precursors are identified at 1% FDR and a sufficient number of steps have been taken.
        The values deemed sufficient are specified in by "optimization_lock_target" and "optmization_lock_min_steps" in the config.

        Returns
        -------
        optimization_lock_eg_idxes : np.ndarray
            The indices (in .spectral_library._precursor_df) of the precursors which will be used for calibration.

        precursor_df : pd.DataFrame
            Dataframe of all precursors accumulated during the optimization lock, including q-values from FDR correction.

        fragments_df : pd.DataFrame
            Dataframe of all fragments accumulated during the optimization lock, including q-values from FDR correction.

        """

        self.elution_group_order = self.spectral_library.precursor_df[
            "elution_group_idx"
        ].unique()
        np.random.shuffle(self.elution_group_order)

        batch_plan = self.get_batch_plan()

        features = []
        fragments = []
        for current_step, (start_index, stop_index) in enumerate(batch_plan):
            self.reporter.log_string(
                f"=== Step {current_step}, extracting elution groups {start_index} to {stop_index} ===",
                verbosity="progress",
            )

            eg_idxes = self.elution_group_order[start_index:stop_index]
            batch_df = self.spectral_library._precursor_df[
                self.spectral_library._precursor_df["elution_group_idx"].isin(eg_idxes)
            ]

            feature_df, fragment_df = self.extract_batch(batch_df)
            features += [feature_df]
            fragments += [fragment_df]
            features_df = pd.concat(features)
            fragments_df = pd.concat(fragments)

            self.reporter.log_string(
                f"=== Step {current_step}, extracted {len(feature_df)} precursors and {len(fragment_df)} fragments ===",
                verbosity="progress",
            )

            precursor_df = self.fdr_correction(
                features_df, fragments_df, self.optimization_manager.classifier_version
            )

            self.reporter.log_string(
                f"=== FDR correction performed with classifier version {self.optimization_manager.classifier_version} ===",
                verbosity="info",
            )

            num_precursors_at_01FDR = len(precursor_df[precursor_df["qval"] < 0.01])

            self.reporter.log_string(
                f"=== Checking if minimum number of precursors for optimization found yet; minimum number is {self.config['calibration']['optimization_lock_target']} ===",
                verbosity="progress",
            )

            self.log_precursor_df(precursor_df)

            self.reporter.log_string(
                f"=== Classifier has been trained for {self.fdr_manager.current_version + 1} iteration(s); minimum number is {self.config['calibration']['optimization_lock_min_steps']} ===",
                verbosity="progress",
            )

            if (
                num_precursors_at_01FDR
                > self.config["calibration"]["optimization_lock_target"]
                and current_step
                >= self.config["calibration"]["optimization_lock_min_steps"] - 1
            ):
                final_stop_index = stop_index  # final_stop_index is the number of elution groups that will be included in the calibration data
                break

        optimization_lock_eg_idxes = self.elution_group_order[:final_stop_index]

        optimization_lock_library_precursor_df = self.spectral_library._precursor_df[
            self.spectral_library._precursor_df["elution_group_idx"].isin(
                optimization_lock_eg_idxes
            )
        ]

        optimization_lock_library_precursor_df = (
            self.randomize_optimization_lock_precursor_idx(
                optimization_lock_library_precursor_df
            )
        )

        optimization_lock_library_fragment_df = (
            self.get_optimization_lock_library_fragment(
                optimization_lock_library_precursor_df
            )
        )

        (
            optimization_lock_library_precursor_df,
            optimization_lock_library_fragment_df,
        ) = self.reindex_optimization_lock(
            optimization_lock_library_precursor_df,
            optimization_lock_library_fragment_df,
        )

        return (
            optimization_lock_library_precursor_df,
            optimization_lock_library_fragment_df,
            precursor_df,
            fragments_df,
        )

    def randomize_optimization_lock_precursor_idx(
        self, optimization_lock_library_precursor_df
    ):
        randomized_idx = np.random.permutation(
            optimization_lock_library_precursor_df["precursor_idx"].values
        )
        self.randomized_idx_mapping = dict(
            zip(
                optimization_lock_library_precursor_df["precursor_idx"].values,
                randomized_idx,
                strict=True,
            )
        )
        optimization_lock_library_precursor_df["precursor_idx"] = randomized_idx
        optimization_lock_library_precursor_df.sort_values(
            by="precursor_idx", inplace=True
        )
        optimization_lock_library_precursor_df.reset_index(drop=True, inplace=True)

        return optimization_lock_library_precursor_df

    def get_optimization_lock_library_fragment(
        self, optimization_lock_library_precursor_df
    ):
        start_indices = optimization_lock_library_precursor_df[
            "flat_frag_start_idx"
        ].values
        stop_indices = optimization_lock_library_precursor_df[
            "flat_frag_stop_idx"
        ].values

        # Create ranges for each row and concatenate them
        optimization_lock_fragment_idxes = np.concatenate(
            [
                np.arange(start, stop)
                for start, stop in zip(start_indices, stop_indices, strict=True)
            ]
        )

        # Extract the fragments for the optimization lock and reset the indices to a consecutive range of positive integers. This simplifies future access based on position
        optimization_lock_library_fragment_df = self.spectral_library._fragment_df.iloc[
            optimization_lock_fragment_idxes
        ]

        return optimization_lock_library_fragment_df

    def reindex_optimization_lock(
        self,
        optimization_lock_library_precursor_df,
        optimization_lock_library_fragment_df,
    ):
        optimization_lock_library_fragment_df = (
            optimization_lock_library_fragment_df.reset_index(drop=True)
        )
        # Change the fragment indices in the precursor_df to match the fragment indices in the optimization lock fragment_df instead of the full spectral library.
        num_frags = (
            optimization_lock_library_precursor_df["flat_frag_stop_idx"]
            - optimization_lock_library_precursor_df["flat_frag_start_idx"]
        )
        optimization_lock_library_precursor_df["flat_frag_stop_idx"] = (
            num_frags.cumsum()
        )
        optimization_lock_library_precursor_df["flat_frag_start_idx"] = (
            optimization_lock_library_precursor_df[
                "flat_frag_stop_idx"
            ].shift(fill_value=0)
        )

        return (
            optimization_lock_library_precursor_df,
            optimization_lock_library_fragment_df,
        )

    def get_ordered_optimizers(self):
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
        config_search_initial = self.config["search_initial"]
        config_search = self.config["search"]

        if config_search["target_ms2_tolerance"] > 0:
            ms2_optimizer = optimization.TargetedMS2Optimizer(
                config_search_initial["initial_ms2_tolerance"],
                config_search["target_ms2_tolerance"],
                self,
            )
        else:
            ms2_optimizer = optimization.AutomaticMS2Optimizer(
                config_search_initial["initial_ms2_tolerance"],
                self,
            )

        if config_search["target_rt_tolerance"] > 0:
            rt_optimizer = optimization.TargetedRTOptimizer(
                config_search_initial["initial_rt_tolerance"],
                config_search["target_rt_tolerance"],
                self,
            )
        else:
            rt_optimizer = optimization.AutomaticRTOptimizer(
                config_search_initial["initial_rt_tolerance"],
                self,
            )
        if self.dia_data.has_ms1:
            if config_search["target_ms1_tolerance"] > 0:
                ms1_optimizer = optimization.TargetedMS1Optimizer(
                    config_search_initial["initial_ms1_tolerance"],
                    config_search["target_ms1_tolerance"],
                    self,
                )
            else:
                ms1_optimizer = optimization.AutomaticMS1Optimizer(
                    config_search_initial["initial_ms1_tolerance"],
                    self,
                )
        else:
            ms1_optimizer = None
        if self.dia_data.has_mobility:
            if config_search["target_mobility_tolerance"] > 0:
                mobility_optimizer = optimization.TargetedMobilityOptimizer(
                    config_search_initial["initial_mobility_tolerance"],
                    config_search["target_mobility_tolerance"],
                    self,
                )
            else:
                mobility_optimizer = optimization.AutomaticMobilityOptimizer(
                    config_search_initial["initial_mobility_tolerance"],
                    self.calibration_manager,
                    self.optimization_manager,
                    self.fdr_manager,
                )
        else:
            mobility_optimizer = None

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

        return ordered_optimizers

    def first_recalibration_and_optimization(
        self,
        precursor_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        ordered_optimizers: list,
    ):
        """Performs the first recalibration and optimization step.
        This step is not recorded in the history of the optimizers.

        Parameters
        ----------
        precursor_df : pd.DataFrame
            Precursor dataframe from optimization lock

        fragments_df : pd.DataFrame
            Fragment dataframe from optimization lock

        ordered_optimizers : list
            List of lists of optimizers in correct order
        """
        precursor_df_filtered, fragments_df_filtered = self.filter_dfs(
            precursor_df, fragments_df
        )

        self.recalibration(precursor_df_filtered, fragments_df_filtered)

        self.reporter.log_string(
            "=== Performing initial optimization on extracted data. ===",
            verbosity="info",
        )

        precursor_df_filtered["precursor_idx"] = precursor_df_filtered[
            "precursor_idx"
        ].map(self.randomized_idx_mapping)
        target = (
            2 * self.config["calibration"]["optimization_lock_target"]
        )  # The target number of precursors for optimization is taken to be more than the number needed to start optimization. This allows a margin of safety.
        # At the moment it is hard-coded to twice the number of precursors needed for the optimization lock but this may be made configurable in the future.
        if (
            len(precursor_df_filtered) > target
        ):  # Takes the target number of precursors if at least that many are found
            self.precursor_cutoff_idx = (
                precursor_df_filtered.sort_values(by="precursor_idx")
                .iloc[target]
                .precursor_idx
            )
        else:  # Otherwise takes the number of precursors used at the start of the optimizer's usage
            self.precursor_cutoff_idx = (
                precursor_df_filtered.sort_values(by="precursor_idx")
                .iloc[-1]
                .precursor_idx
                + 1
            )

        return precursor_df_filtered, fragments_df_filtered

    def set_reduced_optimization_lock(self):
        self.reduced_optimization_lock_library_precursor_df = (
            self.optimization_lock_library_precursor_df[
                self.optimization_lock_library_precursor_df["precursor_idx"]
                < self.precursor_cutoff_idx
            ]
        )

        fragment_cutoff_idx = self.reduced_optimization_lock_library_precursor_df[
            "flat_frag_stop_idx"
        ].iloc[
            -1
        ]  # The last fragment index of the last precursor in the reduced optimization lock.
        # Note that this requires the precursor indices to be sorted prior to the reassignment of fragment indices when creating the optimization lock
        # (as otherwise there is nothing special about this fragment index to guarantee that all fragments for previous precursors will be included).

        self.reduced_optimization_lock_library_fragment_df = (
            self.optimization_lock_library_fragment_df[
                self.optimization_lock_library_fragment_df.index < fragment_cutoff_idx
            ]
        )

        self.reporter.log_string(
            f"Optimization lock library size reduced from {len(self.optimization_lock_library_precursor_df)} to {len(self.reduced_optimization_lock_library_precursor_df)} precursors and {len(self.optimization_lock_library_fragment_df)} to {len(self.reduced_optimization_lock_library_fragment_df)} fragments.",
            verbosity="info",
        )

        self.reporter.log_string(
            f"Reduced optimization lock contains {self.reduced_optimization_lock_library_precursor_df.decoy.mean() * 100:.2f}% target precursors.",
            verbosity="info",
        )

    def calibration(self):
        """Performs optimization of the search parameters. This occurs in two stages:
        1) Optimization lock: the data are searched to acquire a locked set of precursors which is used for search parameter optimization. The classifier is also trained during this stage.
        2) Optimization loop: the search parameters are optimized iteratively using the locked set of precursors.
            In each iteration, the data are searched with the locked library from stage 1, and the properties -- m/z for both precursors and fragments (i.e. MS1 and MS2), RT and mobility -- are recalibrated.
            The optimization loop is repeated for each list of optimizers in ordered_optimizers.

        """
        # First check to see if the calibration has already been performed. Return if so.
        if (
            self.calibration_manager.is_fitted
            and self.calibration_manager.is_loaded_from_file
        ):
            self.reporter.log_string(
                "Skipping calibration as existing calibration was found",
                verbosity="progress",
            )
            return
        self.timing_manager.start("optimization")
        # Get the order of optimization
        ordered_optimizers = self.get_ordered_optimizers()

        self.reporter.log_string(
            "Starting initial classifier training and precursor identification.",
            verbosity="progress",
        )

        # Get the optimization lock
        (
            self.optimization_lock_library_precursor_df,
            self.optimization_lock_library_fragment_df,
            precursor_df,
            fragments_df,
        ) = self.get_optimization_lock()

        self.optimization_manager.fit(
            {"classifier_version": self.fdr_manager.current_version}
        )

        self.reporter.log_string(
            "Required number of precursors found and required number of training iterations performed. Starting search parameter optimization.",
            verbosity="progress",
        )

        # Perform a first recalibration on the optimization lock.
        precursor_df_filtered, fragments_df_filtered = (
            self.first_recalibration_and_optimization(
                precursor_df, fragments_df, ordered_optimizers
            )
        )

        # Start of optimization/recalibration loop
        for optimizers in ordered_optimizers:
            self.set_reduced_optimization_lock()
            for current_step in range(self.config["calibration"]["max_steps"]):
                if np.all([optimizer.has_converged for optimizer in optimizers]):
                    self.reporter.log_string(
                        f"Optimization finished for {', '.join([optimizer.parameter_name for optimizer in optimizers])}.",
                        verbosity="progress",
                    )

                    for optimizer in optimizers:
                        optimizer.plot()

                    break

                features_df, fragments_df = self.extract_batch(
                    self.reduced_optimization_lock_library_precursor_df,
                    self.reduced_optimization_lock_library_fragment_df,
                )

                self.reporter.log_string(
                    f"=== Step {current_step}, extracted {len(features_df)} precursors and {len(fragments_df)} fragments ===",
                    verbosity="progress",
                )

                precursor_df = self.fdr_correction(
                    features_df,
                    fragments_df,
                    self.optimization_manager.classifier_version,
                )

                self.reporter.log_string(
                    f"=== FDR correction performed with classifier version {self.optimization_manager.classifier_version} ===",
                    verbosity="info",
                )

                self.log_precursor_df(precursor_df)

                precursor_df_filtered, fragments_df_filtered = self.filter_dfs(
                    precursor_df, fragments_df
                )

                self.recalibration(precursor_df_filtered, fragments_df_filtered)

                self.reporter.log_string(
                    "=== checking if optimization conditions were reached ===",
                    verbosity="info",
                )

                for optimizer in optimizers:
                    optimizer.step(
                        precursor_df_filtered, fragments_df_filtered, current_step
                    )

                self.reporter.log_string(
                    "==============================================", verbosity="info"
                )

                self.reporter.log_string(
                    f"=== Optimization has been performed for {current_step + 1} step(s); minimum number is {self.config['calibration']['min_steps']} ===",
                    verbosity="progress",
                )

            else:
                self.reporter.log_string(
                    "Optimization did not converge within the maximum number of steps, which is {self.config['calibration']['max_steps']}.",
                    verbosity="warning",
                )

        self.reporter.log_string(
            "Search parameter optimization finished. Values taken forward for search are:",
            verbosity="progress",
        )
        self.reporter.log_string(
            "==============================================", verbosity="progress"
        )
        for optimizers in ordered_optimizers:
            for optimizer in optimizers:
                self.reporter.log_string(
                    f"{optimizer.parameter_name:<15}: {self.optimization_manager.__dict__[optimizer.parameter_name]:.4f}",
                    verbosity="progress",
                )
        self.reporter.log_string(
            "==============================================", verbosity="progress"
        )

        self.timing_manager.end("optimization")

    def filter_dfs(self, precursor_df, fragments_df):
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

        """
        precursor_df_filtered = precursor_df[precursor_df["qval"] < 0.01]
        precursor_df_filtered = precursor_df_filtered[
            precursor_df_filtered["decoy"] == 0
        ]

        fragments_df_filtered = fragments_df[
            fragments_df["precursor_idx"].isin(precursor_df_filtered["precursor_idx"])
        ]

        min_fragments = 500
        max_fragments = 5000
        min_correlation = 0.7
        fragments_df_filtered = fragments_df_filtered.sort_values(
            by=["correlation"], ascending=False
        )
        stop_rank = min(
            max(
                np.searchsorted(
                    fragments_df_filtered["correlation"].values, min_correlation
                ),
                min_fragments,
            ),
            max_fragments,
        )
        fragments_df_filtered = fragments_df_filtered.iloc[:stop_rank]

        self.reporter.log_string(
            f"fragments_df_filtered: {len(fragments_df_filtered)}", verbosity="info"
        )

        return precursor_df_filtered, fragments_df_filtered

    def recalibration(self, precursor_df_filtered, fragments_df_filtered):
        """Performs recalibration of the the MS1, MS2, RT and mobility properties. Also fits the convolution kernel and the score cutoff.
        The calibration manager is used to fit the data and predict the calibrated values.

        Parameters
        ----------
        precursor_df_filtered : pd.DataFrame
            Filtered precursor dataframe (see filter_dfs)

        fragments_df_filtered : pd.DataFrame
            Filtered fragment dataframe (see filter_dfs)

        """
        self.calibration_manager.fit(
            precursor_df_filtered,
            "precursor",
            plot=True,
            skip=["mz"] if not self.dia_data.has_ms1 else [],
            # neptune_run = self.neptune
        )

        self.calibration_manager.fit(
            fragments_df_filtered,
            "fragment",
            plot=True,
            # neptune_run = self.neptune
        )

        self.calibration_manager.predict(
            self.optimization_lock_library_precursor_df,
            "precursor",
        )

        self.calibration_manager.predict(
            self.optimization_lock_library_fragment_df, "fragment"
        )

        self.optimization_manager.fit(
            {
                "column_type": "calibrated",
                "num_candidates": self.config["search"]["target_num_candidates"],
            }
        )

        percentile_001 = np.percentile(precursor_df_filtered["score"], 0.1)
        self.optimization_manager.fit(
            {
                "fwhm_rt": precursor_df_filtered["cycle_fwhm"].median(),
                "fwhm_mobility": precursor_df_filtered["mobility_fwhm"].median(),
                "score_cutoff": percentile_001,
            }
        )

    def fdr_correction(self, features_df, df_fragments, version=-1):
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
            # neptune_run=self.neptune
        )

    def extract_batch(self, batch_df, batch_fragment_df=None, apply_cutoff=False):
        if batch_fragment_df is None:
            batch_fragment_df = self.spectral_library._fragment_df
        self.reporter.log_string(
            f"Extracting batch of {len(batch_df)} precursors", verbosity="progress"
        )

        config = search.HybridCandidateConfig()
        config.update(self.config["selection_config"])
        config.update(
            {
                "top_k_fragments": self.config["search_advanced"]["top_k_fragments"],
                "rt_tolerance": self.optimization_manager.rt_error,
                "mobility_tolerance": self.optimization_manager.mobility_error,
                "candidate_count": self.optimization_manager.num_candidates,
                "precursor_mz_tolerance": self.optimization_manager.ms1_error,
                "fragment_mz_tolerance": self.optimization_manager.ms2_error,
                "exclude_shared_ions": self.config["search"]["exclude_shared_ions"],
                "min_size_rt": self.config["search"]["quant_window"],
            }
        )

        extraction = search.HybridCandidateSelection(
            self.dia_data.jitclass(),
            batch_df,
            batch_fragment_df,
            config.jitclass(),
            rt_column=f"rt_{self.optimization_manager.column_type}",
            mobility_column=f"mobility_{self.optimization_manager.column_type}"
            if self.dia_data.has_mobility
            else "mobility_library",
            precursor_mz_column=f"mz_{self.optimization_manager.column_type}"
            if self.dia_data.has_ms1
            else "mz_library",
            fragment_mz_column=f"mz_{self.optimization_manager.column_type}",
            fwhm_rt=self.optimization_manager.fwhm_rt,
            fwhm_mobility=self.optimization_manager.fwhm_mobility,
        )
        candidates_df = extraction(thread_count=self.config["general"]["thread_count"])

        sns.histplot(candidates_df, x="score", hue="decoy", bins=100)

        if apply_cutoff:
            num_before = len(candidates_df)
            self.reporter.log_string(
                f"Applying score cutoff of {self.optimization_manager.score_cutoff}",
                verbosity="info",
            )
            candidates_df = candidates_df[
                candidates_df["score"] > self.optimization_manager.score_cutoff
            ]
            num_after = len(candidates_df)
            num_removed = num_before - num_after
            self.reporter.log_string(
                f"Removed {num_removed} precursors with score below cutoff",
                verbosity="info",
            )

        config = plexscoring.CandidateConfig()
        config.update(self.config["scoring_config"])
        config.update(
            {
                "top_k_fragments": self.config["search_advanced"]["top_k_fragments"],
                "precursor_mz_tolerance": self.optimization_manager.ms1_error,
                "fragment_mz_tolerance": self.optimization_manager.ms2_error,
                "exclude_shared_ions": self.config["search"]["exclude_shared_ions"],
                "quant_window": self.config["search"]["quant_window"],
                "quant_all": self.config["search"]["quant_all"],
            }
        )

        candidate_scoring = plexscoring.CandidateScoring(
            self.dia_data.jitclass(),
            batch_df,
            batch_fragment_df,
            config=config,
            rt_column=f"rt_{self.optimization_manager.column_type}",
            mobility_column=f"mobility_{self.optimization_manager.column_type}"
            if self.dia_data.has_mobility
            else "mobility_library",
            precursor_mz_column=f"mz_{self.optimization_manager.column_type}"
            if self.dia_data.has_ms1
            else "mz_library",
            fragment_mz_column=f"mz_{self.optimization_manager.column_type}",
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self.config["general"]["thread_count"],
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df

    def extraction(self):
        self.timing_manager.start("extraction")
        self.optimization_manager.fit(
            {
                "num_candidates": self.config["search"]["target_num_candidates"],
                "column_type": "calibrated",
            }
        )

        self.calibration_manager.predict(
            self.spectral_library._precursor_df, "precursor"
        )
        self.calibration_manager.predict(self.spectral_library._fragment_df, "fragment")

        features_df, fragments_df = self.extract_batch(
            self.spectral_library._precursor_df,
            apply_cutoff=True,
        )

        self.reporter.log_string(
            f"=== FDR correction performed with classifier version {self.optimization_manager.classifier_version} ===",
            verbosity="info",
        )

        precursor_df = self.fdr_correction(
            features_df, fragments_df, self.optimization_manager.classifier_version
        )

        precursor_df = precursor_df[precursor_df["qval"] <= self.config["fdr"]["fdr"]]

        logger.info("Removing fragments below FDR threshold")

        # to be optimized later
        fragments_df["candidate_idx"] = utils.candidate_hash(
            fragments_df["precursor_idx"].values, fragments_df["rank"].values
        )
        precursor_df["candidate_idx"] = utils.candidate_hash(
            precursor_df["precursor_idx"].values, precursor_df["rank"].values
        )

        fragments_df = fragments_df[
            fragments_df["candidate_idx"].isin(precursor_df["candidate_idx"])
        ]

        self.log_precursor_df(precursor_df)

        self.timing_manager.end("extraction")

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

        precursor_01fdr = len(
            precursor_df[(precursor_df["qval"] < 0.01) & (precursor_df["decoy"] == 0)]
        )
        proteins_01fdr = precursor_df[
            (precursor_df["qval"] < 0.01) & (precursor_df["decoy"] == 0)
        ]["proteins"].nunique()

        # if self.neptune is not None:
        #    self.neptune['precursors'].log(precursor_01fdr)
        #    self.neptune['proteins'].log(proteins_01fdr)

    def requantify(self, psm_df):
        self.calibration_manager.predict(
            self.spectral_library.precursor_df_unfiltered, "precursor"
        )
        self.calibration_manager.predict(self.spectral_library._fragment_df, "fragment")

        reference_candidates = plexscoring.candidate_features_to_candidates(psm_df)

        if "multiplexing" not in self.config:
            raise ValueError("no multiplexing config found")
        self.reporter.log_string(
            f"=== Multiplexing {len(reference_candidates):,} precursors ===",
            verbosity="progress",
        )

        original_channels = psm_df["channel"].unique().tolist()
        self.reporter.log_string(
            f"original channels: {original_channels}", verbosity="progress"
        )

        reference_channel = self.config["multiplexing"]["reference_channel"]
        self.reporter.log_string(
            f"reference channel: {reference_channel}", verbosity="progress"
        )

        target_channels = [
            int(c) for c in self.config["multiplexing"]["target_channels"].split(",")
        ]
        self.reporter.log_string(
            f"target channels: {target_channels}", verbosity="progress"
        )

        decoy_channel = self.config["multiplexing"]["decoy_channel"]
        self.reporter.log_string(
            f"decoy channel: {decoy_channel}", verbosity="progress"
        )

        channels = list(
            set(
                original_channels
                + [reference_channel]
                + target_channels
                + [decoy_channel]
            )
        )
        multiplexed_candidates = plexscoring.multiplex_candidates(
            reference_candidates,
            self.spectral_library.precursor_df_unfiltered,
            channels=channels,
        )

        channel_count_lib = self.spectral_library.precursor_df_unfiltered[
            "channel"
        ].value_counts()
        channel_count_multiplexed = multiplexed_candidates["channel"].value_counts()
        ## log channels with less than 100 precursors
        for channel in channels:
            if channel not in channel_count_lib:
                self.reporter.log_string(
                    f"channel {channel} not found in library", verbosity="warning"
                )
            if channel not in channel_count_multiplexed:
                self.reporter.log_string(
                    f"channel {channel} could not be mapped to existing IDs.",
                    verbosity="warning",
                )

        self.reporter.log_string(
            f"=== Requantifying {len(multiplexed_candidates):,} precursors ===",
            verbosity="progress",
        )

        config = plexscoring.CandidateConfig()
        config.score_grouped = True
        config.exclude_shared_ions = True
        config.reference_channel = self.config["multiplexing"]["reference_channel"]

        multiplexed_scoring = plexscoring.CandidateScoring(
            self.dia_data.jitclass(),
            self.spectral_library.precursor_df_unfiltered,
            self.spectral_library.fragment_df,
            config=config,
            precursor_mz_column="mz_calibrated",
            fragment_mz_column="mz_calibrated",
            rt_column="rt_calibrated",
            mobility_column="mobility_calibrated"
            if self.dia_data.has_mobility
            else "mobility_library",
        )

        multiplexed_candidates["rank"] = 0

        multiplexed_features, fragments = multiplexed_scoring(multiplexed_candidates)

        target_channels = [
            int(c) for c in self.config["multiplexing"]["target_channels"].split(",")
        ]
        print("target_channels", target_channels)
        reference_channel = self.config["multiplexing"]["reference_channel"]

        psm_df = self.fdr_manager.fit_predict(
            multiplexed_features,
            decoy_strategy="channel",
            competetive=self.config["multiplexing"]["competetive_scoring"],
            decoy_channel=decoy_channel,
        )

        self.log_precursor_df(psm_df)

        return psm_df

    def requantify_fragments(
        self, psm_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Requantify confident precursor identifications for transfer learning.

        Parameters
        ----------

        psm_df: pd.DataFrame
            Dataframe with peptide identifications

        Returns
        -------

        psm_df: pd.DataFrame
            Dataframe with existing peptide identifications but updated frag_start_idx and frag_stop_idx

        frag_df: pd.DataFrame
            Dataframe with fragments in long format
        """

        self.reporter.log_string(
            "=== Transfer learning quantification ===",
            verbosity="progress",
        )

        fragment_types = self.config["transfer_library"]["fragment_types"].split(";")
        max_charge = self.config["transfer_library"]["max_charge"]

        self.reporter.log_string(
            f"creating library for charged fragment types: {fragment_types}",
            verbosity="info",
        )

        candidate_speclib_flat, scored_candidates = _build_candidate_speclib_flat(
            psm_df, fragment_types=fragment_types, max_charge=max_charge
        )

        self.reporter.log_string(
            "Calibrating library",
            verbosity="info",
        )

        # calibrate
        self.calibration_manager.predict(
            candidate_speclib_flat.precursor_df, "precursor"
        )
        self.calibration_manager.predict(candidate_speclib_flat.fragment_df, "fragment")

        self.reporter.log_string(
            f"quantifying {len(scored_candidates):,} precursors with {len(candidate_speclib_flat.fragment_df):,} fragments",
            verbosity="info",
        )

        config = plexscoring.CandidateConfig()
        config.update(
            {
                "top_k_fragments": 1000,  # Use all fragments ever expected, needs to be larger than charged_frag_types(8)*max_sequence_len(100?)
                "precursor_mz_tolerance": self.config["search"]["target_ms1_tolerance"],
                "fragment_mz_tolerance": self.config["search"]["target_ms2_tolerance"],
            }
        )

        scoring = plexscoring.CandidateScoring(
            self.dia_data.jitclass(),
            candidate_speclib_flat.precursor_df,
            candidate_speclib_flat.fragment_df,
            config=config,
            precursor_mz_column="mz_calibrated",
            fragment_mz_column="mz_calibrated",
        )

        # we disregard the precursors, as we want to keep the original scoring from the top12 search
        # this works fine as there is no index pointing from the precursors to the fragments
        # only the fragments arre indexed by precursor_idx and rank
        _, frag_df = scoring(scored_candidates)

        # establish mapping
        # TODO: we are reusing the FragmentCompetition class here which should be refactored
        frag_comp = fragcomp.FragmentCompetition()
        scored_candidates["_candidate_idx"] = utils.candidate_hash(
            scored_candidates["precursor_idx"].values, scored_candidates["rank"].values
        )
        frag_df["_candidate_idx"] = utils.candidate_hash(
            frag_df["precursor_idx"].values, frag_df["rank"].values
        )
        scored_candidates = frag_comp.add_frag_start_stop_idx(
            scored_candidates, frag_df
        )

        return scored_candidates, frag_df


def _build_candidate_speclib_flat(
    psm_df: pd.DataFrame,
    fragment_types: list[str] | None = None,
    max_charge: int = 2,
    optional_columns: list[str] | None = None,
) -> tuple[SpecLibFlat, pd.DataFrame]:
    """Build a candidate spectral library for transfer learning.

    Parameters
    ----------

    psm_df: pd.DataFrame
        Dataframe with peptide identifications

    fragment_types: typing.List[str], optional
        List of fragment types to include in the library, by default ['b','y']

    max_charge: int, optional
        Maximum fragment charge state to consider, by default 2

    optional_columns: typing.List[str], optional
        List of optional columns to include in the library, by default [
            "proba",
            "score",
            "qval",
            "channel",
            "rt_library",
            "mz_library",
            "mobility_library",
            "genes",
            "proteins",
            "decoy",
            "mods",
            "mod_sites",
            "sequence",
            "charge",
            "rt_observed", "mobility_observed", "mz_observed"
        ]

    Returns
    -------
    candidate_speclib_flat: SpecLibFlat
        Candidate spectral library in flat format

    scored_candidates: pd.DataFrame
        Dataframe with scored candidates
    """

    # set default optional columns
    if fragment_types is None:
        fragment_types = ["b", "y"]
    if optional_columns is None:
        optional_columns = [
            "proba",
            "score",
            "qval",
            "channel",
            "rt_library",
            "mz_library",
            "mobility_library",
            "genes",
            "proteins",
            "decoy",
            "mods",
            "mod_sites",
            "sequence",
            "charge",
            "rt_observed",
            "mobility_observed",
            "mz_observed",
        ]

    scored_candidates = plexscoring.candidate_features_to_candidates(
        psm_df, optional_columns=optional_columns
    )

    # create speclib with fragment_types of interest
    candidate_speclib = SpecLibBase()
    candidate_speclib.precursor_df = scored_candidates

    candidate_speclib.charged_frag_types = get_charged_frag_types(
        fragment_types, max_charge
    )

    candidate_speclib.calc_fragment_mz_df()

    candidate_speclib._fragment_intensity_df = candidate_speclib.fragment_mz_df.copy()
    # set all fragment weights to 1 to make sure all are quantified
    candidate_speclib._fragment_intensity_df[candidate_speclib.charged_frag_types] = 1.0

    # create flat speclib
    candidate_speclib_flat = SpecLibFlat()
    candidate_speclib_flat.parse_base_library(candidate_speclib)
    # delete immediately to free memory
    del candidate_speclib

    candidate_speclib_flat.fragment_df.rename(
        columns={"mz": "mz_library"}, inplace=True
    )
    candidate_speclib_flat.fragment_df["cardinality"] = 0
    return candidate_speclib_flat, scored_candidates

# native imports
import os
import logging

logger = logging.getLogger()
import typing

# alphadia imports
from alphadia import plexscoring, hybridselection
from alphadia import fdrexperimental as fdrx
from alphadia.workflow import manager, base

# alpha family imports
from alphabase.spectral_library.base import SpecLibBase

# third party imports
import numpy as np
import pandas as pd

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
]

classifier_base = fdrx.BinaryClassifierLegacyNewBatching(
    test_size=0.001,
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

        self.init_calibration_optimization_manager()
        self.init_fdr_manager()
        self.init_spectral_library()

    @property
    def calibration_optimization_manager(self):
        """Is used during the iterative optimization of the calibration parameters.
        Should not be stored on disk.
        """
        return self._calibration_optimization_manager

    @property
    def com(self):
        """alias for calibration_optimization_manager"""
        return self.calibration_optimization_manager

    def init_calibration_optimization_manager(self):
        self._calibration_optimization_manager = manager.OptimizationManager(
            {
                "current_epoch": 0,
                "current_step": 0,
                "ms1_error": self.config["search_initial"]["initial_ms1_tolerance"],
                "ms2_error": self.config["search_initial"]["initial_ms2_tolerance"],
                "rt_error": self.config["search_initial"]["initial_rt_tolerance"],
                "mobility_error": self.config["search_initial"][
                    "initial_mobility_tolerance"
                ],
                "column_type": "library",
                "num_candidates": self.config["search_initial"][
                    "initial_num_candidates"
                ],
                "recalibration_target": self.config["calibration"][
                    "recalibration_target"
                ],
                "accumulated_precursors": 0,
                "accumulated_precursors_01FDR": 0,
                "accumulated_precursors_001FDR": 0,
            }
        )

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
        active_gradient_start: typing.Union[float, None] = None,
        active_gradient_stop: typing.Union[float, None] = None,
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
            mode = (
                self.config["calibration"]["norm_rt_mode"]
                if "norm_rt_mode" in self.config["calibration"]
                else "tic"
            )
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

    def start_of_calibration(self):
        self.batch_plan = self.get_batch_plan()

    def start_of_epoch(self, current_epoch):
        self.com.current_epoch = current_epoch

        # if self.neptune is not None:
        #    self.neptune["eval/epoch"].log(current_epoch)

        self.elution_group_order = (
            self.spectral_library.precursor_df["elution_group_idx"]
            .sample(frac=1)
            .values
        )

        self.calibration_manager.predict(
            self.spectral_library._precursor_df, "precursor"
        )
        self.calibration_manager.predict(self.spectral_library._fragment_df, "fragment")

        # make updates to the progress dict depending on the epoch
        if self.com.current_epoch > 0:
            self.com.recalibration_target = self.config["calibration"][
                "recalibration_target"
            ] * (1 + current_epoch)

    def start_of_step(self, current_step, start_index, stop_index):
        self.com.current_step = current_step
        # if self.neptune is not None:
        #    self.neptune["eval/step"].log(current_step)

        #    for key, value in self.com.__dict__.items():
        #        self.neptune[f"eval/{key}"].log(value)

        self.reporter.log_string(
            f"=== Epoch {self.com.current_epoch}, step {current_step}, extracting elution groups {start_index} to {stop_index} ===",
            verbosity="progress",
        )

    def check_epoch_conditions(self):
        continue_calibration = False

        if self.com.ms1_error > self.config["search"]["target_ms1_tolerance"]:
            continue_calibration = True

        if self.com.ms2_error > self.config["search"]["target_ms2_tolerance"]:
            continue_calibration = True

        if self.com.rt_error > self.config["search"]["target_rt_tolerance"]:
            continue_calibration = True

        if self.dia_data.has_mobility:
            if (
                self.com.mobility_error
                > self.config["search"]["target_mobility_tolerance"]
            ):
                continue_calibration = True

        if self.com.current_epoch < self.config["calibration"]["min_epochs"]:
            continue_calibration = True

        return continue_calibration

    def calibration(self):
        if (
            self.calibration_manager.is_fitted
            and self.calibration_manager.is_loaded_from_file
        ):
            self.reporter.log_string(
                "Skipping calibration as existing calibration was found",
                verbosity="progress",
            )
            return

        self.start_of_calibration()
        for current_epoch in range(self.config["calibration"]["max_epochs"]):
            if self.check_epoch_conditions():
                pass
            else:
                break

            self.start_of_epoch(current_epoch)

            features = []
            fragments = []
            for current_step, (start_index, stop_index) in enumerate(self.batch_plan):
                self.start_of_step(current_step, start_index, stop_index)

                eg_idxes = self.elution_group_order[start_index:stop_index]
                batch_df = self.spectral_library._precursor_df[
                    self.spectral_library._precursor_df["elution_group_idx"].isin(
                        eg_idxes
                    )
                ]

                feature_df, fragment_df = self.extract_batch(batch_df)
                features += [feature_df]
                fragments += [fragment_df]
                features_df = pd.concat(features)
                fragments_df = pd.concat(fragments)

                self.reporter.log_string(
                    f"=== Epoch {self.com.current_epoch}, step {current_step}, extracted {len(feature_df)} precursors and {len(fragment_df)} fragments ===",
                    verbosity="progress",
                )
                precursor_df = self.fdr_correction(features_df)
                # precursor_df = self.fdr_correction(precursor_df)

                if self.check_recalibration(precursor_df):
                    self.recalibration(precursor_df, fragments_df)
                    break
                else:
                    # check if last step has been reached
                    if current_step == len(self.batch_plan) - 1:
                        self.reporter.log_string(
                            "Searched all data without finding recalibration target",
                            verbosity="error",
                        )
                        raise RuntimeError(
                            "Searched all data without finding recalibration target"
                        )

            self.end_of_epoch()

        if "final_full_calibration" in self.config["calibration"]:
            if self.config["calibration"]["final_full_calibration"]:
                self.reporter.log_string(
                    "Performing final calibration with all precursors",
                    verbosity="progress",
                )
                features_df, fragments_df = self.extract_batch(
                    self.spectral_library._precursor_df
                )
                precursor_df = self.fdr_correction(features_df)
                self.recalibration(precursor_df, fragments_df)

        self.end_of_calibration()

    def end_of_epoch(self):
        pass

    def end_of_calibration(self):
        # self.calibration_manager.predict(self.spectral_library._precursor_df, 'precursor')
        # self.calibration_manager.predict(self.spectral_library._fragment_df, 'fragment')
        self.calibration_manager.save()
        pass

    def recalibration(self, precursor_df, fragments_df):
        precursor_df_filtered = precursor_df[precursor_df["qval"] < 0.01]
        precursor_df_filtered = precursor_df_filtered[
            precursor_df_filtered["decoy"] == 0
        ]

        self.calibration_manager.fit(
            precursor_df_filtered,
            "precursor",
            plot=True,
            # neptune_run = self.neptune
        )

        m1_70 = self.calibration_manager.get_estimator("precursor", "mz").ci(
            precursor_df_filtered, 0.70
        )
        m1_99 = self.calibration_manager.get_estimator("precursor", "mz").ci(
            precursor_df_filtered, 0.95
        )
        rt_70 = self.calibration_manager.get_estimator("precursor", "rt").ci(
            precursor_df_filtered, 0.70
        )
        rt_99 = self.calibration_manager.get_estimator("precursor", "rt").ci(
            precursor_df_filtered, 0.95
        )

        # top_intensity_precursors = precursor_df_filtered.sort_values(by=['intensity'], ascending=False)
        median_precursor_intensity = precursor_df_filtered[
            "weighted_ms1_intensity"
        ].median()
        top_intensity_precursors = precursor_df_filtered[
            precursor_df_filtered["weighted_ms1_intensity"] > median_precursor_intensity
        ]
        fragments_df_filtered = fragments_df[
            fragments_df["precursor_idx"].isin(
                top_intensity_precursors["precursor_idx"]
            )
        ]
        median_fragment_intensity = fragments_df_filtered["intensity"].median()
        fragments_df_filtered = fragments_df_filtered[
            fragments_df_filtered["intensity"] > median_fragment_intensity
        ].head(50000)

        self.calibration_manager.fit(
            fragments_df_filtered,
            "fragment",
            plot=True,
            # neptune_run = self.neptune
        )

        m2_70 = self.calibration_manager.get_estimator("fragment", "mz").ci(
            fragments_df_filtered, 0.70
        )
        m2_99 = self.calibration_manager.get_estimator("fragment", "mz").ci(
            fragments_df_filtered, 0.95
        )

        self.com.fit(
            {
                "ms1_error": max(m1_99, self.config["search"]["target_ms1_tolerance"]),
                "ms2_error": max(m2_99, self.config["search"]["target_ms2_tolerance"]),
                "rt_error": max(rt_99, self.config["search"]["target_rt_tolerance"]),
                "column_type": "calibrated",
                "num_candidates": self.config["search"]["target_num_candidates"],
            }
        )

        if self.dia_data.has_mobility:
            mobility_99 = self.calibration_manager.get_estimator(
                "precursor", "mobility"
            ).ci(precursor_df_filtered, 0.95)
            self.com.fit(
                {
                    "mobility_error": max(
                        mobility_99, self.config["search"]["target_mobility_tolerance"]
                    ),
                }
            )

            # if self.neptune is not None:
            #    self.neptune['eval/99_mobility_error'].log(mobility_99)

        self.optimization_manager.fit(
            {
                "fwhm_rt": precursor_df_filtered["cycle_fwhm"].median(),
                "fwhm_mobility": precursor_df_filtered["mobility_fwhm"].median(),
            }
        )

        # if self.neptune is not None:
        # precursor_df_fdr = precursor_df_filtered[precursor_df_filtered['qval'] < 0.01]
        # self.neptune["eval/precursors"].log(len(precursor_df_fdr))
        # self.neptune['eval/99_ms1_error'].log(m1_99)
        # self.neptune['eval/99_ms2_error'].log(m2_99)
        # self.neptune['eval/99_rt_error'].log(rt_99)

    def check_recalibration(self, precursor_df):
        self.com.accumulated_precursors = len(precursor_df)
        self.com.accumulated_precursors_01FDR = len(
            precursor_df[precursor_df["qval"] < 0.01]
        )

        self.reporter.log_string(
            f"=== checking if recalibration conditions were reached, target {self.com.recalibration_target} precursors ===",
            verbosity="progress",
        )

        self.log_precursor_df(precursor_df)

        perform_recalibration = False

        if self.com.accumulated_precursors_01FDR > self.com.recalibration_target:
            perform_recalibration = True

        return perform_recalibration

    def fdr_correction(self, features_df):
        return self.fdr_manager.fit_predict(
            features_df,
            decoy_strategy="precursor_channel_wise"
            if self.config["fdr"]["channel_wise_fdr"]
            else "precursor",
            competetive=self.config["fdr"]["competetive_scoring"],
            # neptune_run=self.neptune
        )

    def extract_batch(self, batch_df):
        self.reporter.log_string(
            f"Extracting batch of {len(batch_df)} precursors", verbosity="progress"
        )

        config = hybridselection.HybridCandidateConfig()
        config.update(self.config["selection_config"])
        config.update(
            {
                "top_k_fragments": self.config["search_advanced"]["top_k_fragments"],
                "rt_tolerance": self.com.rt_error,
                "mobility_tolerance": self.com.mobility_error,
                "candidate_count": self.com.num_candidates,
                "precursor_mz_tolerance": self.com.ms1_error,
                "fragment_mz_tolerance": self.com.ms2_error,
                "exclude_shared_ions": self.config["search"]["exclude_shared_ions"],
            }
        )

        extraction = hybridselection.HybridCandidateSelection(
            self.dia_data.jitclass(),
            batch_df,
            self.spectral_library.fragment_df,
            config.jitclass(),
            rt_column=f"rt_{self.com.column_type}",
            mobility_column=f"mobility_{self.com.column_type}"
            if self.dia_data.has_mobility
            else "mobility_library",
            precursor_mz_column=f"mz_{self.com.column_type}",
            fragment_mz_column=f"mz_{self.com.column_type}",
            fwhm_rt=self.optimization_manager.fwhm_rt,
            fwhm_mobility=self.optimization_manager.fwhm_mobility,
        )
        candidates_df = extraction(thread_count=self.config["general"]["thread_count"])

        config = plexscoring.CandidateConfig()
        config.update(self.config["scoring_config"])
        config.update(
            {
                "top_k_fragments": self.config["search_advanced"]["top_k_fragments"],
                "precursor_mz_tolerance": self.com.ms1_error,
                "fragment_mz_tolerance": self.com.ms2_error,
                "exclude_shared_ions": self.config["search"]["exclude_shared_ions"],
            }
        )

        candidate_scoring = plexscoring.CandidateScoring(
            self.dia_data.jitclass(),
            self.spectral_library._precursor_df,
            self.spectral_library._fragment_df,
            config=config,
            rt_column=f"rt_{self.com.column_type}",
            mobility_column=f"mobility_{self.com.column_type}"
            if self.dia_data.has_mobility
            else "mobility_library",
            precursor_mz_column=f"mz_{self.com.column_type}",
            fragment_mz_column=f"mz_{self.com.column_type}",
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df, thread_count=self.config["general"]["thread_count"]
        )

        return features_df, fragments_df

    def extraction(self):
        # if self.neptune is not None:
        #    for key, value in self.com.__dict__.items():
        #        if key is not None:
        #            self.neptune[f"eval/{key}"].log(value)

        self.com.fit(
            {
                "num_candidates": self.config["search"]["target_num_candidates"],
                "ms1_error": self.config["search"]["target_ms1_tolerance"],
                "ms2_error": self.config["search"]["target_ms2_tolerance"],
                "rt_error": self.config["search"]["target_rt_tolerance"],
                "mobility_error": self.config["search"]["target_mobility_tolerance"],
                "column_type": "calibrated",
            }
        )

        self.calibration_manager.predict(
            self.spectral_library._precursor_df, "precursor"
        )
        self.calibration_manager.predict(self.spectral_library._fragment_df, "fragment")

        features_df, fragments_df = self.extract_batch(
            self.spectral_library._precursor_df
        )
        precursor_df = self.fdr_correction(features_df)

        precursor_df = precursor_df[precursor_df["qval"] <= self.config["fdr"]["fdr"]]
        self.log_precursor_df(precursor_df)

        return precursor_df, fragments_df

    def log_precursor_df(self, precursor_df):
        total_precursors = len(precursor_df)

        target_precursors = len(precursor_df[precursor_df["decoy"] == 0])
        target_precursors_percentages = target_precursors / total_precursors * 100
        decoy_precursors = len(precursor_df[precursor_df["decoy"] == 1])
        decoy_precursors_percentages = decoy_precursors / total_precursors * 100

        self.reporter.log_string(
            f"============================= Precursor FDR =============================",
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

        self.reporter.log_string(f"", verbosity="progress")
        self.reporter.log_string(f"Precursor Summary:", verbosity="progress")

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

        self.reporter.log_string(f"", verbosity="progress")
        self.reporter.log_string(f"Protein Summary:", verbosity="progress")

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
            f"=========================================================================",
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

        if not "multiplexing" in self.config:
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
            mobility_column="mobility_calibrated",
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

# native imports
import logging

logger = logging.getLogger()
import typing

# alphadia imports
from alphadia import plexscoring, utils, fragcomp
from alphadia import fdrexperimental as fdrx
from alphadia.peakgroup import search
from alphadia.workflow import manager, base

# alpha family imports
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.peptide.fragment import get_charged_frag_types

# third party imports
import numpy as np
import pandas as pd
import seaborn as sns

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


class CalibrationError(Exception):
    """Raised when calibration fails"""

    pass


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
        print("Testing!")

    def load(
        self,
        dia_data_path: str,
        spectral_library: SpecLibBase,
    ) -> None:
        super().load(
            dia_data_path,
            spectral_library,
        )

        self.initialize_managers()


    def initialize_managers(
            self,
    ) -> None :
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
                "fitted_params" : [],
                "ms1_list": [self.config["search_initial"]["initial_ms1_tolerance"]],
                "ms2_list": [self.config["search_initial"]["initial_ms2_tolerance"]],
                "rt_list": [self.config["search_initial"]["initial_rt_tolerance"]],
                "precursor_number" : [],
                "calibrations" : {},
                "param_colname_dict" : {
                    "ms2" : "mz",
                    "ms1" : "mz",
                    "rt" : "rt",
                    "mobility" : "mobility"
                },
                "param_calibration_order" : self.config["calibration"]["param_calibration_order"],
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

        self.params_to_omit = []
        if not self.dia_data.has_mobility :
            self.params_to_omit.append("mobility")
        if not self.dia_data.has_ms1:
            self.params_to_omit.append("ms1")
        self.com.optimal_epoch_for_last_param = 0

    def start_of_epoch(self, current_epoch):
        self.com.current_epoch = current_epoch

        # if self.neptune is not None:
        #    self.neptune["eval/epoch"].log(current_epoch)

        if self.param_index == 0 and self.com.current_epoch == 0:
            self.elution_group_order = self.spectral_library.precursor_df[
                "elution_group_idx"
            ].unique()
            np.random.shuffle(self.elution_group_order)
            self.com.classifier_id_input = "_".join(np.asarray([-1, 0], dtype=str))
            self.com.classifier_id_output = "_".join(np.asarray([-1, 0], dtype=str))
        else:
            self.com.classifier_id_input = "_".join(np.asarray([self.param_index - 1, self.com.optimal_epoch_for_last_param], dtype=str))
            self.com.classifier_id_output = "_".join(np.asarray([self.param_index, self.com.current_epoch], dtype=str))
            self.epochs_since_freeze += 1
            if self.epochs_since_freeze == 1:
                self.reporter.log_string("Calibration dataframe has been frozen.", verbosity="progress")

        self.calibration_manager.predict(
            self.spectral_library._precursor_df, "precursor"
        )
        self.calibration_manager.predict(self.spectral_library._fragment_df, "fragment")

    
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

    def check_epoch_conditions(self, params_to_fit=["ms1", "ms2", "rt", "mobility"]):
        continue_calibration = False

        self.reporter.log_string(
            "=== checking if epoch conditions were reached ===", verbosity="info"
        )
        if self.dia_data.has_ms1 and "ms1" in params_to_fit:
            self.reporter.log_string(
                f" {'ms1_error':<15}: {self.com.ms1_error:.4f} > {self.config['search']['target_ms1_tolerance']}",
                verbosity="info",
            )

        if "ms2" in params_to_fit:
            self.reporter.log_string(
                f" {'ms2_error':<15}: {self.com.ms2_error:.4f} > {self.config['search']['target_ms2_tolerance']}",
                verbosity="info",
            )

        if "rt" in params_to_fit:
            self.reporter.log_string(
                f" {'rt_error':<15}: {self.com.rt_error:.4f} > {self.config['search']['target_rt_tolerance']}",
                verbosity="info",
            )


        if self.dia_data.has_mobility and "mobility" in params_to_fit:
            self.reporter.log_string(
                f" {'mobility_error':<15}: {self.com.mobility_error:.4f} > {self.config['search']['target_mobility_tolerance']}",
                verbosity="info",
            )

        if self.com.current_epoch < self.config["calibration"]["min_epochs"] - 1:
            self.reporter.log_string(
                f"❌ {'current_epoch':<15}: {self.com.current_epoch} < {self.config['calibration']['min_epochs'] - 1}",
                verbosity="info",
            )
            continue_calibration = True
        else:
            self.reporter.log_string(
                f"✅ {'current_epoch':<15}: {self.com.current_epoch} >= {self.config['calibration']['min_epochs'] - 1}",
                verbosity="info",
            )
            

        self.reporter.log_string(
            "==============================================", verbosity="info"
        )
        
        if self.com.current_epoch < 2 or self.epochs_since_freeze < 2:
            continue_calibration = True
        else:
            
            if self.com.precursor_number[-1] > 1.1 * self.com.precursor_number[-2] or self.com.precursor_number[-1] > 1.1 * self.com.precursor_number[-3]:
                continue_calibration = True
                
            else:
                print(self.com.current_epoch)
                for param in np.unique(self.com.fitted_params):
                    optimal_epoch_offset = - np.argmax([self.com.precursor_number[-2], self.com.precursor_number[-3]])
                    if param == "ms2":
                        self.spectral_library.fragment_df[f"{self.com.param_colname_dict[param]}_calibrated"] = self.com.calibrations[self.param_index][self.com.current_epoch + optimal_epoch_offset][param]
                        print(self.com.current_epoch)
                        print(optimal_epoch_offset)
                    else:
                        print(self.com.current_epoch)
                        print(optimal_epoch_offset)
                        self.spectral_library.precursor_df[f"{self.com.param_colname_dict[param]}_calibrated"] = self.com.calibrations[self.param_index][self.com.current_epoch + optimal_epoch_offset][param]
                    
                    self.com.optimal_epoch_for_last_param = self.com.current_epoch + optimal_epoch_offset
                    self.com.current_epoch = 0 
                    self.reporter.log_string(f"Optimal values for {params_to_fit} is/are {self.com.calibrations[self.param_index][self.com.optimal_epoch_for_last_param]}.", verbosity="progress")
        
        return continue_calibration

    def flatten_list(self, nested_list):
        return [x for y in nested_list for x in y]
        
    def calibration(self, param_calibration_order=["ms1", "ms2", "rt", "mobility"]):
        if (
            self.calibration_manager.is_fitted
            and self.calibration_manager.is_loaded_from_file
        ):
            self.reporter.log_string(
                "Skipping calibration as existing calibration was found",
                verbosity="progress",
            )
            return
        self.param_calibration_order = param_calibration_order
        ############ oemd edit #################
        self.feature_columns = feature_columns
        ########################################
        self.start_of_calibration()
        self.epochs_since_freeze = 0
        for param_index, params_to_fit in enumerate(self.param_calibration_order):
            self.reporter.log_string(f"Fitting {params_to_fit}.", verbosity="info")
            self.param_index = param_index
            self.com.calibrations[self.param_index] = {}
            for current_epoch in range(self.config["calibration"]["max_epochs"]):
                if self.check_epoch_conditions(params_to_fit):
                    pass
                else:
                    break
                self.param_index = param_index
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
    
                    feature_df, fragment_df = self.extract_batch(batch_df)  # oemd2: Important
                    features += [feature_df]
                    fragments += [fragment_df]
                    features_df = pd.concat(features)
                    fragments_df = pd.concat(fragments)
                    
                    self.reporter.log_string(
                        f"=== Epoch {self.com.current_epoch}, step {current_step}, extracted {len(feature_df)} precursors and {len(fragment_df)} fragments ===",
                        verbosity="progress",
                    )
                    precursor_df = self.fdr_correction(features_df, fragments_df, classifier_id_input=self.com.classifier_id_input, classifier_id_output=self.com.classifier_id_output)
    
                    if self.check_recalibration(precursor_df):
                        self.recalibration(precursor_df, fragments_df, params_to_fit)
                        
                        break
                    else:
                        # check if last step has been reached
                        if current_step == len(self.batch_plan) - 1:
                            self.reporter.log_string(
                                "Searched all data without finding recalibration target",
                                verbosity="error",
                            )
                            raise CalibrationError(
                                "Searched all data without finding recalibration target"
                            )
                self.com.fitted_params.extend(params_to_fit)
                self.end_of_epoch(current_step=current_step)

            self.end_of_calibration()

    def end_of_epoch(self, current_step=0):
        self.batch_plan = [(self.batch_plan[0][0], self.batch_plan[current_step][1])]
        pass

    def end_of_calibration(self):
        # self.calibration_manager.predict(self.spectral_library._precursor_df, 'precursor')
        # self.calibration_manager.predict(self.spectral_library._fragment_df, 'fragment')
        self.calibration_manager.save()
        pass

    def recalibration(self, precursor_df, fragments_df, params_to_fit=["ms1", "ms2", "rt", "mobility"]):
        precursor_df_filtered = precursor_df[precursor_df["qval"] < 0.01]
        precursor_df_filtered = precursor_df_filtered[
            precursor_df_filtered["decoy"] == 0
        ]

        precursor_params_to_skip = ["mz" if "ms1" not in params_to_fit else "", "rt" if "rt" not in params_to_fit else "", "mobility" if "mobility" not in params_to_fit else ""] 
        
        self.calibration_manager.fit(
            precursor_df_filtered,
            "precursor",
            plot=True,
            skip=precursor_params_to_skip,
            # neptune_run = self.neptune
        )

        if 'ms2' in params_to_fit:
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
    
            print(f"fragments_df_filtered: {len(fragments_df_filtered)}")
    
            self.calibration_manager.fit(
                fragments_df_filtered,
                "fragment",
                plot=True,
                # neptune_run = self.neptune
            )
            
        self.com.fit(
                {
                    "column_type": "calibrated",
                    "num_candidates": self.config["search"]["target_num_candidates"],
                }
        )
        
        if "rt" in params_to_fit:
            rt_99 = self.calibration_manager.get_estimator("precursor", "rt").ci(
                precursor_df_filtered, 0.99
            )
            self.com.fit(
                {
                    "rt_error": max(rt_99 * 1.1, self.config["search"]["target_rt_tolerance"]),
                }
            )
            
        if "ms2" in params_to_fit:
            m2_99 = self.calibration_manager.get_estimator("fragment", "mz").ci(
                fragments_df_filtered, 0.99
            )
            self.com.fit(
                {
                    "ms2_error": max(m2_99 * 1.1, self.config["search"]["target_ms2_tolerance"]),
                }
            )

        if self.dia_data.has_ms1 and "ms1" in params_to_fit:
            m1_99 = self.calibration_manager.get_estimator("precursor", "mz").ci(
                precursor_df_filtered, 0.99
            )
            self.com.fit(
                {
                    "ms1_error": max(
                        m1_99 * 1.1, self.config["search"]["target_ms1_tolerance"]
                    ),
                }
            )

        if self.dia_data.has_mobility and "mobility" in params_to_fit:
            mobility_99 = self.calibration_manager.get_estimator(
                "precursor", "mobility"
            ).ci(precursor_df_filtered, 0.99)
            self.com.fit(
                {
                    "mobility_error": max(
                        mobility_99 * 1.1, self.config["search"]["target_mobility_tolerance"]
                    ),
                }
            )

            # if self.neptune is not None:
            #    self.neptune['eval/99_mobility_error'].log(mobility_99)

        percentile_001 = np.percentile(precursor_df_filtered["score"], 0.1)
        print("score cutoff", percentile_001)

        self.optimization_manager.fit(
            {
                "fwhm_rt": precursor_df_filtered["cycle_fwhm"].median(),
                "fwhm_mobility": precursor_df_filtered["mobility_fwhm"].median(),
                "score_cutoff": percentile_001,
            }
        )

        self.com.calibrations[self.param_index][self.com.current_epoch] = {}
        for param in self.flatten_list(self.param_calibration_order):
            if param == "ms2":
                if param in np.unique(self.com.fitted_params) and param not in self.params_to_omit:
                    self.com.calibrations[self.param_index][self.com.current_epoch][param] = self.spectral_library.fragment_df[f"{self.com.param_colname_dict[param]}_calibrated"]
                else:
                    self.com.calibrations[self.param_index][self.com.current_epoch][param] = self.spectral_library.fragment_df[f"{self.com.param_colname_dict[param]}_library"]
            else:
                if param in np.unique(self.com.fitted_params) and param not in self.params_to_omit:
                    self.com.calibrations[self.param_index][self.com.current_epoch][param] = self.spectral_library.precursor_df[f"{self.com.param_colname_dict[param]}_calibrated"]
                else:
                    self.com.calibrations[self.param_index][self.com.current_epoch][param] = self.spectral_library.precursor_df[f"{self.com.param_colname_dict[param]}_library"]


        self.precursor_df_at_fdr = precursor_df_filtered
        self.com.ms1_list.append(self.com.ms1_error)
        self.com.ms2_list.append(self.com.ms2_error)
        self.com.rt_list.append(self.com.rt_error)
        self.com.precursor_number.append(len(self.precursor_df_at_fdr))

        
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

    def fdr_correction(self, features_df, df_fragments, classifier_id_input=None, classifier_id_output=None):
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
            classifier_id_input=classifier_id_input,
            classifier_id_output=classifier_id_output,
            # neptune_run=self.neptune
        )

    def extract_batch(self, batch_df, apply_cutoff=False, params_to_fit=["ms1", "ms2", "rt", "mobility"]):
        self.reporter.log_string(
            f"Extracting batch of {len(batch_df)} precursors", verbosity="progress"
        )

        config = search.HybridCandidateConfig()
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
                "min_size_rt": self.config["search"]["quant_window"],
            }
        )

        extraction = search.HybridCandidateSelection(
            self.dia_data.jitclass(),
            batch_df,
            self.spectral_library.fragment_df,
            config.jitclass(),
            rt_column=f"rt_{self.com.column_type}" 
            if "rt" in self.com.fitted_params 
            else "rt_library",
            mobility_column=f"mobility_{self.com.column_type}"
            if self.dia_data.has_mobility and "mobility" in self.com.fitted_params
            else "mobility_library",
            precursor_mz_column=f"mz_{self.com.column_type}"
            if self.dia_data.has_ms1 and "ms1" in self.com.fitted_params
            else "mz_library",
            fragment_mz_column=f"mz_{self.com.column_type}"
            if "ms2" in self.com.fitted_params
            else "mz_library",
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
                "precursor_mz_tolerance": self.com.ms1_error,
                "fragment_mz_tolerance": self.com.ms2_error,
                "exclude_shared_ions": self.config["search"]["exclude_shared_ions"],
                "quant_window": self.config["search"]["quant_window"],
                "quant_all": self.config["search"]["quant_all"],
            }
        )

        candidate_scoring = plexscoring.CandidateScoring(
            self.dia_data.jitclass(),
            self.spectral_library._precursor_df,
            self.spectral_library._fragment_df,
            config=config,
            rt_column=f"rt_{self.com.column_type}" 
            if "rt" in self.com.fitted_params 
            else "rt_library",
            mobility_column=f"mobility_{self.com.column_type}"
            if self.dia_data.has_mobility and "mobility" in self.com.fitted_params
            else "mobility_library",
            precursor_mz_column=f"mz_{self.com.column_type}"
            if self.dia_data.has_ms1 and "ms1" in self.com.fitted_params
            else "mz_library",
            fragment_mz_column=f"mz_{self.com.column_type}"
            if "ms2" in self.com.fitted_params
            else "mz_library",
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self.config["general"]["thread_count"],
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df

    def extraction(self):
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
            self.spectral_library._precursor_df,
            apply_cutoff=True,
        )

        precursor_df = self.fdr_correction(features_df, fragments_df)

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

        return precursor_df, fragments_df

    def log_precursor_df(self, precursor_df):
        total_precursors = len(precursor_df)

        target_precursors = len(precursor_df[precursor_df["decoy"] == 0])
        target_precursors_percentages = target_precursors / total_precursors * 100
        decoy_precursors = len(precursor_df[precursor_df["decoy"] == 1])
        decoy_precursors_percentages = decoy_precursors / total_precursors * 100

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
            mobility_column=f"mobility_calibrated"
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
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
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
            f"=== Transfer learning quantification ===",
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
            f"Calibrating library",
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
            precursor_mz_column=f"mz_calibrated",
            fragment_mz_column=f"mz_calibrated",
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
    fragment_types: typing.List[str] = ["b", "y"],
    max_charge: int = 2,
    optional_columns: typing.Union[typing.List[str], None] = None,
) -> typing.Tuple[SpecLibFlat, pd.DataFrame]:
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

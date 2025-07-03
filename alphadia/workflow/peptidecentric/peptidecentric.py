import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase
from workflow.peptidecentric.optimization_handler import OptimizationHandler

from alphadia._fdrx.models.logistic_regression import LogisticRegressionClassifier
from alphadia._fdrx.models.two_step_classifier import TwoStepClassifier
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.fragcomp.utils import candidate_hash
from alphadia.workflow import base
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.recalibration_handler import RecalibrationHandler
from alphadia.workflow.peptidecentric.requantification_handler import (
    RequantificationHandler,
)
from alphadia.workflow.peptidecentric.utils import fdr_correction, log_precursor_df

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

        self._extraction_handler: ExtractionHandler | None = None
        self._recalibration_handler: RecalibrationHandler | None = None
        self._requantification_handler: RequantificationHandler | None = None
        self._optimization_handler: OptimizationHandler | None = None

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
        self._optimization_handler = OptimizationHandler(
            self.config,
            self.optimization_manager,
            self.calibration_manager,
            self.fdr_manager,
            self._extraction_handler,
            self._recalibration_handler,
            self.reporter,
            self.spectral_library,
            self.dia_data,
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

    def search_parameter_optimization(self):
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

        self._optimization_handler.search_parameter_optimization()

        self._save_managers()

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

        precursor_df = fdr_correction(
            self.fdr_manager,
            self.config,
            self.dia_data.cycle,
            features_df,
            fragments_df,
            self.optimization_manager.classifier_version,
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

        log_precursor_df(self.reporter, precursor_df)

        return precursor_df, fragments_df

    def _lazy_init_requantification_handler(self):
        """Initializes the requantification handler if it is not already initialized."""
        if not self._requantification_handler:
            self._requantification_handler = RequantificationHandler(
                self.config,
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
        """TODO.

        Delegates to RequantificationHandler.requantify(), see docstring there for more details.
        """
        self._lazy_init_requantification_handler()

        psm_df = self._requantification_handler.requantify(self.dia_data, psm_df)

        log_precursor_df(self.reporter, psm_df)

        return psm_df

    def requantify_fragments(
        self, psm_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Requantify confident precursor identifications for transfer learning.

        Delegates to RequantificationHandler.requantify_fragments(), see docstring there for more details.
        """
        self._lazy_init_requantification_handler()

        return self._requantification_handler.requantify_fragments(
            self.dia_data, psm_df
        )

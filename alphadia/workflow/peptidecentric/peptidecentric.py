import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia._fdrx.models.logistic_regression import LogisticRegressionClassifier
from alphadia._fdrx.models.two_step_classifier import TwoStepClassifier
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.fragcomp.utils import candidate_hash
from alphadia.workflow import base
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.library_init import init_spectral_library
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.recalibration_handler import RecalibrationHandler
from alphadia.workflow.peptidecentric.requantification_handler import (
    RequantificationHandler,
)
from alphadia.workflow.peptidecentric.utils import (
    fdr_correction,
    feature_columns,
    log_precursor_df,
)


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
        self.fdr_manager: FDRManager | None = None
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

        config_fdr = self.config["fdr"]
        self.fdr_manager = FDRManager(
            feature_columns=feature_columns,
            classifier_base=_get_classifier_base(
                enable_two_step_classifier=config_fdr["enable_two_step_classifier"],
                two_step_classifier_max_iterations=config_fdr[
                    "two_step_classifier_max_iterations"
                ],
                enable_nn_hyperparameter_tuning=config_fdr[
                    "enable_nn_hyperparameter_tuning"
                ],
                fdr_cutoff=config_fdr["fdr"],
            ),
            figure_path=self._figure_path,
        )

        init_spectral_library(
            self.config,
            self.dia_data.cycle,
            self.dia_data.rt_values,
            self.reporter,
            self.spectral_library,
        )
        self._column_name_handler = ColumnNameHandler(
            self.optimization_manager,
            dia_data_has_ms1=self.dia_data.has_ms1,
            dia_data_has_mobility=self.dia_data.has_mobility,
        )

        self._extraction_handler = ExtractionHandler(
            self.config,
            self.optimization_manager,
            self.reporter,
            self._column_name_handler,
            self.spectral_library,
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

    def _save_managers(self):
        """Saves the calibration, optimization and FDR managers to disk so that they can be reused if needed.
        Note the timing manager is not saved at this point as it is saved with every call to it.
        The FDR manager is not saved because it is not used in subsequent parts of the workflow.
        """
        self.calibration_manager.save()
        self.optimization_manager.save()  # this replaces the .save() call when the optimization manager is fitted, since there seems little point in saving an intermediate optimization manager.

    def _lazy_init_requantification_handler(self):
        """Initializes the requantification handler if it is not already initialized."""
        if not self._requantification_handler:
            self._requantification_handler = RequantificationHandler(
                self.config,
                self.calibration_manager,
                self.fdr_manager,
                self.reporter,
                self._column_name_handler,
                self.spectral_library,
            )

    def search_parameter_optimization(self):
        """Performs optimization of the search parameters.

        Delegates the actual optimization to the OptimizationHandler.search_parameter_optimization(), see docstring there for more details.
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

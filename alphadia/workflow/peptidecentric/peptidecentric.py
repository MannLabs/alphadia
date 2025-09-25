import os

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.fdr._fdrx.models.logistic_regression import LogisticRegressionClassifier
from alphadia.fdr._fdrx.models.two_step_classifier import TwoStepClassifier
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.fragcomp.utils import candidate_hash
from alphadia.workflow import base
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationGroups
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.timing_manager import TimingManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.library_init import init_spectral_library
from alphadia.workflow.peptidecentric.multiplexing_requantification_handler import (
    MultiplexingRequantificationHandler,
)
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.transfer_library_requantification_handler import (
    TransferLibraryRequantificationHandler,
)
from alphadia.workflow.peptidecentric.utils import (
    feature_columns,
    log_precursor_df,
    use_timing_manager,
)


def _get_classifier_base(
    enable_two_step_classifier: bool = False,
    two_step_classifier_max_iterations: int = 5,
    enable_nn_hyperparameter_tuning: bool = False,
    fdr_cutoff: float = 0.01,
    random_state: int | None = None,
) -> BinaryClassifierLegacyNewBatching | TwoStepClassifier:
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

    random_state : int | None, optional
        Random state for reproducibility. Default is None.

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
        random_state=random_state,
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
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            instance_name,
            config,
            quant_path,
        )
        self._fdr_manager: FDRManager | None = None

        self._timing_manager: TimingManager = TimingManager(
            path=os.path.join(self.path, self.TIMING_MANAGER_PKL_NAME),
            load_from_file=self.config["general"]["reuse_calibration"],
        )

        if random_state is not None:
            rng = np.random.default_rng(seed=random_state)
            self._random_state_fdr_classifier, self._random_state_fdr_manager = (
                rng.integers(0, 1_000_000, size=(2,))
            )
        else:
            self._random_state_fdr_classifier, self._random_state_fdr_manager = (
                None,
                None,
            )

    @use_timing_manager("load")
    def load(
        self,
        dia_data_path: str,
        spectral_library: SpecLibFlat,
    ) -> None:
        super().load(
            dia_data_path,
            spectral_library,
        )

        self.reporter.log_string(
            f"Initializing workflow {self.instance_name}", verbosity="progress"
        )
        config_fdr = self.config["fdr"]
        self._fdr_manager = FDRManager(
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
                random_state=self._random_state_fdr_classifier,
            ),
            dia_cycle=self.dia_data.cycle,
            config=self.config,
            figure_path=self._figure_path,
            random_state=self._random_state_fdr_manager,
        )

        init_spectral_library(
            self.dia_data.cycle,
            self.dia_data.rt_values,
            self.reporter,
            self.spectral_library,
            self.config["search"]["channel_filter"],
        )

    def _save_managers(self):
        """Saves the calibration, optimization and FDR managers to disk so that they can be reused if needed.
        Note the timing manager is not saved at this point as it is saved with every call to it.
        The FDR manager is not saved because it is not used in subsequent parts of the workflow.
        """
        self.calibration_manager.save()
        self.optimization_manager.save()  # this replaces the .save() call when the optimization manager is fitted, since there seems little point in saving an intermediate optimization manager.

    @use_timing_manager("optimization")
    def search_parameter_optimization(self):
        """Performs optimization of the search parameters.

        Delegates the actual optimization to the OptimizationHandler.search_parameter_optimization(), see docstring there for more details.
        """
        # First check to see if the calibration has already been performed. Return if so.
        if (
            self.calibration_manager.is_loaded_from_file
            and self.calibration_manager.all_fitted
        ):
            self.reporter.log_string(
                "Skipping calibration as existing calibration was found",
                verbosity="progress",
            )
            return

        optimization_handler = OptimizationHandler(
            self.config,
            self.optimization_manager,
            self.calibration_manager,
            self._fdr_manager,
            self.reporter,
            self.spectral_library,
            self.dia_data,
            self._figure_path,
            self._dia_data_ng,
        )

        optimization_handler.search_parameter_optimization()

        self._save_managers()

        self.calibration_manager.predict(
            self.spectral_library.precursor_df, CalibrationGroups.PRECURSOR
        )
        self.calibration_manager.predict(
            self.spectral_library.fragment_df, CalibrationGroups.FRAGMENT
        )

    @use_timing_manager("extraction")
    def extraction(self):
        extraction_handler = ExtractionHandler.create_handler(
            self.config,
            self.optimization_manager,
            self.reporter,
            ColumnNameHandler(
                self.calibration_manager,
                dia_data_has_ms1=self.dia_data.has_ms1,
                dia_data_has_mobility=self.dia_data.has_mobility,
            ),
        )

        features_df, fragments_df = extraction_handler.extract_batch(
            (self.dia_data, self._dia_data_ng)
            if self._dia_data_ng is not None
            else self.dia_data,
            self.spectral_library,
            apply_cutoff=True,
        )

        self.reporter.log_string(
            f"=== Performing FDR correction with classifier version {self.optimization_manager.classifier_version} ===",
        )

        decoy_strategy = (
            "precursor_channel_wise"
            if self._config["fdr"]["channel_wise_fdr"]
            else "precursor"
        )

        precursor_df = self._fdr_manager.fit_predict(
            features_df,
            decoy_strategy=decoy_strategy,
            competetive=self._config["fdr"]["competetive_scoring"],
            df_fragments=fragments_df,
            version=self.optimization_manager.classifier_version,
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

    @use_timing_manager("requantify")
    def requantify(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """TODO.

        Delegates to MultiplexingRequantificationHandler.requantify(), see docstring there for more details.
        """

        requantification_handler = MultiplexingRequantificationHandler(
            self.config,
            self.calibration_manager,
            self._fdr_manager,
            self.reporter,
            ColumnNameHandler(
                self.calibration_manager,
                dia_data_has_ms1=self.dia_data.has_ms1,
                dia_data_has_mobility=self.dia_data.has_mobility,
            ),
            self.spectral_library,
        )

        psm_df = requantification_handler.requantify(self.dia_data, psm_df)

        psm_df = psm_df[psm_df["qval"] <= self.config["fdr"]["fdr"]]

        log_precursor_df(self.reporter, psm_df)

        return psm_df

    @use_timing_manager("requantify_fragments")
    def requantify_fragments(
        self, psm_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Requantify confident precursor identifications for transfer learning.

        Delegates to TransferLibraryRequantificationHandler.requantify(), see docstring there for more details.
        """

        requantification_handler = TransferLibraryRequantificationHandler(
            self.config,
            self.calibration_manager,
            self.optimization_manager,
            ColumnNameHandler(
                self.calibration_manager,
                dia_data_has_ms1=self.dia_data.has_ms1,
                dia_data_has_mobility=self.dia_data.has_mobility,
            ),
            self.reporter,
        )

        return requantification_handler.requantify(self.dia_data, psm_df)

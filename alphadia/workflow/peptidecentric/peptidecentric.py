import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import FeatureTransform
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.workflow import base
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationGroups
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.timing_manager import TimingManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.library_init import init_spectral_library
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.transfer_library_requantification_handler import (
    TransferLibraryRequantificationHandler,
)
from alphadia.workflow.peptidecentric.utils import (
    use_timing_manager,
)


def _get_classifier_base(
    enable_nn_hyperparameter_tuning: bool = False,
    feature_transform: str = FeatureTransform.NONE,
    random_state: int | None = None,
) -> BinaryClassifierLegacyNewBatching:
    """Creates and returns a classifier base instance.

    Parameters
    ----------
    enable_nn_hyperparameter_tuning: bool, optional
        If True, uses hyperparameter tuning for the neural network.
        If False (default), uses default hyperparameters for the neural network.

    feature_transform: str, optional
        Transform applied to the features before they enter the network, either "none" (default) or "quantile".

    random_state : int | None, optional
        Random state for reproducibility. Default is None.

    Returns
    -------
    BinaryClassifierLegacyNewBatching
        Neural network
    """
    return BinaryClassifierLegacyNewBatching(
        test_size=0.001,
        batch_size=5000,
        learning_rate=0.001,
        epochs=10,
        experimental_hyperparameter_tuning=enable_nn_hyperparameter_tuning,
        feature_transform=feature_transform,
        random_state=random_state,
    )


class PeptideCentricWorkflow(base.WorkflowBase, ABC):
    """Peptide-centric workflow for a single raw file.

    Each extraction backend has its own subclass, chosen by `workflow_factory.get_workflow_class`.
    """

    OPTIMIZATION_HANDLER_CLASS: type[OptimizationHandler]
    EXTRACTION_HANDLER_CLASS: type[ExtractionHandler]

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
            feature_columns=self._classifier_feature_columns(),
            classifier_base=_get_classifier_base(
                enable_nn_hyperparameter_tuning=config_fdr[
                    "enable_nn_hyperparameter_tuning"
                ],
                feature_transform=config_fdr["feature_transform"],
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

    @abstractmethod
    def _classifier_feature_columns(self) -> list[str]:
        """Feature columns the FDR classifier is trained on."""

    def _create_extraction_handler(self) -> ExtractionHandler:
        """Create an extraction handler for the current calibration state."""
        return self.EXTRACTION_HANDLER_CLASS(
            self.config,
            self.optimization_manager,
            self._fdr_manager,
            self.reporter,
            ColumnNameHandler(
                self.calibration_manager,
                dia_data_has_ms1=self.dia_data.has_ms1,
                dia_data_has_mobility=self.dia_data.has_mobility,
            ),
        )

    def _save_managers(self):
        """Saves the optimization manager to disk so that it can be reused if needed.
        Note the timing manager is not saved at this point as it is saved with every call to it.
        The FDR manager is not saved because it is not used in subsequent parts of the workflow.
        The calibration metrics are written out as JSON for the output statistics.
        """
        self.calibration_manager.save_stats(
            os.path.join(self.path, self.CALIBRATION_STATS_FILE_NAME)
        )
        self.optimization_manager.save()  # this replaces the .save() call when the optimization manager is fitted, since there seems little point in saving an intermediate optimization manager.

    @use_timing_manager("optimization")
    def search_parameter_optimization(self):
        """Performs optimization of the search parameters.

        Delegates the actual optimization to the OptimizationHandler.search_parameter_optimization(), see docstring there for more details.
        """
        optimization_handler = self.OPTIMIZATION_HANDLER_CLASS(
            self.config,
            self.optimization_manager,
            self.calibration_manager,
            self._fdr_manager,
            self.reporter,
            self.spectral_library,
            self.dia_data,
            self._figure_path,
        )

        optimization_handler.search_parameter_optimization()

        self._save_managers()

        self.calibration_manager.predict(
            self.spectral_library.precursor_df, CalibrationGroups.PRECURSOR
        )
        self.calibration_manager.predict(
            self.spectral_library.fragment_df, CalibrationGroups.FRAGMENT
        )

    @abstractmethod
    def extraction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract, score, FDR-filter and quantify the precursors of the whole library.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Precursors below the FDR threshold and their quantified fragments.
        """

    def requantify(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """Requantify the precursors across multiplexing channels."""
        raise NotImplementedError(
            f"Multiplexing is not supported by {type(self).__name__}."
        )

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
            self._fdr_manager,
            ColumnNameHandler(
                self.calibration_manager,
                dia_data_has_ms1=self.dia_data.has_ms1,
                dia_data_has_mobility=self.dia_data.has_mobility,
            ),
            self.reporter,
            self.EXTRACTION_HANDLER_CLASS,
        )

        return requantification_handler.requantify(self.dia_data, psm_df)

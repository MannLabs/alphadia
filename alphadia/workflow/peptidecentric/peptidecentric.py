import os

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import ConfigKeys
from alphadia.fdr import joint_features, joint_rescoring, mbr_transfer
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
from alphadia.workflow.peptidecentric.rt_refinement import refine_calibrated_rt
from alphadia.workflow.peptidecentric.transfer_library_requantification_handler import (
    TransferLibraryRequantificationHandler,
)
from alphadia.workflow.peptidecentric.utils import (
    add_library_fragment_intensity,
    get_classifier_feature_columns,
    log_precursor_df,
    use_timing_manager,
)

# the MBR step searches the library the first search wrote under this name (SearchPlanOutput.LIBRARY_OUTPUT)
MBR_LIBRARY_FILE_NAME = "speclib.mbr.hdf"


def _get_classifier_base(
    enable_nn_hyperparameter_tuning: bool = False,
    random_state: int | None = None,
) -> BinaryClassifierLegacyNewBatching:
    """Creates and returns a classifier base instance.

    Parameters
    ----------
    enable_nn_hyperparameter_tuning: bool, optional
        If True, uses hyperparameter tuning for the neural network.
        If False (default), uses default hyperparameters for the neural network.

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
        random_state=random_state,
    )


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
        is_mbr_step = (
            os.path.basename(self.config["library_path"] or "") == MBR_LIBRARY_FILE_NAME
        )
        self._pooled_transfers = is_mbr_step and config_fdr["mbr_pooled_transfer_model"]
        # the multiplexing requantification refits the FDR per channel, and the python backend has no final round
        self._joint_rescoring = (
            config_fdr["joint_rescoring"]
            and not is_mbr_step
            and len(self.config.get(ConfigKeys.RAW_PATHS) or [])
            >= joint_rescoring.MIN_RUNS
            and not self.config["multiplexing"]["enabled"]
            and self._config["search"]["extraction_backend"] == "rust"
        )
        # files of an earlier search into this folder must not reach the output step
        joint_features.remove_run_files(self.path)
        self._fdr_manager = FDRManager(
            feature_columns=get_classifier_feature_columns(
                self._config["search"]["extraction_backend"]
            ),
            classifier_base=_get_classifier_base(
                enable_nn_hyperparameter_tuning=config_fdr[
                    "enable_nn_hyperparameter_tuning"
                ],
                random_state=self._random_state_fdr_classifier,
            ),
            dia_cycle=self.dia_data.cycle,
            config=self.config,
            figure_path=self._figure_path,
            random_state=self._random_state_fdr_manager,
            feature_matrix_path=self.path
            if config_fdr["save_feature_matrix"]
            else None,
            # the MBR library builder adds the extra decoys only next to the kept ones
            decoy_multiplicity=config_fdr["mbr_decoys_per_target"]
            if is_mbr_step and config_fdr["keep_decoys_in_mbr_library"]
            else 1,
            transfer_features_path=os.path.join(
                self.path, mbr_transfer.TRANSFER_FEATURES_FILE_NAME
            )
            if self._pooled_transfers
            else None,
            joint_rescoring_folder=self.path if self._joint_rescoring else None,
        )

        init_spectral_library(
            self.dia_data.cycle,
            self.dia_data.rt_values,
            self.reporter,
            self.spectral_library,
            self.config["search"]["channel_filter"],
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
        optimization_handler = OptimizationHandler(
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

        if (
            self.config["calibration"]["rt_sequence_refinement"]
            and optimization_handler.calibration_identifications is not None
        ):
            refine_calibrated_rt(
                self.spectral_library.precursor_df,
                optimization_handler.calibration_identifications,
            )

    @use_timing_manager("extraction")
    def extraction(self):
        extraction_handler = ExtractionHandler.create_handler(
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

        # MS1 features are scored in the final extraction only: calibration rounds set the raw-score cutoff from their
        # IDs, and a strong new feature there lowers the cutoff and floods the final extraction. The rust DIA data
        # always reports has_ms1 False; without MS1 scans the features are constant and the calibration falls back.
        if self._config["search"]["extraction_backend"] == "rust":
            extraction_handler.enable_ms1_scoring()

        candidates_df = extraction_handler.select_candidates(
            self.dia_data,
            self.spectral_library,
            apply_cutoff=True,
        )

        if self._config["search"]["extraction_backend"] == "python":
            precursor_quantified_w_features_df, fragments_df = (
                extraction_handler.score_and_quantify_candidates(
                    candidates_df, self.dia_data, self.spectral_library
                )
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
                precursor_quantified_w_features_df,
                decoy_strategy=decoy_strategy,
                competitive=self._config["fdr"]["competitive_scoring"],
                df_fragments=fragments_df,
                version=self.optimization_manager.classifier_version,
            )

            precursor_df = precursor_df[
                precursor_df["qval"] <= self.config["fdr"]["fdr"]
            ]

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

        else:
            precursor_w_features_df = extraction_handler.score_candidates(
                candidates_df, self.dia_data, self.spectral_library
            )

            candidates_fdr_df, precursor_fdr_df = (
                extraction_handler.perform_fdr_and_filter_candidates(
                    precursor_w_features_df,
                    candidates_df,
                    self.dia_data,
                    self.spectral_library,
                    keep_transfers=self._pooled_transfers,
                    keep_for_joint_rescoring=self._joint_rescoring,
                )
            )

            precursor_df, fragments_df = extraction_handler.quantify_candidates(
                candidates_fdr_df,
                precursor_fdr_df,
                self.dia_data,
                self.spectral_library,
            )

        fragments_df = add_library_fragment_intensity(
            fragments_df, self.spectral_library
        )

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
            self._fdr_manager,
            ColumnNameHandler(
                self.calibration_manager,
                dia_data_has_ms1=self.dia_data.has_ms1,
                dia_data_has_mobility=self.dia_data.has_mobility,
            ),
            self.reporter,
        )

        return requantification_handler.requantify(self.dia_data, psm_df)

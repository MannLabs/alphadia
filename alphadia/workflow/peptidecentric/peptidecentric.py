import os

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

try:  # noqa: SIM105
    from alphadia.workflow.peptidecentric.ng.ng_mapper import get_feature_names
except ImportError:
    pass
from alphadia.constants.keys import FdrClassifier
from alphadia.fdr.classifiers import (
    BinaryClassifierLegacyNewBatching,
    Classifier,
    LightGBMClassifier,
)
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


def _apply_feature_subset(
    feature_columns: list[str], feature_subset: list[str]
) -> list[str]:
    """Restrict the classifier's feature columns to `feature_subset`.

    An unknown name is an error rather than a silent no-op: a typo would quietly shrink
    the feature set and show up only as an unexplained drop in identifications.
    """
    if not feature_subset:
        return feature_columns

    if unknown := sorted(set(feature_subset) - set(feature_columns)):
        raise ValueError(
            f"fdr.feature_subset names features the extraction backend does not "
            f"provide: {unknown}"
        )

    return [column for column in feature_columns if column in set(feature_subset)]


def _get_classifier_base(
    config: Config,
    random_state: int | None = None,
) -> Classifier:
    """Creates and returns a classifier base instance.

    Parameters
    ----------
    config : Config
        The workflow configuration, read for the classifier type and its hyperparameters.

    random_state : int | None, optional
        Random state for reproducibility. Default is None.

    Returns
    -------
    Classifier
        The classifier selected by the configuration.
    """
    config_fdr = config["fdr"]
    classifier_name = config_fdr["classifier"]

    if classifier_name == FdrClassifier.MLP:
        return BinaryClassifierLegacyNewBatching(
            test_size=0.001,
            batch_size=5000,
            learning_rate=0.001,
            epochs=10,
            experimental_hyperparameter_tuning=config_fdr[
                "enable_nn_hyperparameter_tuning"
            ],
            random_state=random_state,
        )

    if classifier_name == FdrClassifier.LIGHTGBM:
        return LightGBMClassifier(
            **config_fdr["lightgbm"],
            num_threads=config["general"]["thread_count"],
            random_state=random_state,
        )

    raise ValueError(f"Unknown FDR classifier: {classifier_name}")


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
        self._fdr_manager = FDRManager(
            feature_columns=_apply_feature_subset(
                get_feature_names()
                if self._config["search"]["extraction_backend"] == "rust"
                else feature_columns,
                self._config["fdr"]["feature_subset"],
            ),
            classifier_base=_get_classifier_base(
                self.config,
                random_state=self._random_state_fdr_classifier,
            ),
            dia_cycle=self.dia_data.cycle,
            config=self.config,
            figure_path=self._figure_path,
            feature_matrix_path=self.path
            if self._config["fdr"]["save_feature_matrix"]
            else None,
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
                is_final=True,
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
                    precursor_w_features_df, candidates_df, is_final=True
                )
            )

            precursor_df, fragments_df = extraction_handler.quantify_candidates(
                candidates_fdr_df,
                precursor_fdr_df,
                self.dia_data,
                self.spectral_library,
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

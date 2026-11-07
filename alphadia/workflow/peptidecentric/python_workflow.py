"""Peptide-centric workflow on the Python (numba) extraction backend."""

import pandas as pd

from alphadia.fragcomp.utils import candidate_hash
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import (
    ClassicExtractionHandler,
)
from alphadia.workflow.peptidecentric.multiplexing_requantification_handler import (
    MultiplexingRequantificationHandler,
)
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow
from alphadia.workflow.peptidecentric.utils import (
    add_library_fragment_intensity,
    feature_columns,
    log_precursor_df,
    use_timing_manager,
)


def _get_decoy_strategy(channel_wise_fdr: bool) -> str:
    return "precursor_channel_wise" if channel_wise_fdr else "precursor"


class PythonOptimizationHandler(OptimizationHandler):
    """Optimization on the Python backend: each batch is scored and quantified in one pass, then FDR-filtered."""

    EXTRACTION_HANDLER_CLASS = ClassicExtractionHandler

    def _extract_batch(self) -> pd.DataFrame:
        extraction_handler = self._create_extraction_handler()

        candidates_df = extraction_handler.select_candidates(
            self._dia_data,
            self._optlock.batch_library,
        )

        precursor_quantified_w_features_df, fragments_df = (
            extraction_handler.score_and_quantify_candidates(
                candidates_df,
                self._dia_data,
                self._optlock.batch_library,
            )
        )

        self._optlock.update_with_extraction(
            precursor_quantified_w_features_df, fragments_df
        )

        return self._fdr_manager.fit_predict(
            self._optlock.features_df,
            decoy_strategy=_get_decoy_strategy(self._config["fdr"]["channel_wise_fdr"]),
            competitive=self._config["fdr"]["competitive_scoring"],
            df_fragments=self._optlock.fragments_df,
            version=self._optimization_manager.classifier_version,
        )


class PythonPeptideCentricWorkflow(PeptideCentricWorkflow):
    """Workflow for `search.extraction_backend: python`.

    Scoring and quantification happen in one pass, FDR is applied to the quantified precursors afterwards.
    """

    OPTIMIZATION_HANDLER_CLASS = PythonOptimizationHandler
    EXTRACTION_HANDLER_CLASS = ClassicExtractionHandler

    def _classifier_feature_columns(self) -> list[str]:
        return feature_columns

    @use_timing_manager("extraction")
    def extraction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        extraction_handler = self._create_extraction_handler()

        candidates_df = extraction_handler.select_candidates(
            self.dia_data,
            self.spectral_library,
            apply_cutoff=True,
        )

        precursor_quantified_w_features_df, fragments_df = (
            extraction_handler.score_and_quantify_candidates(
                candidates_df, self.dia_data, self.spectral_library
            )
        )

        self.reporter.log_string(
            f"=== Performing FDR correction with classifier version {self.optimization_manager.classifier_version} ===",
        )

        precursor_df = self._fdr_manager.fit_predict(
            precursor_quantified_w_features_df,
            decoy_strategy=_get_decoy_strategy(self._config["fdr"]["channel_wise_fdr"]),
            competitive=self._config["fdr"]["competitive_scoring"],
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

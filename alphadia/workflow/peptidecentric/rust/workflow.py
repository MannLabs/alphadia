"""Peptide-centric workflow on the Rust (alphadia-search-rs) extraction backend."""

import time

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import DIAData as DiaDataRust
from alphadia_search_rs import FeatureSet

from alphadia.exceptions import GenericUserError
from alphadia.raw_data import DiaData
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow
from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    RustExtractionHandler,
)
from alphadia.workflow.peptidecentric.rust.mapper import (
    dia_data_to_rust,
    get_feature_names,
    set_thread_count,
)
from alphadia.workflow.peptidecentric.rust.optimization_handler import (
    RustOptimizationHandler,
)
from alphadia.workflow.peptidecentric.utils import (
    add_library_fragment_intensity,
    log_precursor_df,
    use_timing_manager,
)


class RustPeptideCentricWorkflow(PeptideCentricWorkflow):
    """Workflow for `search.extraction_backend: rust`.

    Candidates are scored and FDR-filtered first; only the confident ones are quantified.
    """

    OPTIMIZATION_HANDLER_CLASS = RustOptimizationHandler

    def _classifier_feature_columns(self) -> list[str]:
        return get_feature_names(FeatureSet.Full)

    def _create_extraction_handler(
        self, spectral_library: SpecLibFlat
    ) -> RustExtractionHandler:
        return RustExtractionHandler(
            self.config,
            self.optimization_manager.get_optimization_context(),
            self._fdr_manager,
            self.reporter,
            self._create_column_name_handler(),
            self.dia_data,
            spectral_library,
        )

    def _convert_dia_data(self, dia_data: DiaData) -> DiaDataRust:
        if isinstance(dia_data, TimsTOFTranspose):
            raise GenericUserError(
                "NOT_SUPPORTED_BY_RUST",
                "Rust backend does not support TimsTOF data yet. Please use extraction_backend='python'.",
            )

        time_start = time.time()

        # needs to be the first call to alphadia-search-rs
        set_thread_count(self.config["general"]["thread_count"])

        dia_data_rust = dia_data_to_rust(dia_data)

        # TODO: remove these asserts
        assert dia_data.cycle.shape[1] == dia_data_rust.cycle.shape[1]
        assert all(dia_data.rt_values == dia_data_rust.rt_values)

        self.reporter.log_string(
            f"Converting DIA data for alphadia-search-rs took: {time.time() - time_start}"
        )

        return dia_data_rust

    @use_timing_manager("extraction")
    def extraction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        extraction_handler = self._create_extraction_handler(self.spectral_library)

        candidates_df = extraction_handler.select_candidates(apply_cutoff=True)

        precursor_w_features_df = extraction_handler.score_candidates(
            candidates_df, FeatureSet.Full
        )

        candidates_fdr_df, precursor_fdr_df = (
            extraction_handler.perform_fdr_and_filter_candidates(
                precursor_w_features_df,
                candidates_df,
                self.optimization_manager.classifier_version,
                final_round=True,
                train_on_best_candidate=True,
            )
        )

        precursor_df, fragments_df = extraction_handler.quantify_candidates(
            candidates_fdr_df, precursor_fdr_df
        )

        fragments_df = add_library_fragment_intensity(
            fragments_df, self.spectral_library
        )

        log_precursor_df(self.reporter, precursor_df)

        return precursor_df, fragments_df

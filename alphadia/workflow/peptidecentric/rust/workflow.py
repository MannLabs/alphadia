"""Peptide-centric workflow on the Rust (alphadia-search-rs) extraction backend."""

import pandas as pd
from alphadia_search_rs import FeatureSet

from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow
from alphadia.workflow.peptidecentric.utils import (
    add_library_fragment_intensity,
    log_precursor_df,
    use_timing_manager,
)


class RustPeptideCentricWorkflow(PeptideCentricWorkflow):
    """Workflow for `search.extraction_backend: rust`.

    Candidates are scored and FDR-filtered first; only the confident ones are quantified.
    """

    @use_timing_manager("extraction")
    def extraction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        precursor_w_features_df = extraction_handler.score_candidates(
            candidates_df, self.dia_data, self.spectral_library, FeatureSet.Full
        )

        candidates_fdr_df, precursor_fdr_df = (
            extraction_handler.perform_fdr_and_filter_candidates(
                precursor_w_features_df,
                candidates_df,
                self.dia_data,
                self.spectral_library,
                final_round=True,
                train_on_best_candidate=True,
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

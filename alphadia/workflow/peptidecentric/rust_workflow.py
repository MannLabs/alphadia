"""Peptide-centric workflow on the Rust (alphadia-search-rs) extraction backend."""

import time

import pandas as pd
from alphadia_search_rs import FeatureSet

from alphadia.exceptions import GenericUserError
from alphadia.raw_data import DiaData
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.workflow.peptidecentric.extraction_handler import NgExtractionHandler
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    dia_data_to_ng,
    get_feature_names,
    set_ng_thread_count,
)
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow
from alphadia.workflow.peptidecentric.utils import (
    add_library_fragment_intensity,
    log_precursor_df,
    use_timing_manager,
)


class RustOptimizationHandler(OptimizationHandler):
    """Optimization on the Rust backend: each batch is scored and quantified separately, then FDR-filtered."""

    EXTRACTION_HANDLER_CLASS = NgExtractionHandler

    def _extract_batch(self) -> pd.DataFrame:
        extraction_handler = self._create_extraction_handler()

        candidates_df = extraction_handler.select_candidates(
            self._dia_data,
            self._optlock.batch_library,
        )

        precursor_w_features_df = extraction_handler.score_candidates(
            candidates_df,
            self._dia_data,
            self._optlock.batch_library,
            FeatureSet.Calibration,
        )

        precursor_quantified_df, fragments_df = extraction_handler.quantify_candidates(
            candidates_df,
            None,
            self._dia_data,
            self._optlock.batch_library,
        )

        # in order to keep the relation between precursor_quantified_df and fragments_df, we keep the former, but merge the features from scoring
        precursor_quantified_w_features_df = precursor_quantified_df.merge(
            precursor_w_features_df,
            how="left",
            on=["precursor_idx", "rank"],
            suffixes=("", "__y"),
            validate="one_to_one",
        )
        precursor_quantified_w_features_df.drop(
            columns=[
                col
                for col in precursor_quantified_w_features_df.columns
                if col.endswith("__y")
            ],
            inplace=True,
        )

        self._optlock.update_with_extraction(
            precursor_quantified_w_features_df, fragments_df
        )

        _, precursor_df = extraction_handler.perform_fdr_and_filter_candidates(
            self._optlock.features_df,
            candidates_df,
            self._dia_data,
            self._optlock.batch_library,
            df_fragments=self._optlock.fragments_df,
        )

        return precursor_df


class RustPeptideCentricWorkflow(PeptideCentricWorkflow):
    """Workflow for `search.extraction_backend: rust`.

    Candidates are scored and FDR-filtered first; only the confident ones are quantified.
    """

    OPTIMIZATION_HANDLER_CLASS = RustOptimizationHandler
    EXTRACTION_HANDLER_CLASS = NgExtractionHandler

    def _classifier_feature_columns(self) -> list[str]:
        # The FDR manager uses only the columns present, so calibration rounds use the core subset.
        return get_feature_names(FeatureSet.Full)

    def _prepare_dia_data(self, dia_data: DiaData) -> "DiaDataNG":  # noqa: F821
        if isinstance(dia_data, TimsTOFTranspose):
            raise GenericUserError(
                "NOT_SUPPORTED_BY_NG",
                "Rust backend does not support TimsTOF data yet. Please use extraction_backend='python'.",
            )

        time_start = time.time()

        # needs to be the first call to alphadia-search-rs
        set_ng_thread_count(self.config["general"]["thread_count"])

        dia_data_ng = dia_data_to_ng(dia_data)

        # TODO: remove these asserts
        assert dia_data.cycle.shape[1] == dia_data_ng.cycle.shape[1]
        assert all(dia_data.rt_values == dia_data_ng.rt_values)

        self.reporter.log_string(
            f"Creating DIA data NG object took: {time.time() - time_start}"
        )

        return dia_data_ng

    @use_timing_manager("extraction")
    def extraction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        extraction_handler = self._create_extraction_handler()

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

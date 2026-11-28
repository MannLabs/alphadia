"""Peptide-centric workflow on the Rust (alphadia-search-rs) extraction backend."""

import time

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import DIAData as DiaDataRust
from alphadia_search_rs import FeatureSet

from alphadia.exceptions import GenericUserError
from alphadia.raw_data import DiaData
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.workflow.config import Config
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow
from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    RustExtractionHandler,
)
from alphadia.workflow.peptidecentric.rust.mapper import (
    dia_data_to_rust,
    get_classifier_feature_columns,
    get_enabled_feature_groups,
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

    def __init__(
        self,
        instance_name: str,
        config: Config,
        quant_path: str | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(instance_name, config, quant_path, random_state)
        # Whether the raw file has MS1 scans, set by `_convert_dia_data`: the converted alphadia-search-rs DIA data
        # always reports has_ms1 False.
        self._raw_data_has_ms1: bool = False

    def _classifier_feature_columns(self) -> list[str]:
        return get_classifier_feature_columns(
            get_enabled_feature_groups(self.config["search"])
        )

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
            raw_data_has_ms1=self._raw_data_has_ms1,
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
        self._raw_data_has_ms1 = dia_data.has_ms1

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

        # The calibration rounds train on `FeatureSet.Calibration`, the main search on `FeatureSet.Full` without
        # `FINAL_ROUND_EXCLUDED_FEATURES`. The FDR manager stores classifiers per feature columns, so no calibration
        # classifier fits the main search and its FDR starts from the base classifier.
        candidates_fdr_df, precursor_fdr_df = (
            extraction_handler.perform_fdr_and_filter_candidates(
                precursor_w_features_df,
                candidates_df,
                final_round=True,
                train_on_best_candidate=True,
            )
        )

        precursor_df, fragments_df = extraction_handler.quantify_candidates(
            candidates_fdr_df,
            precursor_fdr_df,
            fragment_isotopes=self.config["search"]["quant_fragment_isotopes"],
        )

        fragments_df = add_library_fragment_intensity(
            fragments_df, self.spectral_library
        )

        log_precursor_df(self.reporter, precursor_df)

        return precursor_df, fragments_df

"""Optimization handler of the Rust (alphadia-search-rs) backend."""

import pandas as pd
from alphadia_search_rs import FeatureSet

from alphadia.workflow.peptidecentric.optimization_handler import (
    ExtractedBatch,
    OptimizationHandler,
)
from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    RustExtractionHandler,
)


class RustOptimizationHandler(OptimizationHandler[RustExtractionHandler]):
    """Optimization on the Rust backend: each batch is scored and quantified separately, then FDR-filtered."""

    def _extract_batch(
        self, extraction_handler: RustExtractionHandler
    ) -> ExtractedBatch:
        candidates_df = extraction_handler.select_candidates()

        precursor_w_features_df = extraction_handler.score_candidates(
            candidates_df, FeatureSet.Calibration
        )

        precursor_quantified_df, fragments_df = extraction_handler.quantify_candidates(
            candidates_df, None
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

        return ExtractedBatch(
            candidates_df, precursor_quantified_w_features_df, fragments_df
        )

    def _perform_fdr(
        self, extraction_handler: RustExtractionHandler, batch: ExtractedBatch
    ) -> pd.DataFrame:
        _, precursor_df = extraction_handler.perform_fdr_and_filter_candidates(
            self._optlock.features_df,
            batch.candidates_df,
            self._optimization_manager.classifier_version,
            df_fragments=self._optlock.fragments_df,
            max_qval=self._get_max_qval(),
            max_train_samples=self._config["calibration"][
                "classifier_max_train_samples"
            ],
        )
        return precursor_df

"""Optimization handler of the Python (numba) backend."""

import pandas as pd

from alphadia.workflow.peptidecentric.optimization_handler import (
    ExtractedBatch,
    OptimizationHandler,
)
from alphadia.workflow.peptidecentric.python.extraction_handler import (
    PythonExtractionHandler,
)
from alphadia.workflow.peptidecentric.python.utils import get_decoy_strategy


class PythonOptimizationHandler(OptimizationHandler[PythonExtractionHandler]):
    """Optimization on the Python backend: each batch is scored and quantified in one pass, then FDR-filtered."""

    def _extract_batch(
        self, extraction_handler: PythonExtractionHandler
    ) -> ExtractedBatch:
        candidates_df = extraction_handler.select_candidates(
            self._dia_data,
            self._optlock.batch_library,
        )

        precursors_df, fragments_df = extraction_handler.score_and_quantify_candidates(
            candidates_df,
            self._dia_data,
            self._optlock.batch_library,
        )

        return ExtractedBatch(candidates_df, precursors_df, fragments_df)

    def _perform_fdr(
        self, extraction_handler: PythonExtractionHandler, batch: ExtractedBatch
    ) -> pd.DataFrame:
        return self._fdr_manager.fit_predict(
            self._optlock.features_df,
            decoy_strategy=get_decoy_strategy(self._config),
            competitive=self._config["fdr"]["competitive_scoring"],
            df_fragments=self._optlock.fragments_df,
            version=self._optimization_manager.classifier_version,
        )

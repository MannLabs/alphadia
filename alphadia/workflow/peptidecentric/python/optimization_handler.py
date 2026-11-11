"""Optimization handler of the Python (numba) backend."""

import pandas as pd

from alphadia.fdr.fdr import FDR_HEURISTIC
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
        candidates_df = extraction_handler.select_candidates()

        precursors_df, fragments_df = extraction_handler.score_and_quantify_candidates(
            candidates_df
        )

        return ExtractedBatch(candidates_df, precursors_df, fragments_df)

    def _perform_fdr(
        self, extraction_handler: PythonExtractionHandler, batch: ExtractedBatch
    ) -> pd.DataFrame:
        max_qval = self._get_max_qval()
        return self._fdr_manager.fit_predict(
            self._optlock.features_df,
            decoy_strategy=get_decoy_strategy(self._config),
            competitive=self._config["fdr"]["competitive_scoring"],
            df_fragments=self._optlock.fragments_df,
            version=self._optimization_manager.classifier_version,
            fdr_heuristic=FDR_HEURISTIC
            if max_qval is None
            else max(FDR_HEURISTIC, max_qval),
            max_train_samples=self._config["calibration"][
                "classifier_max_train_samples"
            ],
        )

"""Peptide-centric workflow on the Python (numba) extraction backend."""

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.fragcomp.utils import candidate_hash
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow
from alphadia.workflow.peptidecentric.python.extraction_handler import (
    PythonExtractionHandler,
)
from alphadia.workflow.peptidecentric.python.multiplexing_requantification_handler import (
    MultiplexingRequantificationHandler,
)
from alphadia.workflow.peptidecentric.python.optimization_handler import (
    PythonOptimizationHandler,
)
from alphadia.workflow.peptidecentric.python.utils import (
    feature_columns,
    get_decoy_strategy,
)
from alphadia.workflow.peptidecentric.utils import (
    add_library_fragment_intensity,
    log_precursor_df,
    use_timing_manager,
)


class PythonPeptideCentricWorkflow(PeptideCentricWorkflow):
    """Workflow for `search.extraction_backend: python`.

    Scoring and quantification happen in one pass, FDR is applied to the quantified precursors afterwards.
    """

    OPTIMIZATION_HANDLER_CLASS = PythonOptimizationHandler
    SUPPORTS_MULTIPLEXING = True

    def _classifier_feature_columns(self) -> list[str]:
        return feature_columns

    def _create_extraction_handler(
        self, spectral_library: SpecLibFlat
    ) -> PythonExtractionHandler:
        return PythonExtractionHandler(
            self.config,
            self.optimization_manager.search_tolerances(),
            self._fdr_manager,
            self.reporter,
            self._create_column_name_handler(),
            self.dia_data,
            spectral_library,
        )

    @use_timing_manager("extraction")
    def extraction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        extraction_handler = self._create_extraction_handler(self.spectral_library)

        candidates_df = extraction_handler.select_candidates(apply_cutoff=True)

        precursor_quantified_w_features_df, fragments_df = (
            extraction_handler.score_and_quantify_candidates(candidates_df)
        )

        self.reporter.log_string(
            f"=== Performing FDR correction with classifier version {self.optimization_manager.classifier_version} ===",
        )

        precursor_df = self._fdr_manager.fit_predict(
            precursor_quantified_w_features_df,
            decoy_strategy=get_decoy_strategy(self._config),
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
            self._create_column_name_handler(),
            self.spectral_library,
        )

        psm_df = requantification_handler.requantify(self.dia_data, psm_df)

        psm_df = psm_df[psm_df["qval"] <= self.config["fdr"]["fdr"]]

        log_precursor_df(self.reporter, psm_df)

        return psm_df

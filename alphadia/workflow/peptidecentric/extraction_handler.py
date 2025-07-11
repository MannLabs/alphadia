import pandas as pd
import seaborn as sns
from alphabase.spectral_library.base import SpecLibBase

from alphadia.peakgroup import search
from alphadia.peakgroup.config_df import HybridCandidateConfig
from alphadia.plexscoring.config import CandidateConfig
from alphadia.plexscoring.plexscoring import CandidateScoring
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.managers.raw_file_manager import DiaData
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler


class ExtractionHandler:
    """Manages precursor and fragment extraction operations."""

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
        spectral_library: SpecLibBase,
    ):
        self._config: Config = config
        self._optimization_manager: OptimizationManager = optimization_manager
        self._reporter: Pipeline = reporter

        self._spectral_library: SpecLibBase = spectral_library
        self._column_name_handler: ColumnNameHandler = column_name_handler

    def extract_batch(
        self,
        dia_data: DiaData,
        batch_precursor_df: pd.DataFrame,
        batch_fragment_df: pd.DataFrame,
        apply_cutoff: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._reporter.log_string(
            f"Extracting batch of {len(batch_precursor_df)} precursors",
            verbosity="progress",
        )

        scoring_config = HybridCandidateConfig()
        scoring_config.update(
            {
                **self._config["selection_config"],
                "top_k_fragments": self._config["search"]["top_k_fragments"],
                "rt_tolerance": self._optimization_manager.rt_error,
                "mobility_tolerance": self._optimization_manager.mobility_error,
                "candidate_count": self._optimization_manager.num_candidates,
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
                "exclude_shared_ions": self._config["search"]["exclude_shared_ions"],
                "min_size_rt": self._config["search"]["quant_window"],
            }
        )

        for log_line in [
            "=== Search parameters used ===",
            f"{'rt_tolerance':<15}: {scoring_config.rt_tolerance}",
            f"{'mobility_tolerance':<15}: {scoring_config.mobility_tolerance}",
            f"{'precursor_mz_tolerance':<15}: {scoring_config.precursor_mz_tolerance}",
            f"{'fragment_mz_tolerance':<15}: {scoring_config.fragment_mz_tolerance}",
            "==============================================",
        ]:
            self._reporter.log_string(log_line, verbosity="debug")

        rt_column = self._column_name_handler.get_rt_column()
        mobility_column = self._column_name_handler.get_mobility_column()
        precursor_mz_column = self._column_name_handler.get_precursor_mz_column()
        fragment_mz_column = self._column_name_handler.get_fragment_mz_column()

        extraction = search.HybridCandidateSelection(
            dia_data,
            batch_precursor_df,
            batch_fragment_df,
            scoring_config,
            rt_column=rt_column,
            mobility_column=mobility_column,
            precursor_mz_column=precursor_mz_column,
            fragment_mz_column=fragment_mz_column,
            fwhm_rt=self._optimization_manager.fwhm_rt,
            fwhm_mobility=self._optimization_manager.fwhm_mobility,
        )
        candidates_df = extraction(thread_count=self._config["general"]["thread_count"])

        sns.histplot(candidates_df, x="score", hue="decoy", bins=100)

        if apply_cutoff:
            num_before = len(candidates_df)
            self._reporter.log_string(
                f"Applying score cutoff of {self._optimization_manager.score_cutoff}",
            )
            candidates_df = candidates_df[
                candidates_df["score"] > self._optimization_manager.score_cutoff
            ]
            num_after = len(candidates_df)
            num_removed = num_before - num_after
            self._reporter.log_string(
                f"Removed {num_removed} precursors with score below cutoff",
            )

        candidate_scoring_config = CandidateConfig()
        candidate_scoring_config.update(
            {
                **self._config["scoring_config"],
                "top_k_fragments": self._config["search"]["top_k_fragments"],
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
                "exclude_shared_ions": self._config["search"]["exclude_shared_ions"],
                "quant_window": self._config["search"]["quant_window"],
                "quant_all": self._config["search"]["quant_all"],
                "experimental_xic": self._config["search"]["experimental_xic"],
            }
        )

        candidate_scoring = CandidateScoring(
            dia_data.jitclass(),
            batch_precursor_df,
            batch_fragment_df,
            config=candidate_scoring_config,
            rt_column=rt_column,
            mobility_column=mobility_column,
            precursor_mz_column=precursor_mz_column,
            fragment_mz_column=fragment_mz_column,
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self._config["general"]["thread_count"],
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df

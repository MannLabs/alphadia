import pandas as pd
import seaborn as sns
from alphabase.spectral_library.base import SpecLibBase

from alphadia.peakgroup import search
from alphadia.peakgroup.config_df import HybridCandidateConfig
from alphadia.plexscoring.config import CandidateConfig
from alphadia.plexscoring.plexscoring import CandidateScoring
from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.optimization_manager import OptimizationManager
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
        self._optimization_manager: OptimizationManager = optimization_manager
        self._reporter: Pipeline = reporter

        self._spectral_library: SpecLibBase = spectral_library
        self._column_name_handler: ColumnNameHandler = column_name_handler

        self._thread_count = config["general"]["thread_count"]

        self._selection_config = HybridCandidateConfig()
        self._selection_config.update(
            {
                **config["selection_config"],
                "top_k_fragments": config["search"]["top_k_fragments"],
                "exclude_shared_ions": config["search"]["exclude_shared_ions"],
                "min_size_rt": config["search"]["quant_window"],
            }
        )

        self._scoring_config = CandidateConfig()
        self._scoring_config.update(
            {
                **config["scoring_config"],
                "top_k_fragments": config["search"]["top_k_fragments"],
                "exclude_shared_ions": config["search"]["exclude_shared_ions"],
                "quant_window": config["search"]["quant_window"],
                "quant_all": config["search"]["quant_all"],
                "experimental_xic": config["search"]["experimental_xic"],
            }
        )

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

        self._selection_config.update(
            {
                "rt_tolerance": self._optimization_manager.rt_error,
                "mobility_tolerance": self._optimization_manager.mobility_error,
                "candidate_count": self._optimization_manager.num_candidates,
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
            }
        )

        for log_line in [
            "=== Search parameters used ===",
            f"{'rt_tolerance':<15}: {self._selection_config.rt_tolerance}",
            f"{'mobility_tolerance':<15}: {self._selection_config.mobility_tolerance}",
            f"{'candidate_count':<15}: {self._selection_config.candidate_count}",
            f"{'precursor_mz_tolerance':<15}: {self._selection_config.precursor_mz_tolerance}",
            f"{'fragment_mz_tolerance':<15}: {self._selection_config.fragment_mz_tolerance}",
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
            self._selection_config,
            rt_column=rt_column,
            mobility_column=mobility_column,
            precursor_mz_column=precursor_mz_column,
            fragment_mz_column=fragment_mz_column,
            fwhm_rt=self._optimization_manager.fwhm_rt,
            fwhm_mobility=self._optimization_manager.fwhm_mobility,
        )
        candidates_df = extraction(thread_count=self._thread_count)

        sns.histplot(candidates_df, x="score", hue="decoy", bins=100)

        if apply_cutoff:
            num_before = len(candidates_df)

            candidates_df = candidates_df[
                candidates_df["score"] > self._optimization_manager.score_cutoff
            ]
            num_after = len(candidates_df)
            num_removed = num_before - num_after
            self._reporter.log_string(
                f"Removed {num_removed} precursors with score below cutoff {self._optimization_manager.score_cutoff}",
            )

        self._scoring_config.update(
            {
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
            }
        )

        candidate_scoring = CandidateScoring(
            dia_data,
            batch_precursor_df,
            batch_fragment_df,
            config=self._scoring_config,
            rt_column=rt_column,
            mobility_column=mobility_column,
            precursor_mz_column=precursor_mz_column,
            fragment_mz_column=fragment_mz_column,
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self._thread_count,
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df

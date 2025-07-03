import pandas as pd
import seaborn as sns
from alphabase.spectral_library.base import SpecLibBase

from alphadia.data.alpharaw_wrapper import AlphaRaw
from alphadia.data.bruker import TimsTOFTranspose
from alphadia.peakgroup import search
from alphadia.peakgroup.config_df import HybridCandidateConfig
from alphadia.plexscoring.config import CandidateConfig
from alphadia.plexscoring.plexscoring import CandidateScoring
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.optimization_manager import OptimizationManager


class ExtractionHandler:
    """Manages precursor and fragment extraction operations."""

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        reporter: Pipeline,
        spectral_library: SpecLibBase,
        *,
        rt_column: str,
        mobility_column: str,
        precursor_mz_column: str,
        fragment_mz_column: str,
    ):
        self._config: Config = config
        # TODO think about passing only what is needed, e.g. rt_error, ms1_error, etc.
        self._optimization_manager: OptimizationManager = optimization_manager
        self._reporter: Pipeline = reporter

        self._spectral_library: SpecLibBase = spectral_library
        self._rt_column: str = rt_column
        self._mobility_column: str = mobility_column
        self._precursor_mz_column: str = precursor_mz_column
        self._fragment_mz_column: str = fragment_mz_column

    def extract_batch(
        self,
        dia_data: TimsTOFTranspose | AlphaRaw,
        batch_precursor_df: pd.DataFrame,
        batch_fragment_df: pd.DataFrame,
        apply_cutoff: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._reporter.log_string(
            f"Extracting batch of {len(batch_precursor_df)} precursors",
            verbosity="progress",
        )

        scoring_config = HybridCandidateConfig()
        scoring_config.update(self._config["selection_config"])
        scoring_config.update(
            {
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

        self._reporter.log_string("=== Search parameters used === ", verbosity="debug")
        self._reporter.log_string(
            f"{'rt_tolerance':<15}: {scoring_config.rt_tolerance}", verbosity="debug"
        )
        self._reporter.log_string(
            f"{'mobility_tolerance':<15}: {scoring_config.mobility_tolerance}",
            verbosity="debug",
        )
        self._reporter.log_string(
            f"{'precursor_mz_tolerance':<15}: {scoring_config.precursor_mz_tolerance}",
            verbosity="debug",
        )
        self._reporter.log_string(
            f"{'fragment_mz_tolerance':<15}: {scoring_config.fragment_mz_tolerance}",
            verbosity="debug",
        )
        self._reporter.log_string(
            "==============================================", verbosity="debug"
        )

        extraction = search.HybridCandidateSelection(
            dia_data,
            batch_precursor_df,
            batch_fragment_df,
            scoring_config,
            rt_column=self._rt_column,
            mobility_column=self._mobility_column,
            precursor_mz_column=self._precursor_mz_column,
            fragment_mz_column=self._fragment_mz_column,
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
        candidate_scoring_config.update(self._config["scoring_config"])
        candidate_scoring_config.update(
            {
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
            rt_column=self._rt_column,
            mobility_column=self._mobility_column,
            precursor_mz_column=self._precursor_mz_column,
            fragment_mz_column=self._fragment_mz_column,
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self._config["general"]["thread_count"],
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df

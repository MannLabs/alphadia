import seaborn as sns
from alphabase.spectral_library.base import SpecLibBase

from alphadia.data.utils import DiaData
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
        dia_data: DiaData,
        spectral_library: SpecLibBase,
        rt_column: str,
        mobility_column: str,
        precursor_mz_column: str,
        fragment_mz_column: str,
    ):
        self.config: Config = config
        self.optimization_manager: OptimizationManager = optimization_manager
        self.reporter: Pipeline = reporter
        self.dia_data: DiaData = dia_data
        self.spectral_library: SpecLibBase = spectral_library
        self._rt_column: str = rt_column
        self._mobility_column: str = mobility_column
        self._precursor_mz_column: str = precursor_mz_column
        self._fragment_mz_column: str = fragment_mz_column

    def extract_batch(
        self, batch_precursor_df, batch_fragment_df=None, apply_cutoff=False
    ):
        if batch_fragment_df is None:
            batch_fragment_df = self.spectral_library._fragment_df
        self.reporter.log_string(
            f"Extracting batch of {len(batch_precursor_df)} precursors",
            verbosity="progress",
        )

        config = HybridCandidateConfig()
        config.update(self.config["selection_config"])
        config.update(
            {
                "top_k_fragments": self.config["search"]["top_k_fragments"],
                "rt_tolerance": self.optimization_manager.rt_error,
                "mobility_tolerance": self.optimization_manager.mobility_error,
                "candidate_count": self.optimization_manager.num_candidates,
                "precursor_mz_tolerance": self.optimization_manager.ms1_error,
                "fragment_mz_tolerance": self.optimization_manager.ms2_error,
                "exclude_shared_ions": self.config["search"]["exclude_shared_ions"],
                "min_size_rt": self.config["search"]["quant_window"],
            }
        )

        self.reporter.log_string("=== Search parameters used === ", verbosity="debug")
        self.reporter.log_string(
            f"{'rt_tolerance':<15}: {config.rt_tolerance}", verbosity="debug"
        )
        self.reporter.log_string(
            f"{'mobility_tolerance':<15}: {config.mobility_tolerance}",
            verbosity="debug",
        )
        self.reporter.log_string(
            f"{'precursor_mz_tolerance':<15}: {config.precursor_mz_tolerance}",
            verbosity="debug",
        )
        self.reporter.log_string(
            f"{'fragment_mz_tolerance':<15}: {config.fragment_mz_tolerance}",
            verbosity="debug",
        )
        self.reporter.log_string(
            "==============================================", verbosity="debug"
        )

        extraction = search.HybridCandidateSelection(
            self.dia_data,
            batch_precursor_df,
            batch_fragment_df,
            config,
            rt_column=self._rt_column,
            mobility_column=self._mobility_column,
            precursor_mz_column=self._precursor_mz_column,
            fragment_mz_column=self._fragment_mz_column,
            fwhm_rt=self.optimization_manager.fwhm_rt,
            fwhm_mobility=self.optimization_manager.fwhm_mobility,
        )
        candidates_df = extraction(thread_count=self.config["general"]["thread_count"])

        sns.histplot(candidates_df, x="score", hue="decoy", bins=100)

        if apply_cutoff:
            num_before = len(candidates_df)
            self.reporter.log_string(
                f"Applying score cutoff of {self.optimization_manager.score_cutoff}",
            )
            candidates_df = candidates_df[
                candidates_df["score"] > self.optimization_manager.score_cutoff
            ]
            num_after = len(candidates_df)
            num_removed = num_before - num_after
            self.reporter.log_string(
                f"Removed {num_removed} precursors with score below cutoff",
            )

        config = CandidateConfig()
        config.update(self.config["scoring_config"])
        config.update(
            {
                "top_k_fragments": self.config["search"]["top_k_fragments"],
                "precursor_mz_tolerance": self.optimization_manager.ms1_error,
                "fragment_mz_tolerance": self.optimization_manager.ms2_error,
                "exclude_shared_ions": self.config["search"]["exclude_shared_ions"],
                "quant_window": self.config["search"]["quant_window"],
                "quant_all": self.config["search"]["quant_all"],
                "experimental_xic": self.config["search"]["experimental_xic"],
            }
        )

        candidate_scoring = CandidateScoring(
            self.dia_data.jitclass(),
            batch_precursor_df,
            batch_fragment_df,
            config=config,
            rt_column=self._rt_column(),
            mobility_column=self._mobility_column(),
            precursor_mz_column=self._precursor_mz_column(),
            fragment_mz_column=self._fragment_mz_column(),
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self.config["general"]["thread_count"],
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df

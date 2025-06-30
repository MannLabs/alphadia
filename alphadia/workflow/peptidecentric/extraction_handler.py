import seaborn as sns

from alphadia.peakgroup import search
from alphadia.plexscoring.config import CandidateConfig
from alphadia.plexscoring.plexscoring import CandidateScoring


class ExtractionHandler:
    """Manages precursor and fragment extraction operations."""

    def __init__(self, workflow):
        self._workflow = workflow

    def extract_batch(self, batch_precursor_df, batch_fragment_df, apply_cutoff=False):
        self._workflow.reporter.log_string(
            f"Extracting batch of {len(batch_precursor_df)} precursors",
            verbosity="progress",
        )

        config = search.HybridCandidateConfig()
        config.update(self._workflow.config["selection_config"])
        config.update(
            {
                "top_k_fragments": self._workflow.config["search"]["top_k_fragments"],
                "rt_tolerance": self._workflow.optimization_manager.rt_error,
                "mobility_tolerance": self._workflow.optimization_manager.mobility_error,
                "candidate_count": self._workflow.optimization_manager.num_candidates,
                "precursor_mz_tolerance": self._workflow.optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._workflow.optimization_manager.ms2_error,
                "exclude_shared_ions": self._workflow.config["search"][
                    "exclude_shared_ions"
                ],
                "min_size_rt": self._workflow.config["search"]["quant_window"],
            }
        )

        self._workflow.reporter.log_string(
            "=== Search parameters used === ", verbosity="debug"
        )
        self._workflow.reporter.log_string(
            f"{'rt_tolerance':<15}: {config.rt_tolerance}", verbosity="debug"
        )
        self._workflow.reporter.log_string(
            f"{'mobility_tolerance':<15}: {config.mobility_tolerance}",
            verbosity="debug",
        )
        self._workflow.reporter.log_string(
            f"{'precursor_mz_tolerance':<15}: {config.precursor_mz_tolerance}",
            verbosity="debug",
        )
        self._workflow.reporter.log_string(
            f"{'fragment_mz_tolerance':<15}: {config.fragment_mz_tolerance}",
            verbosity="debug",
        )
        self._workflow.reporter.log_string(
            "==============================================", verbosity="debug"
        )

        extraction = search.HybridCandidateSelection(
            self._workflow.dia_data.jitclass(),
            batch_precursor_df,
            batch_fragment_df,
            config.jitclass(),
            rt_column=self._workflow._get_rt_column(),
            mobility_column=self._workflow._get_mobility_column(),
            precursor_mz_column=self._workflow.get_precursor_mz_column(),
            fragment_mz_column=self._workflow._get_fragment_mz_column(),
            fwhm_rt=self._workflow.optimization_manager.fwhm_rt,
            fwhm_mobility=self._workflow.optimization_manager.fwhm_mobility,
        )
        candidates_df = extraction(
            thread_count=self._workflow.config["general"]["thread_count"]
        )

        sns.histplot(candidates_df, x="score", hue="decoy", bins=100)

        if apply_cutoff:
            num_before = len(candidates_df)
            self._workflow.reporter.log_string(
                f"Applying score cutoff of {self._workflow.optimization_manager.score_cutoff}",
            )
            candidates_df = candidates_df[
                candidates_df["score"]
                > self._workflow.optimization_manager.score_cutoff
            ]
            num_after = len(candidates_df)
            num_removed = num_before - num_after
            self._workflow.reporter.log_string(
                f"Removed {num_removed} precursors with score below cutoff",
            )

        config = CandidateConfig()
        config.update(self._workflow.config["scoring_config"])
        config.update(
            {
                "top_k_fragments": self._workflow.config["search"]["top_k_fragments"],
                "precursor_mz_tolerance": self._workflow.optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._workflow.optimization_manager.ms2_error,
                "exclude_shared_ions": self._workflow.config["search"][
                    "exclude_shared_ions"
                ],
                "quant_window": self._workflow.config["search"]["quant_window"],
                "quant_all": self._workflow.config["search"]["quant_all"],
                "experimental_xic": self._workflow.config["search"]["experimental_xic"],
            }
        )

        candidate_scoring = CandidateScoring(
            self._workflow.dia_data.jitclass(),
            batch_precursor_df,
            batch_fragment_df,
            config=config,
            rt_column=self._workflow._get_rt_column(),
            mobility_column=self._workflow._get_mobility_column(),
            precursor_mz_column=self._workflow._get_precursor_mz_column(),
            fragment_mz_column=self._workflow._get_fragment_mz_column(),
        )

        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self._workflow.config["general"]["thread_count"],
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df

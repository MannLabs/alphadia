"""Handles the requantification of peptide-centric data for multiplexing."""

import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.search.scoring.config import CandidateScoringConfig
from alphadia.search.scoring.scoring import CandidateScoring
from alphadia.search.scoring.utils import (
    candidate_features_to_candidates,
    multiplex_candidates,
)
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import (
    CalibrationGroups,
    CalibrationManager,
)
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler


class MultiplexingRequantificationHandler:
    """
    Handles the requantification of peptide-centric data for multiplexing.
    """

    def __init__(
        self,
        config: Config,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
        spectral_library: SpecLibBase,
    ):
        self._config = config
        self._calibration_manager = calibration_manager
        self._fdr_manager = fdr_manager
        self._reporter = reporter
        self._column_name_handler = column_name_handler
        self._spectral_library = spectral_library

    def requantify(self, dia_data: DiaData, psm_df: pd.DataFrame) -> pd.DataFrame:
        self._calibration_manager.predict(
            self._spectral_library.precursor_df_unfiltered, CalibrationGroups.PRECURSOR
        )
        self._calibration_manager.predict(
            self._spectral_library._fragment_df, CalibrationGroups.FRAGMENT
        )

        reference_candidates = candidate_features_to_candidates(psm_df)

        if "multiplexing" not in self._config:
            raise ValueError("no multiplexing config found")
        self._reporter.log_string(
            f"=== Multiplexing {len(reference_candidates):,} precursors ===",
            verbosity="progress",
        )

        original_channels = psm_df["channel"].unique().tolist()
        self._reporter.log_string(
            f"original channels: {original_channels}", verbosity="progress"
        )

        reference_channel = self._config["multiplexing"]["reference_channel"]
        self._reporter.log_string(
            f"reference channel: {reference_channel}", verbosity="progress"
        )

        target_channels = [
            int(c) for c in self._config["multiplexing"]["target_channels"].split(",")
        ]
        self._reporter.log_string(
            f"target channels: {target_channels}", verbosity="progress"
        )

        decoy_channel = self._config["multiplexing"]["decoy_channel"]
        self._reporter.log_string(
            f"decoy channel: {decoy_channel}", verbosity="progress"
        )

        channels = list(
            set(
                original_channels
                + [reference_channel]
                + target_channels
                + [decoy_channel]
            )
        )
        multiplexed_candidates = multiplex_candidates(
            reference_candidates,
            self._spectral_library.precursor_df_unfiltered,
            channels=channels,
        )

        channel_count_lib = self._spectral_library.precursor_df_unfiltered[
            "channel"
        ].value_counts()
        channel_count_multiplexed = multiplexed_candidates["channel"].value_counts()
        ## log channels with less than 100 precursors
        for channel in channels:
            if channel not in channel_count_lib:
                self._reporter.log_string(
                    f"channel {channel} not found in library", verbosity="warning"
                )
            if channel not in channel_count_multiplexed:
                self._reporter.log_string(
                    f"channel {channel} could not be mapped to existing IDs.",
                    verbosity="warning",
                )

        self._reporter.log_string(
            f"=== Requantifying {len(multiplexed_candidates):,} precursors ===",
            verbosity="progress",
        )

        config = CandidateScoringConfig()
        config.score_grouped = True
        config.exclude_shared_ions = True
        config.reference_channel = self._config["multiplexing"]["reference_channel"]
        config.experimental_xic = self._config["search"]["experimental_xic"]

        multiplexed_scoring = CandidateScoring(
            dia_data=dia_data,
            precursors_flat=self._spectral_library.precursor_df_unfiltered,
            fragments_flat=self._spectral_library.fragment_df,
            config=config,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
        )

        multiplexed_candidates["rank"] = 0

        multiplexed_features, fragments = multiplexed_scoring(multiplexed_candidates)

        psm_df = self._fdr_manager.fit_predict(
            multiplexed_features,
            decoy_strategy="channel",
            competetive=self._config["multiplexing"]["competetive_scoring"],
            decoy_channel=decoy_channel,
        )

        return psm_df

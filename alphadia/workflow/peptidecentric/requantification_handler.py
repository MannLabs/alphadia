import pandas as pd
from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.data.alpharaw_wrapper import AlphaRaw
from alphadia.data.bruker import TimsTOFTranspose
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash
from alphadia.plexscoring.config import CandidateConfig
from alphadia.plexscoring.plexscoring import CandidateScoring
from alphadia.plexscoring.utils import (
    candidate_features_to_candidates,
    multiplex_candidates,
)
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationManager
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler


class RequantificationHandler:
    """
    Handles the requantification of peptide-centric data.
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
        self.config = config
        self.calibration_manager = calibration_manager
        self.fdr_manager = fdr_manager
        self.reporter = reporter
        self._column_name_handler = column_name_handler
        self.spectral_library = spectral_library

    def requantify(
        self, dia_data: TimsTOFTranspose | AlphaRaw, psm_df: pd.DataFrame
    ) -> pd.DataFrame:
        self.calibration_manager.predict(
            self.spectral_library.precursor_df_unfiltered, "precursor"
        )
        self.calibration_manager.predict(self.spectral_library._fragment_df, "fragment")

        reference_candidates = candidate_features_to_candidates(psm_df)

        if "multiplexing" not in self.config:
            raise ValueError("no multiplexing config found")
        self.reporter.log_string(
            f"=== Multiplexing {len(reference_candidates):,} precursors ===",
            verbosity="progress",
        )

        original_channels = psm_df["channel"].unique().tolist()
        self.reporter.log_string(
            f"original channels: {original_channels}", verbosity="progress"
        )

        reference_channel = self.config["multiplexing"]["reference_channel"]
        self.reporter.log_string(
            f"reference channel: {reference_channel}", verbosity="progress"
        )

        target_channels = [
            int(c) for c in self.config["multiplexing"]["target_channels"].split(",")
        ]
        self.reporter.log_string(
            f"target channels: {target_channels}", verbosity="progress"
        )

        decoy_channel = self.config["multiplexing"]["decoy_channel"]
        self.reporter.log_string(
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
            self.spectral_library.precursor_df_unfiltered,
            channels=channels,
        )

        channel_count_lib = self.spectral_library.precursor_df_unfiltered[
            "channel"
        ].value_counts()
        channel_count_multiplexed = multiplexed_candidates["channel"].value_counts()
        ## log channels with less than 100 precursors
        for channel in channels:
            if channel not in channel_count_lib:
                self.reporter.log_string(
                    f"channel {channel} not found in library", verbosity="warning"
                )
            if channel not in channel_count_multiplexed:
                self.reporter.log_string(
                    f"channel {channel} could not be mapped to existing IDs.",
                    verbosity="warning",
                )

        self.reporter.log_string(
            f"=== Requantifying {len(multiplexed_candidates):,} precursors ===",
            verbosity="progress",
        )

        config = CandidateConfig()
        config.score_grouped = True
        config.exclude_shared_ions = True
        config.reference_channel = self.config["multiplexing"]["reference_channel"]
        config.experimental_xic = self.config["search"]["experimental_xic"]

        multiplexed_scoring = CandidateScoring(
            dia_data.jitclass(),
            self.spectral_library.precursor_df_unfiltered,
            self.spectral_library.fragment_df,
            config=config,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
        )

        multiplexed_candidates["rank"] = 0

        multiplexed_features, fragments = multiplexed_scoring(multiplexed_candidates)

        target_channels = [
            int(c) for c in self.config["multiplexing"]["target_channels"].split(",")
        ]
        reference_channel = self.config["multiplexing"]["reference_channel"]

        psm_df = self.fdr_manager.fit_predict(
            multiplexed_features,
            decoy_strategy="channel",
            competetive=self.config["multiplexing"]["competetive_scoring"],
            decoy_channel=decoy_channel,
        )

        return psm_df

    def requantify_fragments(
        self, dia_data: TimsTOFTranspose | AlphaRaw, psm_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Requantify confident precursor identifications for transfer learning.

        Parameters
        ----------

        psm_df: pd.DataFrame
            Dataframe with peptide identifications

        Returns
        -------

        psm_df: pd.DataFrame
            Dataframe with existing peptide identifications but updated frag_start_idx and frag_stop_idx

        frag_df: pd.DataFrame
            Dataframe with fragments in long format
        """

        self.reporter.log_string(
            "=== Transfer learning quantification ===",
            verbosity="progress",
        )

        fragment_types = self.config["transfer_library"]["fragment_types"]
        max_charge = self.config["transfer_library"]["max_charge"]

        self.reporter.log_string(
            f"creating library for charged fragment types: {fragment_types}",
        )

        candidate_speclib_flat, scored_candidates = _build_candidate_speclib_flat(
            psm_df, fragment_types=fragment_types, max_charge=max_charge
        )

        self.reporter.log_string(
            "Calibrating library",
        )

        # calibrate
        self.calibration_manager.predict(
            candidate_speclib_flat.precursor_df, "precursor"
        )
        self.calibration_manager.predict(candidate_speclib_flat.fragment_df, "fragment")

        self.reporter.log_string(
            f"quantifying {len(scored_candidates):,} precursors with {len(candidate_speclib_flat.fragment_df):,} fragments",
        )

        config = CandidateConfig()
        config.update(
            {
                "top_k_fragments": 9999,  # Use all fragments ever expected, needs to be larger than charged_frag_types(8)*max_sequence_len(100?)
                "precursor_mz_tolerance": self.config["search"]["target_ms1_tolerance"],
                "fragment_mz_tolerance": self.config["search"]["target_ms2_tolerance"],
                "experimental_xic": self.config["search"]["experimental_xic"],
            }
        )

        scoring = CandidateScoring(
            dia_data.jitclass(),  # TODO: move jitclass() in
            candidate_speclib_flat.precursor_df,
            candidate_speclib_flat.fragment_df,
            config=config,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
        )

        # we disregard the precursors, as we want to keep the original scoring from the top12 search
        # this works fine as there is no index pointing from the precursors to the fragments
        # only the fragments arre indexed by precursor_idx and rank
        _, frag_df = scoring(scored_candidates)

        # establish mapping
        scored_candidates["_candidate_idx"] = candidate_hash(
            scored_candidates["precursor_idx"].values, scored_candidates["rank"].values
        )
        frag_df["_candidate_idx"] = candidate_hash(
            frag_df["precursor_idx"].values, frag_df["rank"].values
        )
        scored_candidates = add_frag_start_stop_idx(scored_candidates, frag_df)

        return scored_candidates, frag_df


def _build_candidate_speclib_flat(
    psm_df: pd.DataFrame,
    fragment_types: list[str] | None = None,
    max_charge: int = 2,
    optional_columns: list[str] | None = None,
) -> tuple[SpecLibFlat, pd.DataFrame]:
    """Build a candidate spectral library for transfer learning.

    Parameters
    ----------

    psm_df: pd.DataFrame
        Dataframe with peptide identifications

    fragment_types: typing.List[str], optional
        List of fragment types to include in the library, by default ['b','y']

    max_charge: int, optional
        Maximum fragment charge state to consider, by default 2

    optional_columns: typing.List[str], optional
        List of optional columns to include in the library, by default [
            "proba",
            "score",
            "qval",
            "channel",
            "rt_library",
            "mz_library",
            "mobility_library",
            "genes",
            "proteins",
            "decoy",
            "mods",
            "mod_sites",
            "sequence",
            "charge",
            "rt_observed", "mobility_observed", "mz_observed"
        ]

    Returns
    -------
    candidate_speclib_flat: SpecLibFlat
        Candidate spectral library in flat format

    scored_candidates: pd.DataFrame
        Dataframe with scored candidates
    """

    # set default optional columns
    if fragment_types is None:
        fragment_types = ["b", "y"]
    if optional_columns is None:
        optional_columns = [
            "proba",
            "score",
            "qval",
            "channel",
            "rt_library",
            "mz_library",
            "mobility_library",
            "genes",
            "proteins",
            "decoy",
            "mods",
            "mod_sites",
            "sequence",
            "charge",
            "rt_observed",
            "mobility_observed",
            "mz_observed",
        ]

    scored_candidates = candidate_features_to_candidates(
        psm_df, optional_columns=optional_columns
    )

    # create speclib with fragment_types of interest
    candidate_speclib = SpecLibBase()
    candidate_speclib.precursor_df = scored_candidates

    candidate_speclib.charged_frag_types = get_charged_frag_types(
        fragment_types, max_charge
    )

    candidate_speclib.calc_fragment_mz_df()

    candidate_speclib._fragment_intensity_df = candidate_speclib.fragment_mz_df.copy()
    # set all fragment weights to 1 to make sure all are quantified
    candidate_speclib._fragment_intensity_df[candidate_speclib.charged_frag_types] = 1.0

    # create flat speclib
    candidate_speclib_flat = SpecLibFlat()
    candidate_speclib_flat.parse_base_library(candidate_speclib)
    # delete immediately to free memory
    del candidate_speclib

    candidate_speclib_flat.fragment_df.rename(
        columns={"mz": "mz_library"}, inplace=True
    )
    candidate_speclib_flat.fragment_df["cardinality"] = 0
    return candidate_speclib_flat, scored_candidates

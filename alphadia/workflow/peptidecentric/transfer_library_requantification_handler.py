"""Handles the requantification of fragments for transfer library building."""

import pandas as pd
from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import CalibCols
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash
from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.search.scoring.config import CandidateScoringConfig
from alphadia.search.scoring.scoring import CandidateScoring
from alphadia.search.scoring.utils import (
    candidate_features_to_candidates,
)
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import (
    CalibrationGroups,
    CalibrationManager,
)
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler


class TransferLibraryRequantificationHandler:
    """
    Handles the requantification of fragments for transfer library building.
    """

    def __init__(
        self,
        config: Config,
        calibration_manager: CalibrationManager,
        optimization_manager: OptimizationManager,
        column_name_handler: ColumnNameHandler,
        reporter: Pipeline,
    ):
        self._config = config
        self._calibration_manager = calibration_manager
        self._optimization_manager = optimization_manager
        self._column_name_handler = column_name_handler
        self._reporter = reporter

    def requantify(
        self, dia_data: DiaData, psm_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Requantify confident precursor identifications for transfer learning.

        Parameters
        ----------

        dia_data: DiaData
            DIA data object

        psm_df: pd.DataFrame
            Dataframe with peptide identifications

        Returns
        -------

        psm_df: pd.DataFrame
            Dataframe with existing peptide identifications but updated frag_start_idx and frag_stop_idx

        frag_df: pd.DataFrame
            Dataframe with fragments in long format
        """

        self._reporter.log_string(
            "=== Transfer learning quantification ===",
            verbosity="progress",
        )

        fragment_types = self._config["transfer_library"]["fragment_types"]
        max_charge = self._config["transfer_library"]["max_charge"]

        self._reporter.log_string(
            f"creating library for charged fragment types: {fragment_types}",
        )

        candidate_speclib_flat, scored_candidates = _build_candidate_speclib_flat(
            psm_df, fragment_types=fragment_types, max_charge=max_charge
        )

        self._reporter.log_string(
            "Calibrating library",
        )

        # calibrate
        self._calibration_manager.predict(
            candidate_speclib_flat.precursor_df, CalibrationGroups.PRECURSOR
        )
        self._calibration_manager.predict(
            candidate_speclib_flat.fragment_df, CalibrationGroups.FRAGMENT
        )

        self._reporter.log_string(
            f"quantifying {len(scored_candidates):,} precursors with {len(candidate_speclib_flat.fragment_df):,} fragments",
        )

        config = CandidateScoringConfig()
        config.update(
            {
                "top_k_fragments": 9999,  # Use all fragments ever expected, needs to be larger than charged_frag_types(8)*max_sequence_len(100?)
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
                "experimental_xic": self._config["search"]["experimental_xic"],
            }
        )

        scoring = CandidateScoring(
            dia_data=dia_data,
            precursors_flat=candidate_speclib_flat.precursor_df,
            fragments_flat=candidate_speclib_flat.fragment_df,
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
            CalibCols.RT_LIBRARY,
            CalibCols.MZ_LIBRARY,
            CalibCols.MOBILITY_LIBRARY,
            "genes",
            "proteins",
            "decoy",
            "mods",
            "mod_sites",
            "sequence",
            "charge",
            CalibCols.RT_OBSERVED, CalibCols.MOBILITY_OBSERVED, CalibCols.MZ_OBSERVED
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
            CalibCols.RT_LIBRARY,
            CalibCols.MZ_LIBRARY,
            CalibCols.MOBILITY_LIBRARY,
            "genes",
            "proteins",
            "decoy",
            "mods",
            "mod_sites",
            "sequence",
            "charge",
            CalibCols.RT_OBSERVED,
            CalibCols.MOBILITY_OBSERVED,
            CalibCols.MZ_OBSERVED,
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
        columns={"mz": CalibCols.MZ_LIBRARY}, inplace=True
    )
    candidate_speclib_flat.fragment_df["cardinality"] = 0
    return candidate_speclib_flat, scored_candidates

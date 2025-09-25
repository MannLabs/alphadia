import time
from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.search.scoring.config import CandidateScoringConfig
from alphadia.search.scoring.scoring import CandidateScoring

# TODO: these imports could be conditional: CandidateSelectionConfig, CandidateSelection, CandidateScoringConfig, CandidateScoring
from alphadia.search.selection.config_df import CandidateSelectionConfig
from alphadia.search.selection.selection import CandidateSelection

try:
    from alphadia_ng import PeakGroupSelection, SelectionParameters

    from alphadia.workflow.peptidecentric.ng.ng_mapper import (
        parse_candidates,
        speclib_to_ng,
    )

    HAS_ALPHADIA_NG = True
except ImportError:
    HAS_ALPHADIA_NG = False

from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler


class ExtractionHandler(ABC):
    """Base class for managing precursor and fragment extraction operations."""

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
    ):
        """
        Parameters
        ----------
        config : Config
            Workflow configuration
        optimization_manager : OptimizationManager
            Optimization manager with current parameters
        reporter : Pipeline
            Reporter for logging
        column_name_handler : ColumnNameHandler
            Column name handler for data access
        """
        self._config: Config = config
        self._optimization_manager: OptimizationManager = optimization_manager
        self._reporter: Pipeline = reporter

        self._column_name_handler: ColumnNameHandler = column_name_handler

    @staticmethod
    def create_handler(
        config: Config,
        optimization_manager: OptimizationManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
    ) -> "ExtractionHandler":
        """Create an extraction handler based on configuration.

        Parameters
        ----------
        config : Config
            AlphaDIA configuration
        optimization_manager : OptimizationManager
            Optimization manager with current parameters
        reporter : Pipeline
            Reporter for logging
        column_name_handler : ColumnNameHandler
            Column name handler for data access

        Returns
        -------
        ExtractionHandler
            Configured extraction handler

        Raises
        ------
        ValueError
            If extraction_backend is not supported
        """
        backend = config["search"]["extraction_backend"].lower()

        reporter.log_string(f"Using {backend} extraction backend", verbosity="info")
        if backend == "classic":
            return ClassicExtractionHandler(
                config, optimization_manager, reporter, column_name_handler
            )
        elif backend == "ng":
            return NgExtractionHandler(
                config, optimization_manager, reporter, column_name_handler
            )
        # add implementations for other backends here
        else:
            raise ValueError(
                f"Invalid extraction backend '{backend}'. "
                "Supported backends are: 'classic', 'ng'"
            )

    def extract_batch(
        self,
        dia_data: DiaData,
        spectral_library: SpecLibFlat,
        apply_cutoff: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract precursors and fragments from DIA data.

        Parameters
        ----------
        dia_data : DiaData
            DIA data to extract from
        spectral_library : SpecLibFlat
            Spectral library containing precursors and fragments
        apply_cutoff : bool
            Whether to apply score cutoff filtering

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Features dataframe and fragments dataframe
        """
        self._reporter.log_string(
            f"Extracting batch of {len(spectral_library.precursor_df)} precursors",
            verbosity="progress",
        )
        time_start = time.time()  # TODO: could use TimingManager here
        candidates_df = self._select_candidates(dia_data, spectral_library)
        self._reporter.log_string(
            f"Selection took: {time.time() - time_start}"
        )  # TODO: debug?

        sns.histplot(candidates_df, x="score", hue="decoy", bins=100)

        if apply_cutoff:
            candidates_df = self._apply_score_cutoff(candidates_df)

        time_start = time.time()
        features_df, fragments_df = self._score_candidates(
            candidates_df, dia_data, spectral_library
        )
        self._reporter.log_string(
            f"Scoring took: {time.time() - time_start}"
        )  # TODO: debug?

        return features_df, fragments_df

    def _apply_score_cutoff(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Apply score cutoff (taken from optimization_manager) to candidates dataframe.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame containing candidate matches

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with only candidates above score cutoff
        """
        num_before = len(candidates_df)
        candidates_df = candidates_df[
            candidates_df["score"] > self._optimization_manager.score_cutoff
        ]
        num_after = len(candidates_df)
        num_removed = num_before - num_after
        self._reporter.log_string(
            f"Removed {num_removed} precursors with score below cutoff {self._optimization_manager.score_cutoff}",
        )
        return candidates_df

    @abstractmethod
    def _select_candidates(
        self, dia_data: DiaData, spectral_library: SpecLibFlat
    ) -> pd.DataFrame:
        """Select candidates from DIA data based on spectral library.

        Parameters
        ----------
        dia_data : DiaData
            DIA data to extract from
        spectral_library : SpecLibFlat
            Spectral library containing precursors and fragments

        Returns
        -------
        pd.DataFrame
            DataFrame with selected candidates
        """

    @abstractmethod
    def _score_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: DiaData,
        spectral_library: SpecLibFlat,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score candidates.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame with selected candidates
        dia_data : DiaData
            DIA data to extract from
        spectral_library : SpecLibFlat
            Spectral library containing precursors and fragments
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Features dataframe and fragments dataframe
        """


class ClassicExtractionHandler(ExtractionHandler):
    """Extraction handler using CandidateSelection."""

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
    ):
        super().__init__(config, optimization_manager, reporter, column_name_handler)

        # Initialize selection configuration
        self._selection_config = CandidateSelectionConfig()
        self._selection_config.update(
            {
                **config["selection_config"],
                "top_k_fragments": config["search"]["top_k_fragments_selection"],
                "exclude_shared_ions": config["search"]["exclude_shared_ions"],
                "min_size_rt": config["search"]["quant_window"],
            }
        )

        self._scoring_config = CandidateScoringConfig()
        self._scoring_config.update(
            {
                **config["scoring_config"],
                "top_k_fragments": config["search"]["top_k_fragments_scoring"],
                "exclude_shared_ions": config["search"]["exclude_shared_ions"],
                "quant_window": config["search"]["quant_window"],
                "quant_all": config["search"]["quant_all"],
                "experimental_xic": config["search"]["experimental_xic"],
            }
        )

    def _select_candidates(
        self, dia_data: DiaData, spectral_library: SpecLibFlat
    ) -> pd.DataFrame:
        """Select candidates from DIA data using CandidateSelection.

        See superclass documentation for interface details.
        """
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

        extraction = CandidateSelection(
            dia_data,
            spectral_library.precursor_df,
            spectral_library.fragment_df,
            self._selection_config,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
            fwhm_rt=self._optimization_manager.fwhm_rt,
            fwhm_mobility=self._optimization_manager.fwhm_mobility,
        )
        candidates_df = extraction(thread_count=self._config["general"]["thread_count"])

        return candidates_df

    def _score_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: DiaData,
        spectral_library: SpecLibFlat,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score candidates using CandidateScoring.

        See superclass documentation for interface details.
        """
        self._scoring_config.update(
            {
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
            }
        )
        candidate_scoring = CandidateScoring(
            dia_data=dia_data,
            precursors_flat=spectral_library.precursor_df,
            fragments_flat=spectral_library.fragment_df,
            config=self._scoring_config,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
        )
        features_df, fragments_df = candidate_scoring(
            candidates_df,
            thread_count=self._config["general"]["thread_count"],
            include_decoy_fragment_features=True,
        )

        return features_df, fragments_df


class NgExtractionHandler(ClassicExtractionHandler):
    """Extraction handler using AlphaNG backend for candidate selection and scoring."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not HAS_ALPHADIA_NG:
            raise ImportError(
                "AlphaDIA NG backend is not installed. "
                "Please install 'alphadia-ng' to use this extraction handler."
            )

    def _select_candidates(
        self,
        dia_data: tuple[DiaData, "DiaDataNG"],  # noqa: F821
        spectral_library: SpecLibFlat,
    ) -> pd.DataFrame:
        """Select candidates using NG backend.

        See superclass documentation for interface details.
        """
        # TODO this is a hack that needs to go once we don't need the "classic" dia_data object anymore
        dia_data_: DiaData = dia_data[0]
        dia_data_ng: DiaDataNG = dia_data[1]  # noqa: F821

        # TODO needs to be stored
        speclib_ng = speclib_to_ng(
            spectral_library,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
        )

        scoring_params = SelectionParameters()
        scoring_params.update(
            {
                "fwhm_rt": self._optimization_manager.fwhm_rt,
                # 'kernel_size': 20,  # 15?
                "peak_length": self._config["search"]["quant_window"],
                "mass_tolerance": self._optimization_manager.ms2_error,
                "rt_tolerance": self._optimization_manager.rt_error,
                "candidate_count": self._optimization_manager.num_candidates,
            }
        )

        self._reporter.log_string(
            f"Using parameters: fwhm_rt={scoring_params.fwhm_rt}, "
            f"kernel_size={scoring_params.kernel_size}, "
            f"peak_length={scoring_params.peak_length}, "
            f"mass_tolerance={scoring_params.mass_tolerance}, "
            f"rt_tolerance={scoring_params.rt_tolerance}"
        )

        peak_group_scoring = PeakGroupSelection(scoring_params)

        candidates = peak_group_scoring.search(dia_data_ng, speclib_ng)

        return parse_candidates(dia_data_, candidates, spectral_library.precursor_df)

    def _score_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: tuple[DiaData, "DiaDataNG"],  # noqa: F821
        spectral_library: SpecLibFlat,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO this is a hack that needs to go once we don't need the "classic" dia_data object anymore
        return super()._score_candidates(candidates_df, dia_data[0], spectral_library)

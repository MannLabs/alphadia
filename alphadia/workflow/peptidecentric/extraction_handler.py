from abc import ABC, abstractmethod

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import FeatureSet

from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler


class ExtractionHandler(ABC):
    """Base class for managing precursor and fragment extraction operations."""

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        fdr_manager: FDRManager,
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
        fdr_manager: FDRManager
            FDR manager
        reporter : Pipeline
            Reporter for logging
        column_name_handler : ColumnNameHandler
            Column name handler for data access
        """
        self._config: Config = config
        self._optimization_manager: OptimizationManager = optimization_manager
        self._fdr_manager: FDRManager = fdr_manager
        self._reporter: Pipeline = reporter

        self._column_name_handler: ColumnNameHandler = column_name_handler

    @staticmethod
    def create_handler(
        config: Config,
        optimization_manager: OptimizationManager,
        fdr_manager: FDRManager,
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
        fdr_manager: FDRManager
            FDR manager
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
        # imported here because the backend handlers subclass ExtractionHandler
        from alphadia.workflow.peptidecentric.python.extraction_handler import (
            PythonExtractionHandler,
        )
        from alphadia.workflow.peptidecentric.rust.extraction_handler import (
            RustExtractionHandler,
        )

        if backend == "python":
            return PythonExtractionHandler(
                config, optimization_manager, fdr_manager, reporter, column_name_handler
            )
        elif backend == "rust":
            return RustExtractionHandler(
                config, optimization_manager, fdr_manager, reporter, column_name_handler
            )
        # add implementations for other backends here
        else:
            raise ValueError(
                f"Invalid extraction backend '{backend}'. "
                "Supported backends are: 'python', 'rust'"
            )

    def select_candidates(
        self,
        dia_data: "DiaData | DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
        apply_cutoff: bool = False,
    ) -> pd.DataFrame:
        """Select candidates from DIA data based on spectral library and apply cutoff if requested.

        Parameters
        ----------
        dia_data : DiaData | DiaDataNG
            DIA data to extract from. Can be classic or NG format depending on backend.
        spectral_library : SpecLibFlat
            Spectral library containing precursors and fragments
        apply_cutoff : bool
            Whether to apply score cutoff filtering

        Returns
        -------
        pd.DataFrame
            Candidates dataframe
        """
        self._reporter.log_string(
            f"Extracting batch of {len(spectral_library.precursor_df)} precursors",
            verbosity="progress",
        )
        candidates_df = self._select_candidates(dia_data, spectral_library)

        # plt.hist(candidates_df["score"], bins=100)

        if apply_cutoff:
            candidates_df = self._apply_score_cutoff(candidates_df)

        return candidates_df

    @abstractmethod
    def _select_candidates(
        self,
        dia_data: "DiaData | DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
    ) -> pd.DataFrame:
        """Select candidates from DIA data based on spectral library.

        Parameters
        ----------
        dia_data : DiaData | DiaDataNG
            DIA data to extract from. Can be classic or NG format depending on backend.
        spectral_library : SpecLibFlat
            Spectral library containing precursors and fragments

        Returns
        -------
        pd.DataFrame
            DataFrame with selected candidates
        """

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
        # This is filter 1
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

    def _log_parameters(self) -> None:
        """Log current extraction parameters."""
        for log_line in [
            "=== Search parameters used ===",
            f"{'rt_error':<15}: {self._optimization_manager.rt_error}",
            f"{'mobility_error':<15}: {self._optimization_manager.mobility_error}",
            f"{'num_candidates':<15}: {self._optimization_manager.num_candidates}",
            f"{'ms1_error':<15}: {self._optimization_manager.ms1_error}",
            f"{'ms2_error':<15}: {self._optimization_manager.ms2_error}",
            f"{'fwhm_rt':<15}: {self._optimization_manager.fwhm_rt}",
            f"{'quant_window':<15}: {self._config['search']['quant_window']}",
            "==============================================",
        ]:
            self._reporter.log_string(log_line, verbosity="info")

    def score_and_quantify_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: DiaData,
        spectral_library: SpecLibFlat,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score and quantify candidates.

        Only implemented by classic extraction handler.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame with selected candidates
        dia_data : DiaData
            DIA data to extract from.
        spectral_library : SpecLibFlat
            Spectral library containing precursors and fragments
        top_k_fragments : int, optional
            top k fragments to use for scoring (None means default from config)
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            features dataframe and fragments dataframe
        """
        raise NotImplementedError()

    def score_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: "DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
        feature_set: FeatureSet,
    ) -> pd.DataFrame:
        """Score candidates.

        Only implemented by NG extraction handler.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame with selected candidates
        dia_data : DiaDataNG
            DIA data to extract from.
        spectral_library : SpecLibFlat
            Spectral library
        feature_set : FeatureSet
            `FeatureSet.Calibration` for the calibration rounds, `FeatureSet.Full` for the final round
        Returns
        -------
        pd.DataFrame
            precursors dataframe with results from scoring (= features)
        """
        raise NotImplementedError()

    def perform_fdr_and_filter_candidates(
        self,
        features_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        dia_data: "DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
        df_fragments: pd.DataFrame | None = None,
        final_round: bool = False,
        train_on_best_candidate: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform FDR on features and filter candidates accordingly.

        Only implemented by NG extraction handler.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with features (scored candidates)

        candidates_df : pd.DataFrame
            DataFrame with candidates

        dia_data : DiaDataNG
            DIA data, to quantify the fragments that fragment competition needs.

        spectral_library : SpecLibFlat
            Spectral library of the candidates

        df_fragments : pd.DataFrame, optional
            Quantified fragments of all candidates. If not given, the fragments of the
            candidates that enter fragment competition are quantified on demand.

        final_round : bool, default=False
            Whether this is the FDR of the final search, which leaves `FINAL_ROUND_EXCLUDED_FEATURES` out.
        train_on_best_candidate : bool, optional
            Train the classifier only on the most target-like candidate of each precursor.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Filtered candidates dataframe and post-FDR precursor dataframe
        """
        raise NotImplementedError()

    def quantify_candidates(
        self,
        candidates_filtered: pd.DataFrame,
        precursor_fdr_df: pd.DataFrame | None,
        dia_data: "DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        """Quantify candidates.

        Parameters
        ----------
        candidates_filtered : pd.DataFrame
            DataFrame with filtered candidates
        precursor_fdr_df : pd.DataFrame | None
            DataFrame with post-FDR precursor results. If given, FDR-related columns will be merged into the precursor results.
        dia_data : DiaDataNG
            DIA data to extract from.
        spectral_library : SpecLibFlat
            Spectral library
        top_k_fragments : int, optional
            top k fragments to use for quantification (only for classic backend, None means default from config)
        Returns
        -------
        tuple[pd.DataFrame | None, pd.DataFrame]
            precursor dataframe incl. quantification (only NG, None otherwise) and fragments dataframe

        """
        raise NotImplementedError()

    def add_columns_from_library(
        self, features_or_precursor_df: pd.DataFrame, spectral_library: SpecLibFlat
    ) -> pd.DataFrame:
        """Add relevant columns from spectral library to features or precursor dataframe.

        Only implemented by NG extraction handler.

        Parameters
        ----------
        features_or_precursor_df : pd.DataFrame
            DataFrame with features or precursors to add columns to
        spectral_library : SpecLibFlat
            Spectral library

        Returns
        -------
        pd.DataFrame
            DataFrame with added columns
        """
        raise NotImplementedError()

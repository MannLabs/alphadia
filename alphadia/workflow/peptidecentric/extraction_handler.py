from abc import ABC, abstractmethod

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

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

    @abstractmethod
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

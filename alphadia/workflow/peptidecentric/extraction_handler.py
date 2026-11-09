from abc import ABC, abstractmethod

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import SearchTolerances

from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler


class ExtractionHandler(ABC):
    """Base class for managing precursor and fragment extraction operations.

    A handler extracts from one DIA data object and one spectral library, both fixed at construction.
    """

    def __init__(
        self,
        config: Config,
        tolerances: SearchTolerances,
        fdr_manager: FDRManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
        dia_data: "DiaData | DiaDataRust",  # noqa: F821
        spectral_library: SpecLibFlat,
    ):
        """
        Parameters
        ----------
        config : Config
            Workflow configuration
        tolerances : SearchTolerances
            Search tolerances for this extraction
        fdr_manager: FDRManager
            FDR manager
        reporter : Pipeline
            Reporter for logging
        column_name_handler : ColumnNameHandler
            Column name handler for data access
        dia_data : DiaData | DiaDataRust
            DIA data to extract from, in the format of the backend
        spectral_library : SpecLibFlat
            Spectral library containing precursors and fragments
        """
        self._config: Config = config
        self._tolerances: SearchTolerances = tolerances
        self._fdr_manager: FDRManager = fdr_manager
        self._reporter: Pipeline = reporter

        self._column_name_handler: ColumnNameHandler = column_name_handler
        self._dia_data = dia_data
        self._spectral_library: SpecLibFlat = spectral_library

    def select_candidates(self, apply_cutoff: bool = False) -> pd.DataFrame:
        """Select candidates from the DIA data based on the spectral library and apply cutoff if requested.

        Parameters
        ----------
        apply_cutoff : bool
            Whether to apply score cutoff filtering

        Returns
        -------
        pd.DataFrame
            Candidates dataframe
        """
        self._reporter.log_string(
            f"Extracting batch of {len(self._spectral_library.precursor_df)} precursors",
            verbosity="progress",
        )
        candidates_df = self._select_candidates()

        # plt.hist(candidates_df["score"], bins=100)

        if apply_cutoff:
            candidates_df = self._apply_score_cutoff(candidates_df)

        return candidates_df

    @abstractmethod
    def _select_candidates(self) -> pd.DataFrame:
        """Select candidates from the DIA data based on the spectral library.

        Returns
        -------
        pd.DataFrame
            DataFrame with selected candidates
        """

    def _apply_score_cutoff(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the score cutoff of the search tolerances to the candidates dataframe.

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
            candidates_df["score"] > self._tolerances.score_cutoff
        ]
        num_after = len(candidates_df)
        num_removed = num_before - num_after
        self._reporter.log_string(
            f"Removed {num_removed} precursors with score below cutoff {self._tolerances.score_cutoff}",
        )
        return candidates_df

    def _log_parameters(self) -> None:
        """Log current extraction parameters."""
        for log_line in [
            "=== Search parameters used ===",
            f"{'rt_error':<15}: {self._tolerances.rt_error}",
            f"{'mobility_error':<15}: {self._tolerances.mobility_error}",
            f"{'num_candidates':<15}: {self._tolerances.num_candidates}",
            f"{'ms1_error':<15}: {self._tolerances.ms1_error}",
            f"{'ms2_error':<15}: {self._tolerances.ms2_error}",
            f"{'fwhm_rt':<15}: {self._tolerances.fwhm_rt}",
            f"{'quant_window':<15}: {self._config['search']['quant_window']}",
            "==============================================",
        ]:
            self._reporter.log_string(log_line, verbosity="info")

    @abstractmethod
    def quantify_candidates(
        self,
        candidates_filtered: pd.DataFrame,
        precursor_fdr_df: pd.DataFrame | None,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        """Quantify candidates.

        Parameters
        ----------
        candidates_filtered : pd.DataFrame
            DataFrame with filtered candidates
        precursor_fdr_df : pd.DataFrame | None
            DataFrame with post-FDR precursor results. If given, FDR-related columns will be merged into the precursor results.
        top_k_fragments : int, optional
            top k fragments to use for quantification (only for the Python backend, None means default from config)
        Returns
        -------
        tuple[pd.DataFrame | None, pd.DataFrame]
            precursor dataframe incl. quantification (only Rust backend, None otherwise) and fragments dataframe

        """

"""Extraction handler of the Python (numba) backend."""

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.raw_data import DiaData
from alphadia.reporting.reporting import Pipeline
from alphadia.search.scoring.config import CandidateScoringConfig
from alphadia.search.scoring.scoring import CandidateScoring
from alphadia.search.selection.config_df import CandidateSelectionConfig
from alphadia.search.selection.selection import CandidateSelection
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler


class PythonExtractionHandler(ExtractionHandler):
    """Extraction handler using the Python (numba) backend."""

    # These parameters are passed as keywords arguments to the CandidateSelectionConfig/CandidateScoringConfig class.
    # Do not rename here without adapting those classes!
    _base_selection_config = {
        "peak_len_rt": 10.0,
        "sigma_scale_rt": 0.5,
        "peak_len_mobility": 0.01,
        "sigma_scale_mobility": 1.0,
        "top_k_precursors": 3,
        "kernel_size": 30,
        "f_mobility": 1.0,
        "f_rt": 0.99,
        "center_fraction": 0.5,
        "min_size_mobility": 8,
        "min_size_rt": 3,
        "max_size_mobility": 20,
        "max_size_rt": 15,
        "group_channels": False,
        "use_weighted_score": True,
        "join_close_candidates": False,
        "join_close_candidates_scan_threshold": 0.6,
        "join_close_candidates_cycle_threshold": 0.6,
    }

    _base_scoring_config = {
        "score_grouped": False,
        "top_k_isotopes": 3,
        "reference_channel": -1,
        "precursor_mz_tolerance": 10,
        "fragment_mz_tolerance": 15,
    }

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        fdr_manager: FDRManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
        dia_data: DiaData,
        spectral_library: SpecLibFlat,
    ):
        super().__init__(
            config,
            optimization_manager,
            fdr_manager,
            reporter,
            column_name_handler,
            dia_data,
            spectral_library,
        )

        self._selection_config = CandidateSelectionConfig()
        self._selection_config.update(
            {
                **self._base_selection_config,
                "top_k_fragments": config["search"]["top_k_fragments_selection"],
                "exclude_shared_ions": config["search"]["exclude_shared_ions"],
                "min_size_rt": config["search"]["quant_window"],
            }
        )

        self._scoring_config = CandidateScoringConfig()
        self._scoring_config.update(
            {
                **self._base_scoring_config,
                "exclude_shared_ions": config["search"]["exclude_shared_ions"],
                "quant_window": config["search"]["quant_window"],
                "quant_all": config["search"]["quant_all"],
                "experimental_xic": config["search"]["experimental_xic"],
            }
        )

    def _select_candidates(self) -> pd.DataFrame:
        """Select candidates from DIA data using CandidateSelection.

        See superclass documentation for interface details.
        """

        self._log_parameters()

        self._selection_config.update(
            {
                "rt_tolerance": self._optimization_manager.rt_error,
                "mobility_tolerance": self._optimization_manager.mobility_error,
                "candidate_count": self._optimization_manager.num_candidates,
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
            }
        )

        extraction = CandidateSelection(
            self._dia_data,
            self._spectral_library.precursor_df,
            self._spectral_library.fragment_df,
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

    def score_and_quantify_candidates(
        self,
        candidates_df: pd.DataFrame,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score and quantify candidates.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame with selected candidates
        top_k_fragments : int, optional
            top k fragments to use for scoring (None means default from config)
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            features dataframe and fragments dataframe
        """
        self._scoring_config.update(
            {
                "precursor_mz_tolerance": self._optimization_manager.ms1_error,
                "fragment_mz_tolerance": self._optimization_manager.ms2_error,
                "top_k_fragments": top_k_fragments
                if top_k_fragments is not None
                else self._config["search"]["top_k_fragments_scoring"],
            }
        )

        candidate_scoring = CandidateScoring(
            dia_data=self._dia_data,
            precursors_flat=self._spectral_library.precursor_df,
            fragments_flat=self._spectral_library.fragment_df,
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

    def quantify_candidates(
        self,
        candidates_df: pd.DataFrame,
        precursor_fdr_df: pd.DataFrame | None,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        """Quantify candidates using classic backend.

        Note: because quantification and scoring are intertwined in the classic backend,
        this method performs both scoring and quantification, but only returns the fragments dataframe.

        """
        del precursor_fdr_df

        features_df_, fragments_df = self.score_and_quantify_candidates(
            candidates_df, top_k_fragments
        )
        return None, fragments_df

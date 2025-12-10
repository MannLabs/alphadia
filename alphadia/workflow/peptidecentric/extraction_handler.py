from abc import ABC, abstractmethod

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import (
    PeakGroupQuantification,
    PeakGroupScoring,
    PeakGroupSelection,
    QuantificationParameters,
    ScoringParameters,
    SelectionParameters,
)

from alphadia.constants.keys import CalibCols
from alphadia.fragcomp.utils import candidate_hash
from alphadia.raw_data import DiaData
from alphadia.raw_data.alpharaw_wrapper import DEFAULT_VALUE_NO_MOBILITY
from alphadia.reporting.reporting import Pipeline
from alphadia.search.scoring.config import CandidateScoringConfig
from alphadia.search.scoring.scoring import CandidateScoring

# TODO: these imports could be conditional: CandidateSelectionConfig, CandidateSelection, CandidateScoringConfig, CandidateScoring
from alphadia.search.selection.config_df import CandidateSelectionConfig
from alphadia.search.selection.selection import CandidateSelection
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    candidates_to_ng,
    parse_candidates,
    parse_quantification,
    speclib_to_ng,
    to_features_df,
)


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
        if backend == "python":
            return ClassicExtractionHandler(
                config, optimization_manager, fdr_manager, reporter, column_name_handler
            )
        elif backend == "rust":
            return NgExtractionHandler(
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

        # sns.histplot(candidates_df, x="score", hue="decoy", bins=100)

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
            f"{'num_candidates':<15}: { self._optimization_manager.num_candidates}",
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
        Returns
        -------
        pd.DataFrame
            precursors dataframe with results from scoring (= features)
        """
        raise NotImplementedError()

    def perform_fdr_and_filter_candidates(
        self, features_df: pd.DataFrame, candidates_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform FDR on features and filter candidates accordingly.

        Only implemented by NG extraction handler.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with features (scored candidates)

        candidates_df : pd.DataFrame
            DataFrame with candidates

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


class ClassicExtractionHandler(ExtractionHandler):
    """Extraction handler using classic backend."""

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
    ):
        super().__init__(
            config, optimization_manager, fdr_manager, reporter, column_name_handler
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

    def _select_candidates(
        self,
        dia_data: DiaData,  # noqa: F821
        spectral_library: SpecLibFlat,
    ) -> pd.DataFrame:
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

    def score_and_quantify_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: DiaData,  # noqa: F821
        spectral_library: SpecLibFlat,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score and quantify candidates using CandidateScoring.

        See superclass documentation for interface details.
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

    def quantify_candidates(
        self,
        candidates_df: pd.DataFrame,
        precursor_fdr_df: pd.DataFrame | None,
        dia_data: DiaData,  # noqa: F821
        spectral_library: SpecLibFlat,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        """Quantify candidates using classic backend.

        Note: because quantification and scoring are intertwined in the classic backend,
        this method performs both scoring and quantification, but only returns the fragments dataframe.

        """
        del precursor_fdr_df

        features_df_, fragments_df = self.score_and_quantify_candidates(
            candidates_df, dia_data, spectral_library, top_k_fragments
        )
        return None, fragments_df


class NgExtractionHandler(ExtractionHandler):
    """Extraction handler using AlphaNG backend for candidate selection and scoring."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize NG extraction handler."""
        super().__init__(*args, **kwargs)

        self._speclib_ng: SpecLibFlatNG = None  # noqa: F821

    def _lazy_init_speclib_ng(self, spectral_library: SpecLibFlat) -> None:
        """Initialize the NG speclib if not already done."""
        if self._speclib_ng is None:
            self._speclib_ng = speclib_to_ng(
                spectral_library,
                rt_column=self._column_name_handler.get_rt_column(),
                precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
                fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
            )

    def _select_candidates(
        self,
        dia_data: "DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
    ) -> pd.DataFrame:
        """Select candidates using NG backend.

        See superclass documentation for interface details.
        """

        self._lazy_init_speclib_ng(spectral_library)

        self._log_parameters()

        selection_params = SelectionParameters()
        selection_params.update(
            {
                "fwhm_rt": self._optimization_manager.fwhm_rt,
                # 'kernel_size': 20,  # 15?
                "top_k_fragments": self._config["search"]["top_k_fragments_selection"],
                "peak_length": self._config["search"]["quant_window"],
                "mass_tolerance": self._optimization_manager.ms2_error,
                "rt_tolerance": self._optimization_manager.rt_error,
                "candidate_count": self._optimization_manager.num_candidates,
            }
        )

        candidates = PeakGroupSelection(selection_params).search(
            dia_data, self._speclib_ng
        )

        cands = parse_candidates(candidates, spectral_library, dia_data)

        return cands

    def score_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: "DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
    ) -> pd.DataFrame:
        """Score candidates using NG backend.

        See superclass documentation for interface details.
        """
        self._lazy_init_speclib_ng(spectral_library)

        candidates = candidates_to_ng(candidates_df, dia_data)

        scoring_params = ScoringParameters()
        scoring_params.update(
            {
                "top_k_fragments": self._config["search"]["top_k_fragments_scoring"],
                "mass_tolerance": self._optimization_manager.ms2_error,
            }
        )

        candidate_features = PeakGroupScoring(scoring_params).score(
            dia_data, self._speclib_ng, candidates
        )

        features_df = to_features_df(candidate_features, spectral_library)

        return features_df

    def quantify_candidates(
        self,
        candidates_df: pd.DataFrame,
        precursor_fdr_df: pd.DataFrame | None,
        dia_data: "DiaDataNG",  # noqa: F821
        spectral_library: SpecLibFlat,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Quantify candidates using NG backend.

        See superclass documentation for interface details.
        """
        self._lazy_init_speclib_ng(spectral_library)

        candidates_collection = candidates_to_ng(candidates_df, dia_data)

        # run quantification
        quant_params = QuantificationParameters()
        if top_k_fragments is not None:
            quant_params.update({"top_k_fragments": top_k_fragments})

        peak_group_quantification = PeakGroupQuantification(quant_params)
        quantified_lib = peak_group_quantification.quantify(
            dia_data, self._speclib_ng, candidates_collection
        )
        precursor_df, fragments_df = parse_quantification(quantified_lib)

        # merge in missing columns
        precursor_df = CandidateScoring.merge_candidate_data(
            precursor_df, candidates_df
        )

        precursor_df = self.add_columns_from_library(precursor_df, spectral_library)

        if precursor_fdr_df is not None:
            precursor_df = precursor_df.merge(
                precursor_fdr_df[["precursor_idx", "rank", "qval", "proba"]],
                on=["precursor_idx", "rank"],
                how="left",
            )

        return precursor_df, fragments_df

    def add_columns_from_library(
        self, features_or_precursor_df: pd.DataFrame, spectral_library: SpecLibFlat
    ) -> pd.DataFrame:
        """Add relevant columns from spectral library to features or precursor dataframe.

        See superclass documentation for interface details.
        """
        features_or_precursor_df = CandidateScoring.merge_precursor_data(
            features_or_precursor_df,
            spectral_library.precursor_df,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
        )

        # TODO: get this from the ng backend
        features_or_precursor_df["cycle_fwhm"] = 3  # TODO: remove
        features_or_precursor_df[CalibCols.MZ_OBSERVED] = features_or_precursor_df[
            CalibCols.MZ_LIBRARY
        ]  # required for transfer library building

        # dummy values required to satisfy some downstream calculations
        features_or_precursor_df["mobility_fwhm"] = -1
        features_or_precursor_df[CalibCols.MOBILITY_OBSERVED] = (
            DEFAULT_VALUE_NO_MOBILITY
        )

        return features_or_precursor_df

    def perform_fdr_and_filter_candidates(
        self,
        features_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform FDR on features and filter candidates accordingly.

        See superclass documentation for interface details.
        """

        features_df["_candidate_idx"] = candidate_hash(
            features_df["precursor_idx"].values, features_df["rank"].values
        )
        candidates_df["_candidate_idx"] = candidate_hash(
            candidates_df["precursor_idx"].values, candidates_df["rank"].values
        )

        # apply FDR to PSMs
        precursor_fdr_df = self._fdr_manager.fit_predict(
            features_df,
            decoy_strategy="precursor",  # TODO support channel_wise, raise error for now
            competitive=self._config["fdr"]["competitive_scoring"],
            df_fragments=None,  # TODO: support fragments_df,
            version=self._optimization_manager.classifier_version,
        )
        precursor_fdr_df = precursor_fdr_df[
            precursor_fdr_df["qval"] <= self._config["fdr"]["fdr"]
        ]

        # filter2: candidates by precursors
        candidates_filtered = candidates_df[
            candidates_df["_candidate_idx"].isin(precursor_fdr_df["_candidate_idx"])
        ].copy()
        del candidates_filtered["_candidate_idx"]

        return candidates_filtered, precursor_fdr_df

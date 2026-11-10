"""Extraction handler of the Rust (alphadia-search-rs) backend."""

import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import DIAData as DiaDataRust
from alphadia_search_rs import (
    FeatureSet,
    OptimizationContext,
    PeakGroupQuantification,
    PeakGroupScoring,
    PeakGroupSelection,
    QuantificationParameters,
    ScoringParameters,
    SelectionParameters,
)
from alphadia_search_rs import SpecLibFlat as SpecLibFlatRust

from alphadia.constants.keys import CalibCols, MbrCols
from alphadia.fragcomp.utils import candidate_hash
from alphadia.raw_data.alpharaw_wrapper import DEFAULT_VALUE_NO_MOBILITY
from alphadia.reporting.reporting import Pipeline
from alphadia.search.python.scoring.scoring import CandidateScoring
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.peptidecentric.column_name_handler import ColumnNameHandler
from alphadia.workflow.peptidecentric.extraction_handler import ExtractionHandler
from alphadia.workflow.peptidecentric.rust.mapper import (
    FINAL_ROUND_EXCLUDED_FEATURES,
    candidates_to_rust,
    parse_candidates,
    parse_quantification,
    speclib_to_rust,
    to_features_df,
)


class RustExtractionHandler(ExtractionHandler):
    """Extraction handler using the Rust (alphadia-search-rs) backend."""

    def __init__(
        self,
        config: Config,
        optimization_context: OptimizationContext,
        fdr_manager: FDRManager,
        reporter: Pipeline,
        column_name_handler: ColumnNameHandler,
        dia_data: DiaDataRust,
        spectral_library: SpecLibFlat,
    ):
        super().__init__(
            config,
            optimization_context,
            fdr_manager,
            reporter,
            column_name_handler,
            dia_data,
            spectral_library,
        )

        self._speclib_rust: SpecLibFlatRust = speclib_to_rust(
            spectral_library,
            rt_column=column_name_handler.get_rt_column(),
            precursor_mz_column=column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=column_name_handler.get_fragment_mz_column(),
        )

    def _select_candidates(self) -> pd.DataFrame:
        """Select candidates using the Rust backend.

        See superclass documentation for interface details.
        """
        self._log_parameters()

        selection_params = SelectionParameters()
        selection_params.update(
            {
                "fwhm_rt": self._optimization_context.fwhm_rt,
                # 'kernel_size': 20,  # 15?
                "top_k_fragments": self._config["search"]["top_k_fragments_selection"],
                "peak_length": self._config["search"]["quant_window"],
                "mass_tolerance": self._optimization_context.ms2_error,
                "rt_tolerance": self._optimization_context.rt_error,
                "candidate_count": self._optimization_context.num_candidates,
            }
        )

        candidates = PeakGroupSelection(selection_params).search(
            self._dia_data, self._speclib_rust
        )

        return parse_candidates(candidates, self._spectral_library, self._dia_data)

    def score_candidates(
        self,
        candidates_df: pd.DataFrame,
        feature_set: FeatureSet,
    ) -> pd.DataFrame:
        """Score candidates.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame with selected candidates
        feature_set : FeatureSet
            `FeatureSet.Calibration` for the calibration rounds, `FeatureSet.Full` for the final round
        Returns
        -------
        pd.DataFrame
            precursors dataframe with results from scoring (= features)
        """
        candidates = candidates_to_rust(candidates_df, self._dia_data)

        scoring_params = ScoringParameters()
        scoring_params.update(
            {
                "top_k_fragments": self._config["search"]["top_k_fragments_scoring"],
                "mass_tolerance": self._optimization_context.ms2_error,
                "feature_set": feature_set,
            }
        )

        candidate_features = PeakGroupScoring(scoring_params).score(
            self._dia_data, self._speclib_rust, candidates
        )

        return to_features_df(candidate_features, self._spectral_library)

    def quantify_candidates(
        self,
        candidates_df: pd.DataFrame,
        precursor_fdr_df: pd.DataFrame | None,
        top_k_fragments: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Quantify candidates using the Rust backend.

        See superclass documentation for interface details.
        """
        candidates_collection = candidates_to_rust(candidates_df, self._dia_data)

        # run quantification
        quant_params = QuantificationParameters()
        if top_k_fragments is not None:
            quant_params.update({"top_k_fragments": top_k_fragments})

        peak_group_quantification = PeakGroupQuantification(quant_params)
        quantified_lib = peak_group_quantification.quantify(
            self._dia_data, self._speclib_rust, candidates_collection
        )
        precursor_df, fragments_df = parse_quantification(quantified_lib)

        # merge in missing columns
        precursor_df = CandidateScoring.merge_candidate_data(
            precursor_df, candidates_df
        )

        precursor_df = self.add_columns_from_library(precursor_df)

        if precursor_fdr_df is not None:
            # fwhm_rt is a scoring feature (not produced by quantification), so it is
            # carried over from the post-FDR precursors into the quantified output.
            precursor_df = precursor_df.merge(
                precursor_fdr_df[["precursor_idx", "rank", "qval", "proba", "fwhm_rt"]],
                on=["precursor_idx", "rank"],
                how="left",
            )

        return precursor_df, fragments_df

    def add_columns_from_library(
        self, features_or_precursor_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add relevant columns from spectral library to features or precursor dataframe.

        Parameters
        ----------
        features_or_precursor_df : pd.DataFrame
            DataFrame with features or precursors to add columns to

        Returns
        -------
        pd.DataFrame
            DataFrame with added columns
        """
        features_or_precursor_df = CandidateScoring.merge_precursor_data(
            features_or_precursor_df,
            self._spectral_library.precursor_df,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
        )

        features_or_precursor_df[CalibCols.MZ_OBSERVED] = features_or_precursor_df[
            CalibCols.MZ_LIBRARY
        ]  # required for transfer library building

        # dummy values required to satisfy some downstream calculations
        features_or_precursor_df["fwhm_mobility"] = -1
        features_or_precursor_df[CalibCols.MOBILITY_OBSERVED] = (
            DEFAULT_VALUE_NO_MOBILITY
        )

        return features_or_precursor_df

    def perform_fdr_and_filter_candidates(
        self,
        features_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        classifier_version: int,
        df_fragments: pd.DataFrame | None = None,
        final_round: bool = False,
        train_on_best_candidate: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform FDR on features and filter candidates accordingly.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with features (scored candidates)

        candidates_df : pd.DataFrame
            DataFrame with candidates

        classifier_version : int
            Version of the stored FDR classifier to start from. `FDRManager.fit_predict` defines the values.

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

        features_df["_candidate_idx"] = candidate_hash(
            features_df["precursor_idx"].values, features_df["rank"].values
        )
        candidates_df["_candidate_idx"] = candidate_hash(
            candidates_df["precursor_idx"].values, candidates_df["rank"].values
        )

        def quantified_fragments(psm_df: pd.DataFrame) -> pd.DataFrame:
            # Fragment competition needs observed fragments, which scoring does not return.
            # Quantifying only the PSMs that enter competition avoids quantifying every candidate.
            competing = candidates_df[
                candidates_df["_candidate_idx"].isin(psm_df["_candidate_idx"])
            ].drop(columns="_candidate_idx")
            _, fragments_df = self.quantify_candidates(competing, None)
            return fragments_df

        library_precursors = self._spectral_library.precursor_df
        if MbrCols.TRANSFER in library_precursors.columns:
            features_df[MbrCols.TRANSFER] = (
                features_df["precursor_idx"]
                .map(library_precursors.set_index("precursor_idx")[MbrCols.TRANSFER])
                .to_numpy()
            )

        # apply FDR to PSMs
        precursor_fdr_df = self._fdr_manager.fit_predict(
            features_df,
            decoy_strategy="precursor",  # TODO support channel_wise, raise error for now
            competitive=self._config["fdr"]["competitive_scoring"],
            df_fragments=df_fragments,
            fragment_provider=quantified_fragments if df_fragments is None else None,
            version=classifier_version,
            excluded_columns=FINAL_ROUND_EXCLUDED_FEATURES if final_round else (),
            train_on_best_candidate=train_on_best_candidate,
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

"""Extraction handler of the Rust (alphadia-search-rs) backend."""

import numpy as np
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

from alphadia.constants.keys import CalibCols
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
    MS1_ISOTOPE_COLUMNS,
    candidates_to_rust,
    parse_candidates,
    parse_quantification,
    set_ms1_isotopes,
    speclib_to_rust,
    to_features_df,
)
from alphadia.workflow.peptidecentric.rust.ms1_calibration import (
    Ms1Calibration,
    fit_ms1_calibration,
)

# alphadia-search-rs never calibrates the precursor m/z, so before the MS1 features are scored (search.ms1_features,
# final round only), a pre-pass extracts the MS1 of the best-scoring target candidates at a wide tolerance and fits the
# MS1 mass calibration on them (ms1_calibration.py).
MS1_PREPASS_CANDIDATES = 10_000
MS1_PREPASS_TOLERANCE_PPM = 15.0


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
        # MS1 extraction tolerance (ppm) from the MS1 calibration pre-pass; None until the MS1 isotopes are set
        self._ms1_tolerance_ppm: float | None = None
        # whether the final round can score the MS1 features, see `_can_score_ms1`; None until the final round asks
        self._ms1_scorable: bool | None = None

    def _can_score_ms1(self) -> bool:
        """Return whether the MS1 features can be scored: the raw file has MS1 scans and the library the isotope
        abundances. Otherwise log why search.ms1_features is skipped."""
        if not self._column_name_handler.dia_data_has_ms1:
            reason = "the raw file has no MS1 scans"
        elif MS1_ISOTOPE_COLUMNS[0] not in self._spectral_library.precursor_df.columns:
            reason = (
                f"the library has no isotope abundances ({MS1_ISOTOPE_COLUMNS[0]}..)"
            )
        else:
            return True
        self._reporter.log_string(
            f"search.ms1_features is skipped: {reason}.", verbosity="warning"
        )
        return False

    def _select_candidates(self, min_score: float) -> pd.DataFrame:
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
                "min_fragments": self._config["search"]["min_fragments_selection"],
                "min_score": min_score,
                "peak_length": self._config["search"]["quant_window"],
                "mass_tolerance": self._optimization_context.ms2_error,
                "rt_tolerance": self._optimization_context.rt_error,
                "candidate_count": self._optimization_context.num_candidates,
                "rt_prior_scale": self._optimization_context.rt_prior_scale,
                "rt_prior_df": self._optimization_context.rt_prior_df,
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
        # The MS1 and detection features are in the full feature set only, i.e. in the final round: the calibration
        # rounds set the score cutoff from their identifications, and a strong new feature there would lower the
        # cutoff of the final extraction.
        final_round = feature_set == FeatureSet.Full
        score_ms1 = final_round and self._config["search"]["ms1_features"]
        if score_ms1:
            if self._ms1_scorable is None:
                self._ms1_scorable = self._can_score_ms1()
            score_ms1 = self._ms1_scorable
        score_detection = final_round and self._config["search"]["detection_features"]
        if score_ms1 and self._ms1_tolerance_ppm is None:
            self._ms1_tolerance_ppm = self._prepare_ms1(candidates_df)

        candidates = candidates_to_rust(candidates_df, self._dia_data)

        candidate_features = PeakGroupScoring(
            self._scoring_parameters(
                feature_set,
                ms1_tolerance_ppm=self._ms1_tolerance_ppm if score_ms1 else None,
                detection_features=score_detection,
            )
        ).score(self._dia_data, self._speclib_rust, candidates)

        return to_features_df(
            candidate_features,
            self._spectral_library,
            ms1_scored=score_ms1,
            detection_scored=score_detection,
        )

    def _scoring_parameters(
        self,
        feature_set: FeatureSet,
        *,
        ms1_tolerance_ppm: float | None = None,
        detection_features: bool = False,
    ) -> ScoringParameters:
        params = {
            "top_k_fragments": self._config["search"]["top_k_fragments_scoring"],
            "mass_tolerance": self._optimization_context.ms2_error,
            "feature_set": feature_set,
        }
        if ms1_tolerance_ppm is not None:
            params["ms1_mass_tolerance"] = ms1_tolerance_ppm
        if detection_features:
            params["detection_features"] = True
        scoring_params = ScoringParameters()
        scoring_params.update(params)
        return scoring_params

    def _prepare_ms1(self, candidates_df: pd.DataFrame) -> float:
        """Set the MS1 isotopes at the calibrated MS1 m/z and return the MS1 extraction tolerance (ppm).

        The pre-pass extracts the MS1 of the `MS1_PREPASS_CANDIDATES` best rank-0 target candidates at
        `MS1_PREPASS_TOLERANCE_PPM` around the uncalibrated m/z, and the calibration is fitted on them. Targets only,
        so the calibration carries no decoy information.
        """
        precursor_mz_column = self._column_name_handler.get_precursor_mz_column()
        rt_column = self._column_name_handler.get_rt_column()
        precursor_df = self._spectral_library.precursor_df
        set_ms1_isotopes(
            self._speclib_rust,
            precursor_df,
            precursor_mz_column,
            rt_column,
            Ms1Calibration.identity(),
        )

        best = candidates_df[
            (candidates_df["decoy"] == 0) & (candidates_df["rank"] == 0)
        ].nlargest(MS1_PREPASS_CANDIDATES, "score")
        prepass_params = self._scoring_parameters(
            FeatureSet.Calibration, ms1_tolerance_ppm=MS1_PREPASS_TOLERANCE_PPM
        )
        prepass = PeakGroupScoring(prepass_params).score_ms1_calibration(
            self._dia_data,
            self._speclib_rust,
            candidates_to_rust(best, self._dia_data),
        )
        library = precursor_df.set_index("precursor_idx")
        prepass_idx = prepass["precursor_idx"]
        calibration = fit_ms1_calibration(
            library[precursor_mz_column].reindex(prepass_idx).to_numpy(np.float64),
            library[rt_column].reindex(prepass_idx).to_numpy(np.float64),
            prepass["ms1_ppm_signed"],
            prepass["ms1_n_isotopes"],
            prepass["ms1_log_m0_apex"],
            prepass["ms1_iso_cosine"],
        )
        self._reporter.log_string(
            f"MS1 calibration from the pre-pass on the {len(best)} best target candidates "
            f"(precursor m/z column {precursor_mz_column}, RT column {rt_column}): {calibration.describe()}",
            verbosity="info",
        )
        set_ms1_isotopes(
            self._speclib_rust,
            precursor_df,
            precursor_mz_column,
            rt_column,
            calibration,
        )
        return calibration.tolerance_ppm

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

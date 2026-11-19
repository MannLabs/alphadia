"""Unit tests for the Rust extraction handler."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from alphadia_search_rs import FeatureSet, OptimizationContext

from alphadia.calibration.ms1_calibration import (
    FALLBACK_TOLERANCE_PPM,
    MS1_PREPASS_TOLERANCE_PPM,
)
from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    RustExtractionHandler,
)
from alphadia.workflow.peptidecentric.rust.mapper import (
    FINAL_ROUND_EXCLUDED_FEATURES,
    MS1_ISOTOPE_COLUMNS,
)

SPECLIB_TO_RUST = (
    "alphadia.workflow.peptidecentric.rust.extraction_handler.speclib_to_rust"
)

OPTIMIZATION_CONTEXT = OptimizationContext(
    rt_error=120.0,
    ms1_error=30.0,
    ms2_error=7.5,
    mobility_error=0.1,
    fwhm_rt=5.0,
    fwhm_mobility=-1.0,
    num_candidates=3,
    score_cutoff=0.0,
)


def _create_handler(config, fdr_manager=None):
    """Create a handler on mock DIA data and a mock spectral library, with the library conversion patched."""
    with patch(SPECLIB_TO_RUST, return_value=MagicMock(name="speclib_rust")):
        return RustExtractionHandler(
            config,
            OPTIMIZATION_CONTEXT,
            fdr_manager or MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(name="dia_data"),
            MagicMock(name="spectral_library"),
        )


@patch(SPECLIB_TO_RUST)
def test_converts_the_library_once_at_construction(mock_speclib_to_rust):
    # Given: a spectral library and the columns of the current calibration state
    column_name_handler = MagicMock()
    column_name_handler.get_rt_column.return_value = "rt_calibrated"
    column_name_handler.get_precursor_mz_column.return_value = "mz_calibrated"
    column_name_handler.get_fragment_mz_column.return_value = "mz_library"
    spectral_library = MagicMock(name="spectral_library")

    # When: the test creates the handler
    handler = RustExtractionHandler(
        {},
        OPTIMIZATION_CONTEXT,
        MagicMock(),
        MagicMock(),
        column_name_handler,
        MagicMock(name="dia_data"),
        spectral_library,
    )

    # Then: the handler converts the library with these columns and keeps the result
    mock_speclib_to_rust.assert_called_once_with(
        spectral_library,
        rt_column="rt_calibrated",
        precursor_mz_column="mz_calibrated",
        fragment_mz_column="mz_library",
    )
    assert handler._speclib_rust is mock_speclib_to_rust.return_value


@pytest.mark.parametrize("feature_set", [FeatureSet.Calibration, FeatureSet.Full])
@patch("alphadia.workflow.peptidecentric.rust.extraction_handler.to_features_df")
@patch("alphadia.workflow.peptidecentric.rust.extraction_handler.PeakGroupScoring")
@patch("alphadia.workflow.peptidecentric.rust.extraction_handler.candidates_to_rust")
def test_score_candidates_returns_the_scored_features(
    mock_candidates_to_rust, mock_scoring, mock_to_features_df, feature_set
):
    # Given: a handler with a configured top k fragments value
    config = {
        "search": {
            "top_k_fragments_scoring": 12,
            "ms1_features": False,
            "detection_features": False,
        }
    }
    handler = _create_handler(config)

    # When: the handler scores the candidates with the feature set
    features_df = handler.score_candidates(MagicMock(), feature_set)

    # Then: the scoring parameters use the config, the feature set and the optimization context
    scoring_params = mock_scoring.call_args.args[0]
    assert scoring_params.top_k_fragments == 12
    assert scoring_params.feature_set == feature_set
    assert scoring_params.mass_tolerance == 7.5
    assert scoring_params.detection_features is False
    mock_scoring.return_value.score.assert_called_once_with(
        handler._dia_data, handler._speclib_rust, mock_candidates_to_rust.return_value
    )
    mock_to_features_df.assert_called_once_with(
        mock_scoring.return_value.score.return_value,
        handler._spectral_library,
        unscored_feature_groups=["ms1_features", "detection_features"],
    )
    assert features_df is mock_to_features_df.return_value


def _fdr_handler(fit_predict):
    config = {"fdr": {"competitive_scoring": True, "fdr": 0.01}}
    fdr_manager = MagicMock(name="fdr_manager")
    fdr_manager.fit_predict.side_effect = fit_predict
    return _create_handler(config, fdr_manager=fdr_manager), fdr_manager


def _fdr_inputs():
    features_df = pd.DataFrame({"precursor_idx": [1, 2, 3], "rank": [0, 0, 1]})
    candidates_df = pd.DataFrame(
        {"precursor_idx": [1, 2, 3], "rank": [0, 0, 1], "score": [3.0, 2.0, 1.0]}
    )
    return features_df, candidates_df


def test_perform_fdr_and_filter_candidates_quantifies_only_the_competing_candidates():
    # Given: an FDR step that asks for the fragments of precursors 1 and 3 only
    received = {}

    def fit_predict(features_df, **kwargs):
        received["features_df"] = features_df.copy()
        received["fragments"] = kwargs["fragment_provider"](
            features_df[features_df["precursor_idx"] != 2]
        )
        return features_df.assign(qval=[0.0, 0.5, 0.0])

    handler, _ = _fdr_handler(fit_predict)
    features_df, candidates_df = _fdr_inputs()
    fragments_df = pd.DataFrame({"precursor_idx": [1, 3], "mz_observed": [1.0, 2.0]})

    # When: the handler performs FDR without given fragments
    with patch.object(
        handler, "quantify_candidates", return_value=(None, fragments_df)
    ) as mock_quantify:
        candidates_filtered, precursor_fdr_df = (
            handler.perform_fdr_and_filter_candidates(features_df, candidates_df)
        )

    # Then: the handler quantifies only the competing candidates, the FDR step gets their fragments and the FDR result filters the candidates
    quantified = mock_quantify.call_args.args[0]
    assert quantified[["precursor_idx", "rank"]].values.tolist() == [[1, 0], [3, 1]]
    assert "_candidate_idx" not in quantified.columns
    assert mock_quantify.call_args.args[1] is None
    assert received["fragments"] is fragments_df
    assert candidates_filtered["precursor_idx"].tolist() == [1, 3]
    assert precursor_fdr_df["precursor_idx"].tolist() == [1, 3]


def test_perform_fdr_and_filter_candidates_uses_given_fragments():
    # Given: fragments of all candidates, as the optimization rounds have them
    def fit_predict(features_df, **kwargs):
        return features_df.assign(qval=0.0)

    handler, fdr_manager = _fdr_handler(fit_predict)
    features_df, candidates_df = _fdr_inputs()
    fragments_df = pd.DataFrame({"precursor_idx": [1], "mz_observed": [1.0]})

    # When: the handler performs FDR with these fragments
    handler.perform_fdr_and_filter_candidates(
        features_df,
        candidates_df,
        df_fragments=fragments_df,
    )

    # Then: the FDR step gets them directly and no provider
    kwargs = fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["df_fragments"] is fragments_df
    assert kwargs["fragment_provider"] is None


@pytest.mark.parametrize(
    ("final_round", "excluded"),
    [(False, ()), (True, FINAL_ROUND_EXCLUDED_FEATURES)],
)
def test_perform_fdr_and_filter_candidates_leaves_fwhm_rt_out_of_the_final_round(
    final_round, excluded
):
    # Given: an FDR step that accepts all precursors
    def fit_predict(features_df, **kwargs):
        return features_df.assign(qval=0.0)

    handler, fdr_manager = _fdr_handler(fit_predict)
    features_df, candidates_df = _fdr_inputs()

    # When: the handler performs FDR in the final round or in a calibration round
    handler.perform_fdr_and_filter_candidates(
        features_df,
        candidates_df,
        df_fragments=pd.DataFrame(),
        final_round=final_round,
    )

    # Then: only the final round excludes the final round features
    kwargs = fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["excluded_columns"] == excluded


@pytest.mark.parametrize("train_on_best_candidate", [True, False])
def test_perform_fdr_and_filter_candidates_passes_train_on_best_candidate(
    train_on_best_candidate,
):
    # Given: an FDR step that accepts all precursors
    def fit_predict(features_df, **kwargs):
        return features_df.assign(qval=0.0)

    handler, fdr_manager = _fdr_handler(fit_predict)
    features_df, candidates_df = _fdr_inputs()

    # When: the handler performs FDR with the train on best candidate setting
    handler.perform_fdr_and_filter_candidates(
        features_df,
        candidates_df,
        df_fragments=pd.DataFrame(),
        train_on_best_candidate=train_on_best_candidate,
    )

    # Then: the FDR step gets the setting
    kwargs = fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["train_on_best_candidate"] is train_on_best_candidate


# The MS1 and detection features of the final round

HANDLER = "alphadia.workflow.peptidecentric.rust.extraction_handler"


def _precursor_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "elution_group_idx": [5, 5, 9],
            "decoy": [0, 1, 1],
            "charge": [2, 2, 3],
            "mz_library": [500.0, 507.0, 610.0],
            "rt": [10.0, 10.0, 20.0],
            "i_0": [1.0, 1.0, 1.0],
            "i_1": [0.6, 0.7, 0.8],
            "i_2": [0.2, 0.3, 0.4],
            "i_3": [0.05, 0.1, 0.1],
            "channel": 0,
            "proteins": "P",
        }
    )


class _RecordingScoringParameters:
    """Stands in for alphadia_search_rs.ScoringParameters, which needs a release with the MS1 parameters."""

    def __init__(self):
        self.params = {}

    def update(self, params):
        self.params.update(params)


def _handler(
    ms1_features: bool = True,  # noqa: FBT001, FBT002
    detection_features: bool = False,  # noqa: FBT001, FBT002
    *,
    has_ms1: bool = True,
    precursor_df: pd.DataFrame | None = None,
    reporter: MagicMock | None = None,
) -> RustExtractionHandler:
    config = {
        "search": {
            "top_k_fragments_scoring": 12,
            "ms1_features": ms1_features,
            "detection_features": detection_features,
        }
    }
    column_name_handler = MagicMock(name="column_name_handler")
    column_name_handler.get_precursor_mz_column.return_value = "mz_library"
    column_name_handler.get_rt_column.return_value = "rt"
    column_name_handler.dia_data_has_ms1 = has_ms1
    with patch(f"{HANDLER}.speclib_to_rust", return_value=MagicMock()):
        return RustExtractionHandler(
            config,
            OPTIMIZATION_CONTEXT,
            MagicMock(),
            reporter or MagicMock(),
            column_name_handler,
            MagicMock(name="dia_data"),
            SimpleNamespace(
                precursor_df=_precursor_df() if precursor_df is None else precursor_df
            ),
        )


def _candidates_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 0],
            "rank": [0, 0, 0, 1],
            "decoy": [0, 1, 1, 0],
            "score": [3.0, 9.0, 8.0, 7.0],
        }
    )


@patch(f"{HANDLER}.ScoringParameters", _RecordingScoringParameters)
@patch(f"{HANDLER}.to_features_df")
@patch(f"{HANDLER}.set_ms1_isotopes")
@patch(f"{HANDLER}.PeakGroupScoring")
@patch(f"{HANDLER}.candidates_to_rust")
def test_score_candidates_calibrates_ms1_once_on_target_candidates(
    mock_candidates_to_rust,
    mock_scoring,
    mock_set_ms1_isotopes,
    mock_to_features_df,
):
    # given: the pre-pass finds no strict candidates, so the calibration falls back to no correction
    handler = _handler()
    mock_scoring.return_value.score_ms1_calibration.return_value = {
        "precursor_idx": np.array([0]),
        "rank": np.array([0]),
        "ms1_ppm_signed": np.array([0.0]),
        "ms1_n_isotopes": np.array([0.0]),
        "ms1_log_m0_apex": np.array([0.0]),
        "ms1_iso_cosine": np.array([0.0]),
    }

    # when
    for _ in range(2):
        handler.score_candidates(_candidates_df(), FeatureSet.Full)

    # then: the pre-pass extracts the MS1 of the rank-0 targets only, once
    prepass_candidates = mock_candidates_to_rust.call_args_list[0].args[0]
    assert prepass_candidates["precursor_idx"].tolist() == [0]
    mock_scoring.return_value.score_ms1_calibration.assert_called_once()
    assert mock_set_ms1_isotopes.call_count == 2
    # and the extraction uses the fallback tolerance, with the MS1 features kept
    params = [c.args[0].params for c in mock_scoring.call_args_list]
    assert [p["ms1_mass_tolerance"] for p in params] == [
        MS1_PREPASS_TOLERANCE_PPM,
        FALLBACK_TOLERANCE_PPM,
        FALLBACK_TOLERANCE_PPM,
    ]
    assert [p["detection_features"] for p in params] == [False, False, False]
    assert [p["feature_set"] for p in params] == [
        FeatureSet.Calibration,
        FeatureSet.Full,
        FeatureSet.Full,
    ]
    assert mock_to_features_df.call_args.kwargs == {
        "unscored_feature_groups": ["detection_features"]
    }


@pytest.mark.parametrize("ms1_features", [True, False])
@patch(f"{HANDLER}.ScoringParameters", _RecordingScoringParameters)
@patch(f"{HANDLER}.to_features_df")
@patch(f"{HANDLER}.set_ms1_isotopes")
@patch(f"{HANDLER}.PeakGroupScoring")
@patch(f"{HANDLER}.candidates_to_rust")
def test_calibration_rounds_and_runs_without_the_flags_score_no_ms1_or_detection(
    mock_candidates_to_rust,
    mock_scoring,
    mock_set_ms1_isotopes,
    mock_to_features_df,
    ms1_features,
):
    # given: the flags on for the calibration round, off for the final round
    handler = _handler(ms1_features=ms1_features, detection_features=ms1_features)
    feature_set = FeatureSet.Calibration if ms1_features else FeatureSet.Full

    # when
    handler.score_candidates(_candidates_df(), feature_set)

    # then: no pre-pass, no isotopes, no MS1 tolerance, no detection and no MS1 or detection columns
    mock_set_ms1_isotopes.assert_not_called()
    assert mock_scoring.call_count == 1
    params = mock_scoring.call_args.args[0].params
    assert "ms1_mass_tolerance" not in params
    assert params["detection_features"] is False
    assert mock_to_features_df.call_args.kwargs == {
        "unscored_feature_groups": ["ms1_features", "detection_features"]
    }


@patch(f"{HANDLER}.ScoringParameters", _RecordingScoringParameters)
@patch(f"{HANDLER}.to_features_df")
@patch(f"{HANDLER}.set_ms1_isotopes")
@patch(f"{HANDLER}.PeakGroupScoring")
@patch(f"{HANDLER}.candidates_to_rust")
def test_score_candidates_asks_for_the_detection_features_in_the_final_round(
    mock_candidates_to_rust,
    mock_scoring,
    mock_set_ms1_isotopes,
    mock_to_features_df,
):
    # given
    handler = _handler(ms1_features=False, detection_features=True)

    # when
    handler.score_candidates(_candidates_df(), FeatureSet.Full)

    # then
    mock_set_ms1_isotopes.assert_not_called()
    assert mock_scoring.call_args.args[0].params["detection_features"] is True
    assert mock_to_features_df.call_args.kwargs == {
        "unscored_feature_groups": ["ms1_features"]
    }


@pytest.mark.parametrize(
    ("has_ms1", "drop_isotopes", "reason"),
    [
        (False, False, "the raw file has no MS1 scans"),
        (True, True, "the library has no isotope abundances"),
    ],
)
@patch(f"{HANDLER}.ScoringParameters", _RecordingScoringParameters)
@patch(f"{HANDLER}.to_features_df")
@patch(f"{HANDLER}.set_ms1_isotopes")
@patch(f"{HANDLER}.PeakGroupScoring")
@patch(f"{HANDLER}.candidates_to_rust")
def test_ms1_features_are_skipped_with_a_warning_when_they_cannot_be_scored(
    mock_candidates_to_rust,
    mock_scoring,
    mock_set_ms1_isotopes,
    mock_to_features_df,
    has_ms1,
    drop_isotopes,
    reason,
):
    # given: ms1_features on, but no MS1 scans or no isotope abundances in the library
    precursor_df = _precursor_df()
    if drop_isotopes:
        precursor_df = precursor_df.drop(columns=MS1_ISOTOPE_COLUMNS, errors="ignore")
    reporter = MagicMock()
    handler = _handler(has_ms1=has_ms1, precursor_df=precursor_df, reporter=reporter)

    # when
    handler.score_candidates(_candidates_df(), FeatureSet.Full)

    # then: no pre-pass, no MS1 tolerance, no MS1 columns, and a warning that says why
    mock_scoring.return_value.score_ms1_calibration.assert_not_called()
    mock_set_ms1_isotopes.assert_not_called()
    assert "ms1_mass_tolerance" not in mock_scoring.call_args.args[0].params
    assert (
        "ms1_features"
        in mock_to_features_df.call_args.kwargs["unscored_feature_groups"]
    )
    message = reporter.log_string.call_args.args[0]
    assert message.startswith("search.ms1_features is skipped") and reason in message

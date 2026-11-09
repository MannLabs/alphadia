"""Unit tests for the Rust extraction handler."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from alphadia_search_rs import FeatureSet, SearchTolerances

from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    RustExtractionHandler,
)
from alphadia.workflow.peptidecentric.rust.mapper import (
    FINAL_ROUND_EXCLUDED_FEATURES,
)

SPECLIB_TO_RUST = (
    "alphadia.workflow.peptidecentric.rust.extraction_handler.speclib_to_rust"
)

TOLERANCES = SearchTolerances(
    rt_error=120.0,
    ms1_error=30.0,
    ms2_error=7.5,
    mobility_error=0.1,
    fwhm_rt=5.0,
    fwhm_mobility=-1.0,
    num_candidates=3,
    score_cutoff=0.0,
)
CLASSIFIER_VERSION = 4


def _create_handler(config, fdr_manager=None):
    """Create a handler bound to mock DIA data and library, without converting a real library."""
    with patch(SPECLIB_TO_RUST, return_value=MagicMock(name="speclib_rust")):
        return RustExtractionHandler(
            config,
            TOLERANCES,
            fdr_manager or MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(name="dia_data"),
            MagicMock(name="spectral_library"),
        )


@patch(SPECLIB_TO_RUST)
def test_converts_the_library_once_at_construction(mock_speclib_to_rust):
    # given
    column_name_handler = MagicMock()
    column_name_handler.get_rt_column.return_value = "rt_calibrated"
    column_name_handler.get_precursor_mz_column.return_value = "mz_calibrated"
    column_name_handler.get_fragment_mz_column.return_value = "mz_library"
    spectral_library = MagicMock(name="spectral_library")

    # when
    handler = RustExtractionHandler(
        {},
        TOLERANCES,
        MagicMock(),
        MagicMock(),
        column_name_handler,
        MagicMock(name="dia_data"),
        spectral_library,
    )

    # then
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
    # given
    config = {"search": {"top_k_fragments_scoring": 12}}
    handler = _create_handler(config)

    # when
    features_df = handler.score_candidates(MagicMock(), feature_set)

    # then
    scoring_params = mock_scoring.call_args.args[0]
    assert scoring_params.top_k_fragments == 12
    assert scoring_params.feature_set == feature_set
    assert scoring_params.mass_tolerance == 7.5
    mock_scoring.return_value.score.assert_called_once_with(
        handler._dia_data, handler._speclib_rust, mock_candidates_to_rust.return_value
    )
    mock_to_features_df.assert_called_once_with(
        mock_scoring.return_value.score.return_value, handler._spectral_library
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
    # given: an FDR step that asks for the fragments of precursors 1 and 3 only
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

    # when
    with patch.object(
        handler, "quantify_candidates", return_value=(None, fragments_df)
    ) as mock_quantify:
        candidates_filtered, precursor_fdr_df = (
            handler.perform_fdr_and_filter_candidates(
                features_df, candidates_df, CLASSIFIER_VERSION
            )
        )

    # then: only the competing candidates were quantified, and their fragments returned
    quantified = mock_quantify.call_args.args[0]
    assert quantified[["precursor_idx", "rank"]].values.tolist() == [[1, 0], [3, 1]]
    assert "_candidate_idx" not in quantified.columns
    assert mock_quantify.call_args.args[1] is None
    assert received["fragments"] is fragments_df
    # and the candidates are filtered by the FDR result
    assert candidates_filtered["precursor_idx"].tolist() == [1, 3]
    assert precursor_fdr_df["precursor_idx"].tolist() == [1, 3]


def test_perform_fdr_and_filter_candidates_uses_given_fragments():
    # given: fragments of all candidates, as the optimization rounds have them
    def fit_predict(features_df, **kwargs):
        return features_df.assign(qval=0.0)

    handler, fdr_manager = _fdr_handler(fit_predict)
    features_df, candidates_df = _fdr_inputs()
    fragments_df = pd.DataFrame({"precursor_idx": [1], "mz_observed": [1.0]})

    # when
    handler.perform_fdr_and_filter_candidates(
        features_df,
        candidates_df,
        CLASSIFIER_VERSION,
        df_fragments=fragments_df,
    )

    # then: the FDR step gets them directly and no provider, with the given classifier version
    kwargs = fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["df_fragments"] is fragments_df
    assert kwargs["fragment_provider"] is None
    assert kwargs["version"] == CLASSIFIER_VERSION


@pytest.mark.parametrize(
    ("final_round", "excluded"),
    [(False, ()), (True, FINAL_ROUND_EXCLUDED_FEATURES)],
)
def test_perform_fdr_and_filter_candidates_leaves_fwhm_rt_out_of_the_final_round(
    final_round, excluded
):
    # given
    def fit_predict(features_df, **kwargs):
        return features_df.assign(qval=0.0)

    handler, fdr_manager = _fdr_handler(fit_predict)
    features_df, candidates_df = _fdr_inputs()

    # when
    handler.perform_fdr_and_filter_candidates(
        features_df,
        candidates_df,
        CLASSIFIER_VERSION,
        df_fragments=pd.DataFrame(),
        final_round=final_round,
    )

    # then
    kwargs = fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["excluded_columns"] == excluded


@pytest.mark.parametrize("train_on_best_candidate", [True, False])
def test_perform_fdr_and_filter_candidates_passes_train_on_best_candidate(
    train_on_best_candidate,
):
    # given
    def fit_predict(features_df, **kwargs):
        return features_df.assign(qval=0.0)

    handler, fdr_manager = _fdr_handler(fit_predict)
    features_df, candidates_df = _fdr_inputs()

    # when
    handler.perform_fdr_and_filter_candidates(
        features_df,
        candidates_df,
        CLASSIFIER_VERSION,
        df_fragments=pd.DataFrame(),
        train_on_best_candidate=train_on_best_candidate,
    )

    # then
    kwargs = fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["train_on_best_candidate"] is train_on_best_candidate

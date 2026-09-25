"""Unit tests for the NG extraction handler."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from conftest import mock_context_features

from alphadia.workflow.peptidecentric.extraction_handler import NgExtractionHandler
from alphadia.workflow.peptidecentric.ng.ng_mapper import get_context_feature_names


@pytest.mark.parametrize("candidate_context_features", [True, False])
@patch("alphadia.workflow.peptidecentric.extraction_handler.CandidateContext")
@patch("alphadia.workflow.peptidecentric.extraction_handler.to_features_df")
@patch("alphadia.workflow.peptidecentric.extraction_handler.PeakGroupScoring")
@patch("alphadia.workflow.peptidecentric.extraction_handler.candidates_to_ng")
def test_score_candidates_adds_context_features_if_enabled(
    mock_candidates_to_ng,
    mock_scoring,
    mock_to_features_df,
    mock_candidate_context,
    candidate_context_features,
):
    # given
    config = {
        "search": {
            "top_k_fragments_scoring": 12,
            "candidate_context_features": candidate_context_features,
        }
    }
    handler = NgExtractionHandler(
        config, MagicMock(ms2_error=7.5), MagicMock(), MagicMock(), MagicMock()
    )
    handler._speclib_ng = MagicMock(name="speclib_ng")
    dia_data = MagicMock(name="dia_data")
    mock_to_features_df.return_value = pd.DataFrame(
        {"precursor_idx": [1, 2], "rank": [0, 0], "score": [1.0, 2.0]}
    )
    mock_candidate_context.return_value.compute.return_value = mock_context_features(
        [2, 1], [0, 0]
    )

    # when
    features_df = handler.score_candidates(MagicMock(), dia_data, MagicMock())

    # then
    has_context = set(get_context_feature_names()) <= set(features_df.columns)
    assert has_context == candidate_context_features
    if candidate_context_features:
        mock_candidate_context.assert_called_once_with(
            mass_tolerance=7.5, top_k_fragments=12
        )
        mock_candidate_context.return_value.compute.assert_called_once_with(
            dia_data, handler._speclib_ng, mock_candidates_to_ng.return_value
        )


def _fdr_handler(fit_predict):
    config = {"fdr": {"competitive_scoring": True, "fdr": 0.01}}
    fdr_manager = MagicMock(name="fdr_manager")
    fdr_manager.fit_predict.side_effect = fit_predict
    column_name_handler = MagicMock(name="column_name_handler")
    column_name_handler.get_precursor_mz_column.return_value = "mz_calibrated"
    return (
        NgExtractionHandler(
            config, MagicMock(), fdr_manager, MagicMock(), column_name_handler
        ),
        fdr_manager,
    )


def _fdr_inputs():
    features_df = pd.DataFrame({"precursor_idx": [1, 2, 3], "rank": [0, 0, 1]})
    candidates_df = pd.DataFrame(
        {"precursor_idx": [1, 2, 3], "rank": [0, 0, 1], "score": [3.0, 2.0, 1.0]}
    )
    spectral_library = MagicMock(name="spectral_library")
    spectral_library.precursor_df = pd.DataFrame(
        {"precursor_idx": [3, 2, 1], "mz_calibrated": [503.0, 502.0, 501.0]}
    )
    return features_df, candidates_df, spectral_library


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
    features_df, candidates_df, spectral_library = _fdr_inputs()
    fragments_df = pd.DataFrame({"precursor_idx": [1, 3], "mz_observed": [1.0, 2.0]})
    dia_data = MagicMock(name="dia_data")

    # when
    with patch.object(
        handler, "quantify_candidates", return_value=(None, fragments_df)
    ) as mock_quantify:
        candidates_filtered, precursor_fdr_df = (
            handler.perform_fdr_and_filter_candidates(
                features_df, candidates_df, dia_data, spectral_library
            )
        )

    # then: only the competing candidates were quantified, and their fragments returned
    quantified = mock_quantify.call_args.args[0]
    assert quantified[["precursor_idx", "rank"]].values.tolist() == [[1, 0], [3, 1]]
    assert "_candidate_idx" not in quantified.columns
    assert mock_quantify.call_args.args[2:] == (dia_data, spectral_library)
    assert received["fragments"] is fragments_df
    # and every PSM carries its precursor m/z for the isolation window assignment
    assert received["features_df"]["mz_observed"].tolist() == [501.0, 502.0, 503.0]
    # and the candidates are filtered by the FDR result
    assert candidates_filtered["precursor_idx"].tolist() == [1, 3]
    assert precursor_fdr_df["precursor_idx"].tolist() == [1, 3]


def test_perform_fdr_and_filter_candidates_uses_given_fragments():
    # given: fragments of all candidates, as the optimization rounds have them
    def fit_predict(features_df, **kwargs):
        return features_df.assign(qval=0.0)

    handler, fdr_manager = _fdr_handler(fit_predict)
    features_df, candidates_df, spectral_library = _fdr_inputs()
    fragments_df = pd.DataFrame({"precursor_idx": [1], "mz_observed": [1.0]})

    # when
    handler.perform_fdr_and_filter_candidates(
        features_df,
        candidates_df,
        MagicMock(),
        spectral_library,
        df_fragments=fragments_df,
    )

    # then: the FDR step gets them directly and no provider
    kwargs = fdr_manager.fit_predict.call_args.kwargs
    assert kwargs["df_fragments"] is fragments_df
    assert kwargs["fragment_provider"] is None

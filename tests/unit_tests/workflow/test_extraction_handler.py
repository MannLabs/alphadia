"""Unit tests for the NG extraction handler."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from conftest import mock_context_features

from alphadia.workflow.peptidecentric.extraction_handler import NgExtractionHandler
from alphadia.workflow.peptidecentric.ng.ng_mapper import get_context_feature_names


@pytest.mark.parametrize("competition_features", [True, False])
@patch("alphadia.workflow.peptidecentric.extraction_handler.CandidateContext")
@patch("alphadia.workflow.peptidecentric.extraction_handler.to_features_df")
@patch("alphadia.workflow.peptidecentric.extraction_handler.PeakGroupScoring")
@patch("alphadia.workflow.peptidecentric.extraction_handler.candidates_to_ng")
def test_score_candidates_adds_context_features_if_enabled(
    mock_candidates_to_ng,
    mock_scoring,
    mock_to_features_df,
    mock_candidate_context,
    competition_features,
):
    # given
    config = {
        "search": {
            "top_k_fragments_scoring": 12,
            "competition_features": competition_features,
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
    assert has_context == competition_features
    if competition_features:
        mock_candidate_context.assert_called_once_with(
            mass_tolerance=7.5, top_k_fragments=12
        )
        mock_candidate_context.return_value.compute.assert_called_once_with(
            dia_data, handler._speclib_ng, mock_candidates_to_ng.return_value
        )

"""Unit tests for the NG extraction handler."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from alphadia.workflow.peptidecentric.extraction_handler import NgExtractionHandler
from alphadia.workflow.peptidecentric.ng.ng_mapper import get_context_feature_names


def _handler(config: dict) -> NgExtractionHandler:
    optimization_manager = MagicMock()
    optimization_manager.ms2_error = 7.5
    handler = NgExtractionHandler(
        config, optimization_manager, MagicMock(), MagicMock(), MagicMock()
    )
    handler._speclib_ng = MagicMock(name="speclib_ng")
    return handler


def _context_features(precursor_idx: list[int], rank: list[int]) -> dict:
    context_features = {
        "precursor_idx": np.array(precursor_idx, dtype=np.uint64),
        "rank": np.array(rank, dtype=np.uint64),
    }
    for name in get_context_feature_names():
        context_features[name] = np.zeros(len(precursor_idx), dtype=np.float32)
    return context_features


@patch("alphadia.workflow.peptidecentric.extraction_handler.CandidateContext")
def test_add_context_features_passes_parameters_and_merges(mock_candidate_context):
    # given
    config = {
        "search": {"top_k_fragments_scoring": 12},
        "fdr": {"competition_features": True, "competition_min_shared": 4},
    }
    handler = _handler(config)
    features_df = pd.DataFrame(
        {"precursor_idx": [1, 2], "rank": [0, 0], "score": [1.0, 2.0]}
    )
    candidates = MagicMock(name="candidates")
    dia_data = MagicMock(name="dia_data")
    mock_candidate_context.return_value.compute.return_value = _context_features(
        [2, 1], [0, 0]
    )

    # when
    result_df = handler._add_context_features(features_df, candidates, dia_data)

    # then
    mock_candidate_context.assert_called_once_with(
        mass_tolerance=7.5, top_k_fragments=12, min_shared=4
    )
    mock_candidate_context.return_value.compute.assert_called_once_with(
        dia_data, handler._speclib_ng, candidates
    )
    assert len(result_df) == 2
    for name in get_context_feature_names():
        assert name in result_df.columns
        assert not result_df[name].isna().any()

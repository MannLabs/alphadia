"""Unit tests for the NG extraction handler."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from conftest import mock_context_features

from alphadia.fdr import riding
from alphadia.workflow.peptidecentric import extraction_handler
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
    return (
        NgExtractionHandler(config, MagicMock(), fdr_manager, MagicMock(), MagicMock()),
        fdr_manager,
    )


def _fdr_inputs():
    features_df = pd.DataFrame({"precursor_idx": [1, 2, 3], "rank": [0, 0, 1]})
    candidates_df = pd.DataFrame(
        {"precursor_idx": [1, 2, 3], "rank": [0, 0, 1], "score": [3.0, 2.0, 1.0]}
    )
    return features_df, candidates_df, MagicMock(name="spectral_library")


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


@pytest.mark.parametrize("remove_riding", [True, False])
def test_perform_fdr_and_filter_candidates_provides_riding_fragments_if_asked(
    remove_riding,
):
    # given: precursors 1 and 3 are charge states of one peptide
    received = {}

    def fit_predict(features_df, **kwargs):
        received["features_df"] = features_df.copy()
        received["kwargs"] = kwargs
        if kwargs["riding_provider"] is not None:
            # the provider is asked for the PSMs in a different order than the candidates
            received["riding"] = kwargs["riding_provider"](features_df.iloc[[2, 0]])
        return features_df.assign(qval=0.0)

    handler, _ = _fdr_handler(fit_predict)
    handler._optimization_manager.ms2_error = 6.5
    features_df, candidates_df, spectral_library = _fdr_inputs()
    spectral_library.precursor_df = pd.DataFrame(
        {"precursor_idx": [3, 2, 1], "sequence": ["PEPTIDEK", "SAMPLER", "PEPTIDEK"]}
    )
    fragments_df = pd.DataFrame()

    # when
    with (
        patch.object(
            handler, "quantify_candidates", return_value=(None, fragments_df)
        ) as mock_quantify,
        patch.object(extraction_handler, "_riding_fragments") as mock_riding_fragments,
    ):
        handler.perform_fdr_and_filter_candidates(
            features_df,
            candidates_df,
            MagicMock(),
            spectral_library,
            remove_riding=remove_riding,
            train_on_best_candidate=remove_riding,
        )

    # then
    kwargs = received["kwargs"]
    assert kwargs["ms2_tolerance_ppm"] == 6.5
    assert kwargs["train_on_best_candidate"] is remove_riding
    if not remove_riding:
        assert kwargs["riding_provider"] is None
        assert riding.SEQUENCE_ID_COLUMN not in received["features_df"].columns
        return
    # charge states of one peptide share a sequence id, so they never explain each other's fragments
    sequence_id = received["features_df"][riding.SEQUENCE_ID_COLUMN].tolist()
    assert sequence_id[0] == sequence_id[2] != sequence_id[1]
    # the PSMs are quantified in the order asked, with the riding fragments at the calibrated tolerance
    quantified = mock_quantify.call_args.args[0]
    assert quantified[["precursor_idx", "rank"]].values.tolist() == [[3, 1], [1, 0]]
    assert mock_quantify.call_args.kwargs == {
        "top_k_fragments": riding.TOP_K_FRAGMENTS,
        "tolerance_ppm": 6.5,
    }
    assert mock_riding_fragments.call_args.args[0] is fragments_df
    assert received["riding"] is mock_riding_fragments.return_value


def test_riding_fragments_selects_top_library_fragments_and_fills_missing():
    # given: precursor 7 with five library fragments (one y1, one without intensity), quantified at rank 0 and 1
    spectral_library = MagicMock(name="spectral_library")
    spectral_library.precursor_df = pd.DataFrame(
        {
            "precursor_idx": [9, 7],
            "flat_frag_start_idx": [5, 0],
            "flat_frag_stop_idx": [6, 5],
        }
    )
    spectral_library.fragment_df = pd.DataFrame(
        {
            "mz_library": [200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
            "intensity": [0.5, 1.0, 0.0, 0.9, 0.2, 1.0],
            "type": [ord("b"), ord("y"), ord("b"), ord("y"), ord("b"), ord("b")],
            "number": [2, 1, 3, 4, 5, 2],
        }
    )
    candidates_df = pd.DataFrame({"precursor_idx": [7, 7], "rank": [1, 0]})
    fragments_df = pd.DataFrame(
        {
            "precursor_idx": [7, 7],
            "rank": [0, 1],
            "mz_library": [500.0, 200.0],
            "intensity": [90.0, 40.0],
            "correlation": [0.9, 0.4],
        }
    )

    # when
    mz, library, observed, correlation = extraction_handler._riding_fragments(
        fragments_df, candidates_df, spectral_library
    )

    # then: most intense first, y1 and zero-intensity fragments left out, unquantified fragments carry no signal
    k = riding.TOP_K_FRAGMENTS
    assert mz.shape == (2, k)
    assert mz[:, :3].tolist() == [[500.0, 200.0, 600.0]] * 2
    assert (mz[:, 3:] == 0).all()
    np.testing.assert_allclose(library[0, :3], [0.9, 0.5, 0.2])
    assert observed[:, :3].tolist() == [[0.0, 40.0, 0.0], [90.0, 0.0, 0.0]]
    np.testing.assert_allclose(correlation[:, :3], [[0.0, 0.4, 0.0], [0.9, 0.0, 0.0]])

"""Unit tests for the NG extraction handler."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from conftest import mock_claim_features, mock_context_features

from alphadia.workflow.peptidecentric.extraction_handler import NgExtractionHandler
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_claim_feature_names,
    get_context_feature_names,
)


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
            "fragment_claim_features": False,
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


def _claim_config(fragment_claim_features: bool, fragment_claim_soft: bool) -> dict:
    return {
        "search": {
            "top_k_fragments_scoring": 12,
            "candidate_context_features": False,
            "fragment_claim_features": fragment_claim_features,
            "fragment_claim_soft": fragment_claim_soft,
            "fragment_claim_tolerance": None,
            "fragment_claim_cycle_radius": 2,
        }
    }


@pytest.mark.parametrize(
    ("fragment_claim_features", "fragment_claim_soft"),
    [(False, False), (True, False), (True, True)],
)
@patch("alphadia.workflow.peptidecentric.extraction_handler.FragmentClaim")
@patch("alphadia.workflow.peptidecentric.extraction_handler.to_features_df")
@patch("alphadia.workflow.peptidecentric.extraction_handler.PeakGroupScoring")
@patch("alphadia.workflow.peptidecentric.extraction_handler.candidates_to_ng")
def test_score_candidates_adds_claim_features_if_enabled(
    mock_candidates_to_ng,
    mock_scoring,
    mock_to_features_df,
    mock_fragment_claim,
    fragment_claim_features,
    fragment_claim_soft,
):
    # given a handler with the hard and soft claims switched on or off
    handler = NgExtractionHandler(
        _claim_config(fragment_claim_features, fragment_claim_soft),
        MagicMock(ms2_error=7.5),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    handler._speclib_ng = MagicMock(name="speclib_ng")
    dia_data = MagicMock(name="dia_data")
    mock_to_features_df.return_value = pd.DataFrame(
        {"precursor_idx": [1, 2], "rank": [0, 0], "score": [1.0, 2.0]}
    )
    mock_fragment_claim.return_value.compute.return_value = mock_claim_features(
        [2, 1], [0, 0], soft_claims=fragment_claim_soft
    )

    # when
    features_df = handler.score_candidates(MagicMock(), dia_data, MagicMock())

    # then the hard claims are present exactly when enabled, the soft ones only with soft claims
    claim_columns = [c for c in features_df.columns if c.startswith("clm_")]
    expected_columns = [
        name
        for name in get_claim_feature_names()
        if fragment_claim_features
        and (fragment_claim_soft or not name.startswith("clm_soft_"))
    ]
    assert claim_columns == expected_columns
    if fragment_claim_features:
        mock_fragment_claim.assert_called_once_with(
            mass_tolerance=7.5,
            top_k_fragments=12,
            claim_tolerance=None,
            cycle_radius=2,
            soft_claims=fragment_claim_soft,
        )
        mock_fragment_claim.return_value.compute.assert_called_once_with(
            dia_data, handler._speclib_ng, mock_candidates_to_ng.return_value
        )
        # and the rows are aligned on the candidate, not on the position
        assert features_df["clm_n_claimed"].tolist() == [1.0, 0.0]
    else:
        mock_fragment_claim.assert_not_called()


@patch("alphadia.workflow.peptidecentric.extraction_handler.FragmentClaim")
@patch("alphadia.workflow.peptidecentric.extraction_handler.to_features_df")
@patch("alphadia.workflow.peptidecentric.extraction_handler.PeakGroupScoring")
@patch("alphadia.workflow.peptidecentric.extraction_handler.candidates_to_ng")
def test_score_candidates_gives_an_optimization_batch_the_columns_of_the_extraction(
    mock_candidates_to_ng,
    mock_scoring,
    mock_to_features_df,
    mock_fragment_claim,
):
    # given claims on, an optimization batch of one candidate and the full extraction of two
    handler = NgExtractionHandler(
        _claim_config(fragment_claim_features=True, fragment_claim_soft=True),
        MagicMock(ms2_error=7.5),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    handler._speclib_ng = MagicMock(name="speclib_ng")
    mock_to_features_df.side_effect = [
        pd.DataFrame({"precursor_idx": [1], "rank": [0], "score": [1.0]}),
        pd.DataFrame({"precursor_idx": [1, 2], "rank": [0, 0], "score": [1.0, 2.0]}),
    ]
    mock_fragment_claim.return_value.compute.side_effect = [
        mock_claim_features([1], [0], soft_claims=True),
        mock_claim_features([1, 2], [0, 0], soft_claims=True),
    ]

    # when
    batch_df = handler.score_candidates(MagicMock(), MagicMock(), MagicMock())
    extraction_df = handler.score_candidates(MagicMock(), MagicMock(), MagicMock())

    # then both carry the same columns, so the FDR manager keys them to the same classifier
    assert list(batch_df.columns) == list(extraction_df.columns)
    assert set(get_claim_feature_names()) <= set(batch_df.columns)


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

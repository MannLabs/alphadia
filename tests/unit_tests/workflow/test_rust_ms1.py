"""Unit tests for the final-round MS1 and detection features: MS1 calibration, isotope setup, feature columns and the
MS1 pre-pass.

The tests do not need an alphadia-search-rs release with these features: the scoring and its parameters are mocked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from alphadia_search_rs import FeatureSet, OptimizationContext

from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    MS1_PREPASS_TOLERANCE_PPM,
    RustExtractionHandler,
)
from alphadia.workflow.peptidecentric.rust.mapper import (
    DETECTION_FEATURES,
    MS1_FEATURES,
    set_ms1_isotopes,
    to_features_df,
)
from alphadia.workflow.peptidecentric.rust.ms1_calibration import (
    FALLBACK_TOLERANCE_PPM,
    Ms1Calibration,
    fit_ms1_calibration,
)

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


class _RecordingSpeclib:
    def set_isotopes(self, *arrays):
        self.arrays = arrays


def test_set_ms1_isotopes_scores_every_precursor_at_its_own_calibrated_mass():
    # given: a calibration of +2 ppm over the whole m/z range
    speclib = _RecordingSpeclib()
    calibration = Ms1Calibration(
        np.array([400.0, 700.0]),
        np.array([2.0, 2.0]),
        np.array([0.0]),
        np.array([0.0]),
        1.0,
        4.0,
        500,
    )

    # when
    set_ms1_isotopes(speclib, _precursor_df(), "mz_library", "rt", calibration)

    # then: decoys keep their own m/z, charge and envelope
    idx, charge, mz, intensity = speclib.arrays
    assert idx.tolist() == [0, 1, 2]
    assert charge.tolist() == [2, 2, 3]
    np.testing.assert_allclose(mz, np.array([500.0, 507.0, 610.0]) * (1 + 2e-6))
    np.testing.assert_allclose(intensity[1], [1.0, 0.7, 0.3, 0.1])
    assert intensity.shape == (3, 4)


def test_set_ms1_isotopes_uses_the_isotopes_the_library_has():
    # given: a library with three isotopes only
    speclib = _RecordingSpeclib()
    precursor_df = _precursor_df().drop(columns="i_3")

    # when
    set_ms1_isotopes(
        speclib, precursor_df, "mz_library", "rt", Ms1Calibration.identity()
    )

    # then
    assert speclib.arrays[3].shape == (3, 3)


def test_set_ms1_isotopes_rejects_a_library_without_isotopes():
    # given
    precursor_df = _precursor_df().drop(columns=["i_0", "i_1", "i_2", "i_3"])

    # when / then
    with pytest.raises(ValueError, match="isotope abundances"):
        set_ms1_isotopes(
            _RecordingSpeclib(),
            precursor_df,
            "mz_library",
            "rt",
            Ms1Calibration.identity(),
        )


def test_fit_ms1_calibration_recovers_an_mz_trend_and_the_spread():
    # given
    rng = np.random.default_rng(0)
    n = 5000
    mz = rng.uniform(400, 900, n)
    rt = rng.uniform(0, 100, n)
    true_offset = 1.5 - 0.002 * (mz - 400)
    ppm = true_offset + rng.normal(0, 0.8, n)
    # a quarter of the candidates match another peak in the wide window, or fail the strict rules
    ppm[: n // 4] = rng.uniform(-15, 15, n // 4)
    iso_cosine = np.where(np.arange(n) < n // 8, 0.5, 0.97)

    # when
    calibration = fit_ms1_calibration(
        mz, rt, ppm, np.full(n, 4.0), np.full(n, 5.0), iso_cosine
    )

    # then
    test_mz = np.array([450.0, 650.0, 850.0])
    np.testing.assert_allclose(
        calibration.offset_ppm(test_mz, np.full(3, 50.0)),
        1.5 - 0.002 * (test_mz - 400),
        atol=0.25,
    )
    assert 0.6 < calibration.sigma_ppm < 1.0
    assert calibration.tolerance_ppm == np.clip(4 * calibration.sigma_ppm, 3, 15)
    assert "strict target candidates" in calibration.describe()


def test_fit_ms1_calibration_falls_back_without_enough_strict_candidates():
    # when
    calibration = fit_ms1_calibration(
        np.full(10, 500.0),
        np.zeros(10),
        np.zeros(10),
        np.full(10, 4.0),
        np.full(10, 5.0),
        np.full(10, 0.97),
    )

    # then
    assert calibration.num_candidates == 0
    assert calibration.tolerance_ppm == FALLBACK_TOLERANCE_PPM
    assert calibration.offset_ppm(np.array([500.0]), np.array([0.0]))[0] == 0.0


def test_fit_ms1_calibration_falls_back_without_candidates():
    # when
    empty = np.zeros(0)
    calibration = fit_ms1_calibration(empty, empty, empty, empty, empty, empty)

    # then
    assert calibration.num_candidates == 0


def _candidate_features() -> SimpleNamespace:
    columns = {
        "precursor_idx": np.array([0, 1, 2]),
        "rank": np.array([0, 0, 0]),
        "score": np.array([1.0, 2.0, 3.0]),
    }
    for k, name in enumerate(MS1_FEATURES + DETECTION_FEATURES):
        columns[name] = np.full(3, float(k))
    return SimpleNamespace(to_dict_arrays=lambda: columns)


@pytest.mark.parametrize(
    ("ms1_scored", "detection_scored"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_to_features_df_keeps_only_the_scored_blocks(ms1_scored, detection_scored):
    # given
    library = SimpleNamespace(precursor_df=_precursor_df())

    # when
    features_df = to_features_df(
        _candidate_features(),
        library,
        ms1_scored=ms1_scored,
        detection_scored=detection_scored,
    )

    # then
    assert ms1_scored == any(name in features_df.columns for name in MS1_FEATURES)
    assert detection_scored == any(
        name in features_df.columns for name in DETECTION_FEATURES
    )
    assert "score" in features_df.columns


def test_the_blocks_have_the_13_selected_features():
    # given / when / then
    assert len(MS1_FEATURES) == 10
    assert len(DETECTION_FEATURES) == 3
    assert all(name.startswith("ms1_") for name in MS1_FEATURES)


class _RecordingScoringParameters:
    """Stands in for alphadia_search_rs.ScoringParameters, which needs a release with the MS1 parameters."""

    def __init__(self):
        self.params = {}

    def update(self, params):
        self.params.update(params)


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


def _handler(
    ms1_features: bool = True,  # noqa: FBT001, FBT002
    detection_features: bool = False,  # noqa: FBT001, FBT002
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
    with patch(f"{HANDLER}.speclib_to_rust", return_value=MagicMock()):
        return RustExtractionHandler(
            config,
            OPTIMIZATION_CONTEXT,
            MagicMock(),
            MagicMock(),
            column_name_handler,
            MagicMock(name="dia_data"),
            SimpleNamespace(precursor_df=_precursor_df()),
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
    assert [p["feature_set"] for p in params] == [
        FeatureSet.Calibration,
        FeatureSet.Full,
        FeatureSet.Full,
    ]
    assert mock_to_features_df.call_args.kwargs == {
        "ms1_scored": True,
        "detection_scored": False,
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
    assert "detection_features" not in params
    assert mock_to_features_df.call_args.kwargs == {
        "ms1_scored": False,
        "detection_scored": False,
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
        "ms1_scored": False,
        "detection_scored": True,
    }

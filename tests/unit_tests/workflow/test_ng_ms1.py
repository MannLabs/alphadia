"""Unit tests for the final-round MS1 features: calibration, isotope setup, feature columns and the pre-pass."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from alphadia.workflow.peptidecentric.extraction_handler import NgExtractionHandler
from alphadia.workflow.peptidecentric.ng.ms1_calibration import (
    Ms1Calibration,
    fit_ms1_calibration,
)
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    MS1_CLASSIFIER_FEATURES,
    get_final_round_feature_names,
    set_ms1_isotopes,
    to_features_df,
)
from alphadia.workflow.peptidecentric.utils import get_classifier_feature_columns


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
    assert calibration.offset_ppm(np.array([500.0]), np.array([0.0]))[0] == 0.0


def _candidate_features() -> SimpleNamespace:
    return SimpleNamespace(
        to_dict_arrays=lambda: {
            "precursor_idx": np.array([0, 1, 2]),
            "rank": np.array([0, 0, 0]),
            "score": np.array([1.0, 2.0, 3.0]),
            "ms1_log_m0_apex": np.array([4.0, 3.0, 0.0]),
            "ms1_n_isotopes": np.array([4.0, 2.0, 0.0]),
            "ms1_frag_corr": np.array([0.9, 0.1, 0.0]),
            "ms1_iso_cosine": np.array([0.99, 0.5, 0.0]),
            "ms1_contrast": np.array([1.0, 0.2, 0.0]),
            "ms1_ppm_signed": np.array([0.5, -2.0, 0.0]),
            "ms1_fine_frag_corr": np.array([0.8, 0.2, 0.0]),
            "ms1_fine_shape": np.array([0.9, 0.1, 0.0]),
            "ms1_fine_apex_shift": np.array([0.3, -4.0, 0.0]),
            "ms1_fine_iso_corr": np.array([0.95, 0.3, 0.0]),
            "ms1_fine_n_points": np.array([7.0, 2.0, 0.0]),
        }
    )


def test_to_features_df_keeps_only_the_classifier_ms1_features():
    # given
    library = SimpleNamespace(precursor_df=_precursor_df())

    # when
    features_df = to_features_df(_candidate_features(), library, ms1_scored=True)

    # then
    ms1_columns = [c for c in features_df.columns if c.startswith("ms1")]
    assert sorted(ms1_columns) == sorted(MS1_CLASSIFIER_FEATURES)
    assert features_df["ms1_log_m0_apex"].tolist() == [4.0, 3.0, 0.0]


def test_to_features_df_drops_unscored_ms1_columns():
    # given
    library = SimpleNamespace(precursor_df=_precursor_df())

    # when
    features_df = to_features_df(_candidate_features(), library)

    # then
    assert not any(c.startswith("ms1") for c in features_df.columns)


def test_only_the_classifier_ms1_features_enter_the_classifier_in_the_final_round():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert sorted(c for c in columns if c.startswith("ms1")) == sorted(
        MS1_CLASSIFIER_FEATURES
    )
    assert set(MS1_CLASSIFIER_FEATURES) <= set(get_final_round_feature_names())


def _ms1_handler() -> NgExtractionHandler:
    config = {
        "search": {"top_k_fragments_scoring": 12, "candidate_context_features": False}
    }
    column_name_handler = MagicMock(name="column_name_handler")
    column_name_handler.get_precursor_mz_column.return_value = "mz_library"
    column_name_handler.get_rt_column.return_value = "rt"
    handler = NgExtractionHandler(
        config, MagicMock(ms2_error=7.5), MagicMock(), MagicMock(), column_name_handler
    )
    handler._speclib_ng = MagicMock(name="speclib_ng")
    return handler


@patch("alphadia.workflow.peptidecentric.extraction_handler.to_features_df")
@patch("alphadia.workflow.peptidecentric.extraction_handler.set_ms1_isotopes")
@patch("alphadia.workflow.peptidecentric.extraction_handler.PeakGroupScoring")
@patch("alphadia.workflow.peptidecentric.extraction_handler.candidates_to_ng")
def test_score_candidates_calibrates_ms1_once_on_target_candidates(
    mock_candidates_to_ng,
    mock_scoring,
    mock_set_ms1_isotopes,
    mock_to_features_df,
):
    # given: the pre-pass finds no strict candidates, so the calibration falls back to no correction
    handler = _ms1_handler()
    handler.enable_ms1_scoring()
    candidates_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 0],
            "rank": [0, 0, 0, 1],
            "decoy": [0, 1, 1, 0],
            "score": [3.0, 9.0, 8.0, 7.0],
        }
    )
    spectral_library = SimpleNamespace(precursor_df=_precursor_df())
    mock_scoring.return_value.score.return_value.to_dict_arrays.return_value = {
        "precursor_idx": np.array([0]),
        "ms1_ppm_signed": np.array([0.0]),
        "ms1_n_isotopes": np.array([0.0]),
        "ms1_log_m0_apex": np.array([0.0]),
        "ms1_iso_cosine": np.array([0.0]),
    }

    # when
    handler.score_candidates(candidates_df, MagicMock(), spectral_library)
    handler.score_candidates(candidates_df, MagicMock(), spectral_library)

    # then: the pre-pass scores the rank-0 targets only, once
    prepass_candidates = mock_candidates_to_ng.call_args_list[0].args[0]
    assert prepass_candidates["precursor_idx"].tolist() == [0]
    assert mock_set_ms1_isotopes.call_count == 2
    assert mock_scoring.call_count == 3
    # and the extraction uses the fallback tolerance, with the MS1 features kept
    tolerances = [c.args[0].ms1_mass_tolerance for c in mock_scoring.call_args_list]
    assert tolerances == [15.0, 10.0, 10.0]
    assert mock_to_features_df.call_args.kwargs["ms1_scored"] is True

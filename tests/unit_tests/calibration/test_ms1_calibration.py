"""Unit tests for the MS1 mass calibration of the final-round MS1 features."""

import numpy as np

from alphadia.calibration.ms1_calibration import (
    FALLBACK_TOLERANCE_PPM,
    Ms1PrepassMatches,
    fit_ms1_calibration,
)


def _matches(mz, rt, ppm_signed, n_isotopes, log_m0_apex, iso_cosine):
    return Ms1PrepassMatches(
        mz=mz,
        rt=rt,
        ppm_signed=ppm_signed,
        n_isotopes=n_isotopes,
        log_m0_apex=log_m0_apex,
        iso_cosine=iso_cosine,
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
        _matches(mz, rt, ppm, np.full(n, 4.0), np.full(n, 5.0), iso_cosine)
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
        _matches(
            np.full(10, 500.0),
            np.zeros(10),
            np.zeros(10),
            np.full(10, 4.0),
            np.full(10, 5.0),
            np.full(10, 0.97),
        )
    )

    # then
    assert calibration.num_candidates == 0
    assert calibration.tolerance_ppm == FALLBACK_TOLERANCE_PPM
    assert calibration.offset_ppm(np.array([500.0]), np.array([0.0]))[0] == 0.0


def test_fit_ms1_calibration_falls_back_without_candidates():
    # when
    empty = np.zeros(0)
    calibration = fit_ms1_calibration(
        _matches(empty, empty, empty, empty, empty, empty)
    )

    # then
    assert calibration.num_candidates == 0

import numpy as np
import pandas as pd

from alphadia.calibration.estimator import CalibrationEstimator


def _mz_testdata(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    library_mz = np.linspace(100, 1000, n)
    observed_mz = library_mz * (1 + 1e-5) + rng.normal(0, 1e-3, n)
    return pd.DataFrame({"library_mz": library_mz, "observed_mz": observed_mz})


def test_fit_predict_ppm():
    mz_df = _mz_testdata()

    mz_calibration = CalibrationEstimator(
        name="mz",
        n_kernels=2,
        transform_deviation=1e6,
        input_column="library_mz",
        target_column="observed_mz",
        output_column="calibrated_mz",
    )

    assert mz_calibration.is_fitted is False
    assert mz_calibration.transform_deviation == 1e6

    mz_calibration.fit(mz_df, plot=False)
    mz_calibration.predict(mz_df)

    assert mz_calibration.is_fitted is True
    assert "calibrated_mz" in mz_df.columns
    assert mz_calibration.metrics.median_accuracy >= 0
    assert mz_calibration.metrics.median_precision >= 0

    # calibration should bring the predicted values closer to the observed ones
    raw_err = np.median(np.abs(mz_df["observed_mz"] - mz_df["library_mz"]))
    cal_err = np.median(np.abs(mz_df["observed_mz"] - mz_df["calibrated_mz"]))
    assert cal_err < raw_err


def test_fit_predict_rt_absolute():
    rng = np.random.default_rng(0)
    rt_library = np.linspace(0, 100, 500)
    rt_observed = rt_library + np.sin(rt_library * 0.05) + rng.normal(0, 0.1, 500)
    rt_df = pd.DataFrame({"rt_library": rt_library, "rt_observed": rt_observed})

    rt_calibration = CalibrationEstimator(
        name="rt",
        n_kernels=6,
        transform_deviation=None,
        input_column="rt_library",
        target_column="rt_observed",
        output_column="rt_calibrated",
    )

    assert rt_calibration.transform_deviation is None

    rt_calibration.fit(rt_df, plot=False)
    predicted = rt_calibration.predict(rt_df, inplace=False)

    assert predicted is not None
    assert len(predicted) == len(rt_df)
    # the nonlinear warp should be captured well
    assert np.median(np.abs(predicted - rt_observed)) < 0.5


def test_predict_before_fit_returns_none():
    mz_calibration = CalibrationEstimator(
        name="mz",
        n_kernels=2,
        transform_deviation=1e6,
        input_column="library_mz",
        target_column="observed_mz",
        output_column="calibrated_mz",
    )
    assert mz_calibration.predict(_mz_testdata()) is None


def test_ci():
    mz_df = _mz_testdata()
    mz_calibration = CalibrationEstimator(
        name="mz",
        n_kernels=2,
        transform_deviation=1e6,
        input_column="library_mz",
        target_column="observed_mz",
        output_column="calibrated_mz",
    )
    assert mz_calibration.ci(mz_df) == 0  # not fitted

    mz_calibration.fit(mz_df, plot=False)
    ci = mz_calibration.ci(mz_df, 0.95)
    assert ci > 0

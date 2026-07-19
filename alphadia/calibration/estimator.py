"""Calibration estimator module.

Thin Python wrapper around the Rust calibration estimator
(`alphadia_search_rs.CalibrationEstimator`). The numeric algorithm (LOESS fit,
prediction, deviation transform, metrics) lives in Rust; this wrapper only pulls
the relevant columns out of a dataframe, passes them as arrays, and keeps the
plotting on the Python side.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from alphadia_search_rs import CalibrationEstimator as _RustCalibrationEstimator

from alphadia.calibration.plot import plot_calibration


@dataclass
class CalibrationMetrics:
    """Fitted calibration quality metrics for a single estimator.

    Attributes
    ----------
    median_accuracy : float
        Median absolute calibrated deviation (systematic bias after calibration).

    median_precision : float
        Median absolute residual deviation (spread after calibration).

    """

    median_accuracy: float
    median_precision: float


class CalibrationEstimator:
    """A single estimator for a property (mz, rt, mobility).

    Calibration models the deviation of an input value (e.g. ``mz_library``) from
    an observed property (e.g. ``mz_observed``). Once fitted, calibrated values
    (e.g. ``mz_calibrated``) are predicted from the input values. The numeric LOESS
    model lives in the Rust backend; its two configuration values (``n_kernels`` and
    the deviation ``transform``) are passed in from here.
    """

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        name: str,
        n_kernels: int,
        transform_deviation: float | None,
        input_column: str,
        target_column: str,
        output_column: str,
    ):
        """Initialize the estimator.

        Parameters
        ----------
        name : str
            Name of the estimator for logging and plotting e.g. 'mz'.

        n_kernels : int
            Number of LOESS kernels used by the underlying model.

        transform_deviation : float | None
            If set, the deviation is expressed as a fraction of the input value
            (e.g. ``1e6`` for ppm). If None, the deviation is absolute.

        input_column : str
            Column used as input for the estimator e.g. 'mz_library'.

        target_column : str
            Column used as target for the estimator e.g. 'mz_observed'.

        output_column : str
            Column the calibrated prediction is written to e.g. 'mz_calibrated'.

        """
        self.name = name
        self._n_kernels = n_kernels
        self.transform_deviation = (
            float(transform_deviation) if transform_deviation is not None else None
        )
        # kept as a list for backwards-compatible access from the plotting code
        self.input_columns = [input_column]
        self._target_columns = [target_column]
        self._output_columns = [output_column]

        self._rust = _RustCalibrationEstimator(n_kernels, self.transform_deviation)

        self.is_fitted = False
        self.metrics: CalibrationMetrics | None = None

    def __repr__(self) -> str:
        """Return a string representation of the Calibration object."""
        return f"<Calibration {self.name}, is_fitted: {self.is_fitted}>"

    def _validate_columns(self, df: pd.DataFrame, required_columns: list[str]) -> bool:
        """Validate that the required columns are present in the dataframe."""
        required_columns_set = set(required_columns)
        if not required_columns_set.issubset(df.columns):
            logging.warning(
                f"{self.name}, at least one column {required_columns_set} not found in dataframe"
            )
            return False
        return True

    def _input_array(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.input_columns[0]].to_numpy(dtype=np.float32)

    def _target_array(self, df: pd.DataFrame) -> np.ndarray:
        return df[self._target_columns[0]].to_numpy(dtype=np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        *,
        plot: bool = True,
        figure_path: str | None = None,
    ) -> None:
        """Fit the estimator based on the input and target columns of the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the input and target columns.

        plot : bool, default=True
            If True, a plot of the calibration is generated.

        figure_path : str, default=None
            If not None, the plot is saved to the given path.

        """
        if not self._validate_columns(df, self.input_columns + self._target_columns):
            raise ValueError(
                f"{self.name} calibration fitting: failed input validation"
            )

        input_values = self._input_array(df)
        target_values = self._target_array(df)

        # a fresh Rust estimator is created for every fit (no reuse across runs)
        self._rust = _RustCalibrationEstimator(
            self._n_kernels, self.transform_deviation
        )
        try:
            self._rust.fit(input_values, target_values)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Could not fit estimator {self.name}: {e}")
            return

        self.is_fitted = True

        metrics = self._rust.metrics()
        self.metrics = (
            CalibrationMetrics(
                median_accuracy=float(metrics[0]),
                median_precision=float(metrics[1]),
            )
            if metrics is not None
            else None
        )

        if plot:
            plot_calibration(self, df, figure_path=figure_path)

    def predict(self, df: pd.DataFrame, *, inplace: bool = True) -> np.ndarray | None:
        """Predict calibrated values based on the input column of the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the input column.

        inplace : bool, default=True
            If True, the prediction is added as the output column of the dataframe.

        Returns
        -------
        np.ndarray | None
            Array of predictions, or None if written inplace / not fitted.

        """
        if not self.is_fitted:
            logging.warning(
                f"{self.name} prediction was skipped as it has not been fitted yet"
            )
            return None

        if not self._validate_columns(df, self.input_columns):
            raise ValueError(
                f"{self.name} calibration prediction: failed input validation"
            )

        predicted_values = np.asarray(
            self._rust.predict(self._input_array(df)), dtype=np.float64
        )

        if inplace:
            df[self._output_columns[0]] = predicted_values
            return None

        return predicted_values

    def calc_deviation(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate the observed, calibrated and residual deviations.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the input and target columns.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 3 + n_input_columns) with columns
            [observed deviation, calibrated deviation, residual deviation, input].

        """
        observed, calibrated, residual = self._rust.deviation(
            self._input_array(df), self._target_array(df)
        )
        input_values = df[self.input_columns].to_numpy()
        return np.concatenate(
            [
                np.asarray(observed)[:, np.newaxis],
                np.asarray(calibrated)[:, np.newaxis],
                np.asarray(residual)[:, np.newaxis],
                input_values,
            ],
            axis=1,
        )

    def ci(self, df: pd.DataFrame, ci: float = 0.95) -> float:
        """Calculate the residual deviation at the given confidence interval.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the input and target columns.

        ci : float, default=0.95
            confidence interval.

        Returns
        -------
        float
            the confidence interval of the residual deviation after calibration.

        """
        if not 0 < ci < 1:
            raise ValueError("Confidence interval must be between 0 and 1")

        if not self.is_fitted:
            return 0

        return float(self._rust.ci(self._input_array(df), self._target_array(df), ci))

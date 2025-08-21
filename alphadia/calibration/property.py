import logging
import pickle

import numpy as np
import pandas as pd
import sklearn.base
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from alphadia.calibration.models import LOESSRegression
from alphadia.calibration.plot import plot_calibration


class Calibration:
    def __init__(
        self,
        name: str,
        function: sklearn.base.BaseEstimator,
        input_columns: list[str],
        target_columns: list[str],
        output_columns: list[str],
        transform_deviation: None | float = None,
    ):
        """A single estimator for a property (mz, rt, etc.).

        Calibration is performed by modeling the deviation of an input values (e.g. mz_library) from
        an observed property (e.g. mz_observed) using a function (e.g. LinearRegression).
        Once calibrated, calibrated values (e.g. mz_calibrated) can be predicted from input values (e.g. mz_library).
        Additional explaining variables can be added to the input values (e.g. rt_library) to improve the calibration.

        Parameters
        ----------
        name : str
            Name of the estimator for logging and plotting e.g. 'mz'

        function : sklearn.base.BaseEstimator
            The estimator object instance which must have a fit and predict method.
            This will usually be a sklearn estimator or a custom estimator.

        input_columns : list of str
            The columns of the dataframe that are used as input for the estimator e.g. ['mz_library'].
            The first column is the property which should be calibrated, additional columns can be used as explaining variables e.g. ['mz_library', 'rt_library'].

        target_columns : list of str
            The columns of the dataframe that are used as target for the estimator e.g. ['mz_observed'].
            At the moment only one target column is supported.

        output_columns : list of str
            The columns of the dataframe that are used as output for the estimator e.g. ['mz_calibrated'].
            At the moment only one output column is supported.

        transform_deviation : typing.List[Union[None, float]]
            If set to a valid float, the deviation is expressed as a fraction of the input value e.g. 1e6 for ppm.
            If set to None, the deviation is expressed in absolute units.

        """
        self.name = name
        self.function = function
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.output_columns = output_columns
        self.transform_deviation = (
            float(transform_deviation) if transform_deviation is not None else None
        )

        self.is_fitted = False
        self.metrics = None

        if len(output_columns) != 1 or len(target_columns) != 1:
            raise ValueError(
                f"{self.name} calibration: only one output and target column is supported, got {len(output_columns)=} {len(target_columns)=}"
            )

    def __repr__(self) -> str:
        return f"<Calibration {self.name}, is_fitted: {self.is_fitted}>"

    def save(self, file_name: str):
        """Save the estimator to pickle file.

        Parameters
        ----------

        file_name : str
            Path to the pickle file

        """

        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_name: str):
        """Load the estimator from pickle file.

        Parameters
        ----------

        file_name : str
            Path to the pickle file

        """

        with open(file_name, "rb") as f:
            loaded_calibration = pickle.load(f)

        new_calibration = Calibration(
            name=loaded_calibration.name,
            function=loaded_calibration.function,
            input_columns=loaded_calibration.input_columns,
            target_columns=loaded_calibration.target_columns,
            output_columns=loaded_calibration.output_columns,
            transform_deviation=loaded_calibration.transform_deviation,
        )
        new_calibration.__dict__.update(loaded_calibration.__dict__)
        return new_calibration

    def _validate_columns(self, dataframe: pd.DataFrame, required_columns: list[str]):
        """Validate that the input and target columns are present in the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns
        required_columns : list[str]
            List of required columns to check in the dataframe

        Returns
        -------
        bool
            True if all columns are present, False otherwise

        """

        valid = True

        if len(self.target_columns) > 1:
            logging.warning("Only one target column supported")
            valid = False

        required_columns = set(required_columns)
        if not required_columns.issubset(dataframe.columns):
            logging.warning(
                f"{self.name}, at least one column {required_columns} not found in dataframe"
            )
            valid = False

        return valid

    def fit(
        self,
        dataframe: pd.DataFrame,
        plot: bool = True,
        figure_path: str | None = None,
    ):
        """Fit the estimator based on the input and target columns of the dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        plot : bool, default=True
            If True, a plot of the calibration is generated.

        figure_path : str, default=None
            If not None, a plot of the calibration is generated and saved.

        Returns
        -------

        np.ndarray
            Array of shape (n_input_columns, ) containing the mean absolute deviation of the residual deviation at the given confidence interval

        """

        if not self._validate_columns(
            dataframe, self.input_columns + self.target_columns
        ):
            raise ValueError(f"{self.name} calibration: failed input validation")

        input_values = dataframe[self.input_columns].values
        target_value = dataframe[self.target_columns].values

        try:
            self.function.fit(input_values, target_value)
            self.is_fitted = True
        except Exception as e:
            logging.exception(f"Could not fit estimator {self.name}: {e}")
            raise e

        self.metrics = self._get_metrics(dataframe)

        if plot:
            plot_calibration(self, dataframe, figure_path=figure_path)

    def predict(
        self, dataframe: pd.DataFrame, inplace: bool = True
    ) -> np.ndarray | None:
        """Perform a prediction based on the input columns of the dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe containing the input and target columns

        inplace : bool, default=True
            If True, the prediction is added as a new column to the dataframe.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, ) containing the prediction

        """

        if not self.is_fitted:
            logging.warning(
                f"{self.name} prediction was skipped as it has not been fitted yet"
            )
            return None

        if not self._validate_columns(dataframe, self.input_columns):
            raise ValueError(f"{self.name} calibration: failed input validation")

        input_values = dataframe[self.input_columns].values
        predicted_values = self.function.predict(input_values)

        if inplace:
            dataframe[self.output_columns[0]] = predicted_values
        else:
            return predicted_values

        return None

    def calc_deviation(self, dataframe: pd.DataFrame):
        """Calculate the deviations between the input, target and calibrated values.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 3 + n_input_columns).
            The second dimension contains the observed deviation, calibrated deviation, residual deviation and the input values.

        """

        # the first column is the unclaibrated input property
        # all other columns are explaining variables
        input_values = dataframe[self.input_columns].values

        # the first column is the unclaibrated input property
        uncalibrated_values = input_values[:, [0]]

        # only one target column is supported
        target_values = dataframe[self.target_columns].values[:, [0]]
        input_transform = self.transform_deviation

        calibrated_values = self.predict(dataframe, inplace=False)
        if calibrated_values.ndim == 1:
            calibrated_values = calibrated_values[:, np.newaxis]

        # only one output column is supported
        calibrated_dim = calibrated_values[:, [0]]

        # deviation is the difference between the (observed) target value and the uncalibrated input value
        observed_deviation = target_values - uncalibrated_values
        if input_transform is not None:
            observed_deviation = (
                observed_deviation / uncalibrated_values * float(input_transform)
            )

        # calibrated deviation is the explained difference between the (calibrated) target value and the uncalibrated input value
        calibrated_deviation = calibrated_dim - uncalibrated_values
        if input_transform is not None:
            calibrated_deviation = (
                calibrated_deviation / uncalibrated_values * float(input_transform)
            )

        # residual deviation is the unexplained difference between the (observed) target value and the (calibrated) target value
        residual_deviation = observed_deviation - calibrated_deviation

        return np.concatenate(
            [
                observed_deviation,
                calibrated_deviation,
                residual_deviation,
                input_values,
            ],
            axis=1,
        )

    def _get_metrics(self, dataframe: pd.DataFrame) -> dict[str, float]:
        deviation = self.calc_deviation(dataframe)
        return {
            "median_accuracy": np.median(np.abs(deviation[:, 1])),
            "median_precision": np.median(np.abs(deviation[:, 2])),
        }

    def ci(self, dataframe, ci: float = 0.95):
        """Calculate the residual deviation at the given confidence interval.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        ci : float, default=0.95
            confidence interval

        Returns
        -------

        float
            the confidence interval of the residual deviation after calibration
        """

        if not 0 < ci < 1:
            raise ValueError("Confidence interval must be between 0 and 1")

        if not self.is_fitted:
            return 0

        ci_percentile = [100 * (1 - ci) / 2, 100 * (1 + ci) / 2]

        deviation = self.calc_deviation(dataframe)
        residual_deviation = deviation[:, 2]
        return np.mean(np.abs(np.percentile(residual_deviation, ci_percentile)))


class CalibrationModelProvider:
    def __init__(self):
        """Provides a collection of scikit-learn compatible models for calibration."""
        self.model_dict = {}

    def __repr__(self) -> str:
        string = "<CalibrationModelProvider, \n[\n"
        for key, value in self.model_dict.items():
            string += f" \t {key}: {value}\n"
        string += "]>"
        return string

    def register_model(
        self, model_name: str, model_template: type[sklearn.base.BaseEstimator]
    ):
        """Register a model template with a given name.

        Parameters
        ----------
        model_name : str
            Name of the model

        model_template : type[sklearn.base.BaseEstimator]
            The model template which must have a fit and predict method.

        """
        self.model_dict[model_name] = model_template

    def get_model(self, model_name: str) -> type[sklearn.base.BaseEstimator]:
        """Get a model template by name.

        Parameters
        ----------

        model_name : str
            Name of the model

        Returns
        -------
        type[sklearn.base.BaseEstimator]
            The model template which must have a fit and predict method.

        """

        if model_name not in self.model_dict:
            raise ValueError(f"Unknown model {model_name}")
        else:
            return self.model_dict[model_name]


def PolynomialRegression(degree=2, include_bias=False):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("linear", LinearRegression()),
        ]
    )


calibration_model_provider = CalibrationModelProvider()
calibration_model_provider.register_model("LinearRegression", LinearRegression)
calibration_model_provider.register_model("LOESSRegression", LOESSRegression)
calibration_model_provider.register_model("PolynomialRegression", PolynomialRegression)

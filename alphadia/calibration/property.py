# native imports
import logging
import pickle

import numpy as np

# third party imports
import pandas as pd
import sklearn.base
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from alphadia.calibration.models import LOESSRegression

# alphadia imports
from alphadia.plotting.utils import density_scatter


class Calibration:
    def __init__(
        self,
        name: str = "",
        function: object = None,
        input_columns: list[str] | None = None,
        target_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        transform_deviation: None | float = None,
        **kwargs,
    ):
        """A single estimator for a property (mz, rt, etc.).

        Calibration is performed by modeling the deviation of an input values (e.g. mz_library) from an observed property (e.g. mz_observed) using a function (e.g. LinearRegression). Once calibrated, calibrated values (e.g. mz_calibrated) can be predicted from input values (e.g. mz_library). Additional explaining variabels can be added to the input values (e.g. rt_library) to improve the calibration.

        Parameters
        ----------

        name : str
            Name of the estimator for logging and plotting e.g. 'mz'

        function : object
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
        if output_columns is None:
            output_columns = []
        if target_columns is None:
            target_columns = []
        if input_columns is None:
            input_columns = []
        self.name = name
        self.function = function
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.output_columns = output_columns
        self.transform_deviation = (
            float(transform_deviation) if transform_deviation is not None else None
        )
        self.is_fitted = False

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

    def load(self, file_name: str):
        """Load the estimator from pickle file.

        Parameters
        ----------

        file_name : str
            Path to the pickle file

        """

        with open(file_name, "rb") as f:
            loaded_calibration = pickle.load(f)
            self.__dict__.update(loaded_calibration.__dict__)

    def validate_columns(self, dataframe: pd.DataFrame):
        """Validate that the input and target columns are present in the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        Returns
        -------
        bool
            True if all columns are present, False otherwise

        """

        valid = True

        if len(self.target_columns) > 1:
            logging.warning("Only one target column supported")
            valid = False

        required_columns = set(self.input_columns + self.target_columns)
        if not required_columns.issubset(dataframe.columns):
            logging.warning(
                f"{self.name}, at least one column {required_columns} not found in dataframe"
            )
            valid = False

        return valid

    def fit(self, dataframe: pd.DataFrame, plot: bool = False, **kwargs):
        """Fit the estimator based on the input and target columns of the dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        plot : bool, default=False
            If True, a plot of the calibration is generated.

        Returns
        -------

        np.ndarray
            Array of shape (n_input_columns, ) containing the mean absolute deviation of the residual deviation at the given confidence interval

        """

        if not self.validate_columns(dataframe):
            logging.warning(f"{self.name} calibration was skipped")
            return

        if self.function is None:
            raise ValueError("No estimator function provided")

        input_values = dataframe[self.input_columns].values
        target_value = dataframe[self.target_columns].values

        try:
            self.function.fit(input_values, target_value)
            self.is_fitted = True
        except Exception as e:
            logging.error(f"Could not fit estimator {self.name}: {e}")
            return

        if plot is True:
            self.plot(dataframe, **kwargs)

    def predict(self, dataframe, inplace=True):
        """Perform a prediction based on the input columns of the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        inplace : bool, default=True
            If True, the prediction is added as a new column to the dataframe. If False, the prediction is returned as a numpy array.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, ) containing the prediction

        """

        if self.is_fitted is False:
            logging.warning(
                f"{self.name} prediction was skipped as it has not been fitted yet"
            )
            return

        if not set(self.input_columns).issubset(dataframe.columns):
            logging.warning(
                f"{self.name} calibration was skipped as input column {self.input_columns} not found in dataframe"
            )
            return

        input_values = dataframe[self.input_columns].values

        if inplace:
            dataframe[self.output_columns[0]] = self.function.predict(input_values)
        else:
            return self.function.predict(input_values)

    def fit_predict(
        self, dataframe: pd.DataFrame, plot: bool = False, inplace: bool = True
    ):
        """Fit the estimator and perform a prediction based on the input columns of the dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        plot : bool, default=False
            If True, a plot of the calibration is generated.

        inplace : bool, default=True
            If True, the prediction is added as a new column to the dataframe. If False, the prediction is returned as a numpy array.

        """
        self.fit(dataframe, plot=plot)
        return self.predict(dataframe, inplace=inplace)

    def deviation(self, dataframe: pd.DataFrame):
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

        deviation = self.deviation(dataframe)
        residual_deviation = deviation[:, 2]
        return np.mean(np.abs(np.percentile(residual_deviation, ci_percentile)))

    def get_transform_unit(self, transform_deviation: None | float):
        """Get the unit of the deviation based on the transform deviation.

        Parameters
        ----------

        transform_deviation : typing.Union[None, float]
            If set to a valid float, the deviation is expressed as a fraction of the input value e.g. 1e6 for ppm.

        Returns
        -------
        str
            The unit of the deviation

        """
        if transform_deviation is not None:
            if np.isclose(transform_deviation, 1e6):
                return "(ppm)"
            elif np.isclose(transform_deviation, 1e2):
                return "(%)"
            else:
                return f"({transform_deviation})"
        else:
            return "(absolute)"

    def plot(
        self,
        dataframe: pd.DataFrame,
        figure_path: str = None,
        # neptune_run : str = None,
        # neptune_key :str = None,
        **kwargs,
    ):
        """Plot the data and calibration model.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        figure_path : str, default=None
            If set, the figure is saved to the given path.

        neptune_run : str, default=None
            If set, the figure is logged to the given neptune run.

        neptune_key : str, default=None
            key under which the figure is logged to the neptune run.

        """

        deviation = self.deviation(dataframe)

        n_input_properties = deviation.shape[1] - 3

        transform_unit = self.get_transform_unit(self.transform_deviation)

        fig, axs = plt.subplots(
            n_input_properties,
            2,
            figsize=(6.5, 3.5 * n_input_properties),
            squeeze=False,
        )

        for input_property in range(n_input_properties):
            # plot the relative observed deviation
            density_scatter(
                deviation[:, 3 + input_property],
                deviation[:, 0],
                axis=axs[input_property, 0],
                s=1,
            )

            # plot the calibration model
            x_values = deviation[:, 3 + input_property]
            y_values = deviation[:, 1]
            order = np.argsort(x_values)
            x_values = x_values[order]
            y_values = y_values[order]

            axs[input_property, 0].plot(x_values, y_values, color="red")

            # plot the calibrated deviation

            density_scatter(
                deviation[:, 3 + input_property],
                deviation[:, 2],
                axis=axs[input_property, 1],
                s=1,
            )

            for ax, dim in zip(axs[input_property, :], [0, 2], strict=True):
                ax.set_xlabel(self.input_columns[input_property])
                ax.set_ylabel(f"observed deviation {transform_unit}")

                # get absolute y value and set limites to plus minus absolute y
                y = deviation[:, dim]
                y_abs = np.abs(y)
                ax.set_ylim(-y_abs.max() * 1.05, y_abs.max() * 1.05)

        fig.tight_layout()

        # log figure to neptune ai
        # if neptune_run is not None and neptune_key is not None:
        #    neptune_run[f'calibration/{neptune_key}'].log(fig)

        # if figure_path is not None:

        #    i = 0
        #    file_name = os.path.join(figure_path, f'calibration_{neptune_key}_{i}.png')
        #    while os.path.exists(file_name):
        #        file_name = os.path.join(figure_path, f'calibration_{neptune_key}_{i}.png')
        #        i += 1

        #    fig.savefig(file_name)

        plt.show()

        plt.close()


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
        self, model_name: str, model_template: sklearn.base.BaseEstimator
    ):
        """Register a model template with a given name.

        Parameters
        ----------
        model_name : str
            Name of the model

        model_template : sklearn.base.BaseEstimator
            The model template which must have a fit and predict method.

        """
        self.model_dict[model_name] = model_template

    def get_model(self, model_name: str):
        """Get a model template by name.

        Parameters
        ----------

        model_name : str
            Name of the model

        Returns
        -------
        sklearn.base.BaseEstimator
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

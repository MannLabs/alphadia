import logging

import pandas as pd

from alphadia.calibration.estimator import (
    CalibrationEstimator,
    calibration_model_provider,
)
from alphadia.constants.keys import CalibCols, ConstantsClass
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


EstimatorGroups = dict[str, dict[str, CalibrationEstimator]]
CalibrationConfig = dict[str, dict[str, dict[str, str | int | list[str]]]]


class CalibrationGroups(metaclass=ConstantsClass):
    """String constants for calibration groups."""

    FRAGMENT = "fragment"
    PRECURSOR = "precursor"


class CalibrationEstimators(metaclass=ConstantsClass):
    """String constants for calibration estimators."""

    MZ = "mz"
    RT = "rt"
    MOBILITY = "mobility"


# Configuration for the calibration manager.
# Note: The mapping to which columns to actually use it currently done in ColumnNameHandler. # TODO: rethink this coupling
CALIBRATION_GROUPS_CONFIG: CalibrationConfig = {
    CalibrationGroups.FRAGMENT: {
        CalibrationEstimators.MZ: {
            "input_columns": [CalibCols.MZ_LIBRARY],
            "target_columns": [CalibCols.MZ_OBSERVED],
            "output_columns": [CalibCols.MZ_CALIBRATED],
            "model": "LOESSRegression",
            "model_args": {"n_kernels": 2},
            "transform_deviation": "1e6",
        }
    },
    CalibrationGroups.PRECURSOR: {
        CalibrationEstimators.MZ: {
            "input_columns": [CalibCols.MZ_LIBRARY],
            "target_columns": [CalibCols.MZ_OBSERVED],
            "output_columns": [CalibCols.MZ_CALIBRATED],
            "model": "LOESSRegression",
            "model_args": {"n_kernels": 2},
            "transform_deviation": "1e6",
        },
        CalibrationEstimators.RT: {
            "input_columns": [CalibCols.RT_LIBRARY],
            "target_columns": [CalibCols.RT_OBSERVED],
            "output_columns": [CalibCols.RT_CALIBRATED],
            "model": "LOESSRegression",
            "model_args": {"n_kernels": 6},
        },
        CalibrationEstimators.MOBILITY: {
            "input_columns": [CalibCols.MOBILITY_LIBRARY],
            "target_columns": [CalibCols.MOBILITY_OBSERVED],
            "output_columns": [CalibCols.MOBILITY_CALIBRATED],
            "model": "LOESSRegression",
            "model_args": {"n_kernels": 2},
        },
    },
}


class CalibrationManager(BaseManager):
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
        has_ms1: bool = True,
        has_mobility: bool = True,
        **kwargs,
    ):
        """Contains, updates and applies all calibrations for a single run.

        Calibrations are grouped into calibration groups. Each calibration group is applied to a single data structure (precursor dataframe, fragment fataframe, etc.). Each calibration group contains multiple estimators which each calibrate a single property (mz, rt, etc.). Each estimator is a `Calibration` object which contains the estimator function.

        Parameters
        ----------
        path : str, default=None
            Path where the current parameter set is saved to and loaded from.

        load_from_file : bool, default=True
            If True, the manager will be loaded from file if it exists.

        has_ms1 : bool, default=True
            If True, the calibration manager will include MS1 calibration. This will include an MS1 estimator in the precursor group.

        has_mobility : bool, default=True
            If True, the calibration manager will include mobility calibration. This will include a mobility estimator in the precursor group.

        kwargs :
             Will be passed to the parent class `BaseManager`, need to be valid keyword arguments.

        """

        super().__init__(path=path, load_from_file=load_from_file, **kwargs)

        self._has_mobility = has_mobility
        self._has_ms1 = has_ms1

        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            self.all_fitted = False
            self.estimator_groups: EstimatorGroups = self.setup_estimator_groups(
                CALIBRATION_GROUPS_CONFIG
            )

    @property
    def estimator_groups(self) -> EstimatorGroups:
        """List of calibration groups."""
        return self._estimator_groups

    @estimator_groups.setter
    def estimator_groups(self, value: EstimatorGroups):
        self._estimator_groups = value

    def setup_estimator_groups(self, calibration_config: CalibrationConfig):
        """Load calibration configuration.

        Each calibration config is a list of calibration groups which consist of multiple estimators.
        For each estimator the `model` and `model_args` are used to request a model from the calibration_model_provider and to initialize it.
        The estimator is then initialized with the `Calibration` class and added to the group.

        Parameters
        ----------
        calibration_config : CalibrationConfig
            Calibration configuration

        Example
        -------
        Create a calibration manager with a single group and a single estimator:

        .. code-block:: python

            calibration_manager = CalibrationManager()
            calibration_manager.load_config({
                'mz_calibration': [
                    {
                        'name': 'mz',
                        'model': 'LOESSRegression',
                        'model_args': { 'n_kernels': 2 },
                        'input_columns': [CalibCols.MZ_LIBRARY],
                        'target_columns': [CalibCols.MZ_OBSERVED],
                        'output_columns': [CalibCols.MZ_CALIBRATED],
                        'transform_deviation': 1e6
                    }
                ]
            })

        """
        self.reporter.log_string("Setting up calibration estimators ..")

        estimator_groups: EstimatorGroups = {}
        for group_name, estimators_params_in_group in calibration_config.items():
            self.reporter.log_string(
                f"Found {len(estimators_params_in_group)} estimator(s) in calibration group '{group_name}'"
            )

            initialized_estimators: dict[str, CalibrationEstimator] = {}
            for estimator_name, estimator_params in estimators_params_in_group.items():
                if (
                    not self._has_mobility
                    and estimator_name == CalibrationEstimators.MOBILITY
                ):
                    self.reporter.log_string(
                        f"Skipping estimator '{CalibrationEstimators.MOBILITY}' in group '{group_name}' as it is not available in the raw data",
                    )
                    continue

                if (
                    not self._has_ms1
                    and group_name == CalibrationGroups.PRECURSOR
                    and estimator_name == CalibrationEstimators.MZ
                ):
                    self.reporter.log_string(
                        f"Skipping estimator '{CalibrationEstimators.MZ}' in group '{group_name}' as it is not available in the raw data",
                    )
                    continue

                model = calibration_model_provider.get_model(estimator_params["model"])
                model_args = estimator_params.get("model_args", {})

                self.reporter.log_string(
                    f"Initializing estimator '{estimator_name}' in group '{group_name}' with '{estimator_params}' .."
                )
                initialized_estimators[estimator_name] = CalibrationEstimator(
                    name=estimator_name,
                    model=model(**model_args),
                    input_columns=estimator_params["input_columns"],
                    target_columns=estimator_params["target_columns"],
                    output_columns=estimator_params["output_columns"],
                    transform_deviation=estimator_params.get(
                        "transform_deviation", None
                    ),
                )

            estimator_groups[group_name] = initialized_estimators

        self.reporter.log_string("Done setting up calibration estimators.")

        return estimator_groups

    def get_estimator(
        self, group_name: str, estimator_name: str
    ) -> CalibrationEstimator | None:
        """Get an estimator from a calibration group.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        estimator_name : str
            Name of the estimator

        Returns
        -------
        CalibrationEstimator | None
            The estimator object or None if not found

        """
        try:
            return self.estimator_groups[group_name][estimator_name]
        except KeyError:
            return None

    def fit(
        self,
        df: pd.DataFrame,
        group_name: str,
        plot: bool = True,
        figure_path: None | str = None,
    ):
        """Fit all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        plot: bool, default=True
            If True, a plot of the calibration is generated.

        figure_path: str, default=None
            If set, the generated plot is saved to the given path.

        """

        # only iterate over the first group with the given name
        for estimator in self.estimator_groups[group_name].values():
            self.reporter.log_string(
                f"Fitting estimator '{estimator.name}' in calibration group '{group_name}' .."
            )
            estimator.fit(df, plot=plot, figure_path=figure_path)

        all_fitted = True
        for group in self.estimator_groups.values():
            for estimator in group.values():
                all_fitted &= estimator.is_fitted
        self.all_fitted = all_fitted

    def predict(self, df: pd.DataFrame, group_name: str):
        """Predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        """

        for estimator in self.estimator_groups[group_name].values():
            self.reporter.log_string(
                f"Predicting estimator '{estimator.name}' in calibration group '{group_name}' .."
            )
            estimator.predict(df, inplace=True)

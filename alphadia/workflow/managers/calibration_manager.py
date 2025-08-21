import logging

import pandas as pd

from alphadia.calibration.property import Calibration, calibration_model_provider
from alphadia.constants.keys import MRMCols
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


CalibrationConfig: type = list[
    dict[str, str | list[dict[str, list[str] | str | dict[str, str | int | list[str]]]]]
]


class CalibrationGroups:
    """String constants for calibration groups."""

    FRAGMENT = "fragment"
    PRECURSOR = "precursor"


class CalibrationEstimators:
    """String constants for calibration estimators."""

    MZ = "mz"
    RT = "rt"
    MOBILITY = "mobility"


# configuration for the calibration manager
# the config has to start with the calibration keyword and consists of a list of calibration groups.
# each group consists of datapoints which have multiple properties.
# This can be for example precursors (mz, rt ...), fragments (mz, ...), quadrupole (transfer_efficiency)
# TODO simplify this structure and the config loading
CALIBRATION_MANAGER_CONFIG: CalibrationConfig = [
    {
        "name": CalibrationGroups.FRAGMENT,
        "estimators": [
            {
                "input_columns": [MRMCols.MZ_LIBRARY],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 2},
                "name": CalibrationEstimators.MZ,
                "output_columns": [MRMCols.MZ_CALIBRATED],
                "target_columns": [MRMCols.MZ_OBSERVED],
                "transform_deviation": "1e6",
            }
        ],
    },
    {
        "name": CalibrationGroups.PRECURSOR,
        "estimators": [
            {
                "input_columns": [MRMCols.MZ_LIBRARY],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 2},
                "name": CalibrationEstimators.MZ,
                "output_columns": [MRMCols.MZ_CALIBRATED],
                "target_columns": [MRMCols.MZ_OBSERVED],
                "transform_deviation": "1e6",
            },
            {
                "input_columns": [MRMCols.RT_LIBRARY],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 6},
                "name": CalibrationEstimators.RT,
                "output_columns": [MRMCols.RT_CALIBRATED],
                "target_columns": [MRMCols.RT_OBSERVED],
            },
            {
                "input_columns": [MRMCols.MOBILITY_LIBRARY],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 2},
                "name": CalibrationEstimators.MOBILITY,
                "output_columns": [MRMCols.MOBILITY_CALIBRATED],
                "target_columns": [MRMCols.MOBILITY_OBSERVED],
            },
        ],
    },
]


class CalibrationManager(BaseManager):
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
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

        has_mobility : bool, default=True
            If True, the calibration manager will include mobility calibration. This will add a mobility estimator to the precursor group.

        kwargs :
             Will be passed to the parent class `BaseManager`, need to be valid keyword arguments.

        """

        super().__init__(path=path, load_from_file=load_from_file, **kwargs)

        self._has_mobility = has_mobility

        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            self.all_fitted = False
            self.estimator_groups = []
            self.load_config(CALIBRATION_MANAGER_CONFIG)

    @property
    def estimator_groups(self) -> list[dict[str, str | list[Calibration]]]:
        """List of calibration groups."""
        return self._estimator_groups

    @estimator_groups.setter
    def estimator_groups(self, value: list[dict[str, str | list[Calibration]]]):
        self._estimator_groups = value

    def load_config(self, calibration_config: CalibrationConfig):
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
            calibration_manager.load_config([{
                'name': 'mz_calibration',
                'estimators': [
                    {
                        'name': 'mz',
                        'model': 'LOESSRegression',
                        'model_args': { 'n_kernels': 2 },
                        'input_columns': ['MRMCols.MZ_LIBRARY'],
                        'target_columns': ['MRMCols.MZ_OBSERVED'],
                        'output_columns': ['MRMCols.MZ_CALIBRATED'],
                        'transform_deviation': 1e6
                    }
                ]
            }])

        """
        self.reporter.log_string(
            f"Loading calibration configuration: {calibration_config}"
        )

        for group in calibration_config:
            group_name = group["name"]
            self.reporter.log_string(
                f'Calibration group: {group_name}, found {len(group["estimators"])} estimator(s)'
            )

            initialized_estimators = []
            for estimator in group["estimators"]:
                if (
                    not self._has_mobility
                    and estimator["name"] == CalibrationEstimators.MOBILITY
                ):
                    self.reporter.log_string(
                        f"Skipping estimator '{CalibrationEstimators.MOBILITY}' in group '{group_name}' as it is not available in the raw data",
                    )
                    continue

                model = calibration_model_provider.get_model(estimator["model"])
                model_args = estimator.get("model_args", {})

                initialized_estimators.append(
                    Calibration(
                        name=estimator["name"],
                        function=model(**model_args),
                        input_columns=estimator["input_columns"],
                        target_columns=estimator["target_columns"],
                        output_columns=estimator["output_columns"],
                        transform_deviation=estimator.get("transform_deviation", None),
                    )
                )

            self.estimator_groups.append(
                {"name": group_name, "estimators": initialized_estimators}
            )

    def get_group_names(self):
        """Get the names of all calibration groups.

        Returns
        -------
        list of str
            List of calibration group names

        """

        return [x["name"] for x in self.estimator_groups]

    def get_group(self, group_name: str):
        """Get the calibration group by name.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        Returns
        -------
        dict
            Calibration group dict with `name` and `estimators` keys\

        """
        for group in self.estimator_groups:
            if group["name"] == group_name:
                return group

        raise ValueError(f"Could not get {group_name} from {self.estimator_groups}")

    def get_estimator_names(self, group_name: str):
        """Get the names of all estimators in a calibration group.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        Returns
        -------
        list of str
            List of estimator names
        """

        group = self.get_group(group_name)
        return [x.name for x in group["estimators"]]

    def get_estimator(self, group_name: str, estimator_name: str):
        """Get an estimator from a calibration group.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        estimator_name : str
            Name of the estimator

        Returns
        -------
        Calibration
            The estimator object

        """
        group = self.get_group(group_name)
        for estimator in group["estimators"]:
            if estimator.name == estimator_name:
                return estimator

        raise ValueError(
            f"could not get estimator {estimator_name} for group {group_name} from  {group['estimators']}"
        )

    def fit(
        self,
        df: pd.DataFrame,
        group_name: str,
        skip: list | None = None,
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

        skip: list[str], default=None
            List of estimator names (e.g. "mz") to skip during fitting. If None, no estimators are skipped.

        plot: bool, default=True
            If True, a plot of the calibration is generated.

        figure_path: str, default=None
            If set, the generated plot is saved to the given path.

        """

        if skip is None:
            skip = []

        group_indices = self._get_indices_for_group_name(group_name)

        # only iterate over the first group with the given name
        for group_idx in group_indices:
            for estimator in self.estimator_groups[group_idx]["estimators"]:
                if estimator.name in skip:
                    continue
                self.reporter.log_string(
                    f"calibration group: {group_name}, fitting {estimator.name} estimator.."
                )
                estimator.fit(df, plot=plot, figure_path=figure_path)

        all_fitted = True
        for group in self.estimator_groups:
            for estimator in group["estimators"]:
                all_fitted &= estimator.is_fitted
        self.all_fitted = all_fitted

    def _get_indices_for_group_name(self, group_name: str) -> list[int]:
        """Get the indices of the calibration group with the given name."""

        if len(self.estimator_groups) == 0:
            raise ValueError("No estimators defined")

        group_idx = [
            i
            for i, group in enumerate(self.estimator_groups)
            if group["name"] == group_name
        ]

        if len(group_idx) == 0:
            raise ValueError(f"No group named {group_name} found")

        return group_idx

    def predict(self, df: pd.DataFrame, group_name: str):
        """Predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        """

        group_indices = self._get_indices_for_group_name(group_name)

        for group_idx in group_indices:
            for estimator in self.estimator_groups[group_idx]["estimators"]:
                self.reporter.log_string(
                    f"calibration group: {group_name}, predicting {estimator.name}"
                )
                estimator.predict(df, inplace=True)  # noqa: B026 Star-arg unpacking after a keyword argument is strongly discouraged

    def fit_predict(
        self,
        df: pd.DataFrame,
        group_name: str,
        plot: bool = True,
    ):
        """Fit and predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        plot : bool, default=True
            If True, a plot of the calibration is generated.

        """
        self.fit(df, group_name, plot=plot)
        self.predict(df, group_name)

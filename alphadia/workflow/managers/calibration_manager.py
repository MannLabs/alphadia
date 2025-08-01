import logging

import pandas as pd

from alphadia.calibration.property import Calibration, calibration_model_provider
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


# configuration for the calibration manager
# the config has to start with the calibration keyword and consists of a list of calibration groups.
# each group consists of datapoints which have multiple properties.
# This can be for example precursors (mz, rt ...), fragments (mz, ...), quadrupole (transfer_efficiency)
# TODO simplify this structure and the config loading
CALIBRATION_MANAGER_CONFIG = [
    {
        "estimators": [
            {
                "input_columns": ["mz_library"],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 2},
                "name": "mz",
                "output_columns": ["mz_calibrated"],
                "target_columns": ["mz_observed"],
                "transform_deviation": "1e6",
            }
        ],
        "name": "fragment",
    },
    {
        "estimators": [
            {
                "input_columns": ["mz_library"],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 2},
                "name": "mz",
                "output_columns": ["mz_calibrated"],
                "target_columns": ["mz_observed"],
                "transform_deviation": "1e6",
            },
            {
                "input_columns": ["rt_library"],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 6},
                "name": "rt",
                "output_columns": ["rt_calibrated"],
                "target_columns": ["rt_observed"],
            },
            {
                "input_columns": ["mobility_library"],
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 2},
                "name": "mobility",
                "output_columns": ["mobility_calibrated"],
                "target_columns": ["mobility_observed"],
            },
        ],
        "name": "precursor",
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
            self.is_fitted = False
            self.estimator_groups = []
            self.load_config(CALIBRATION_MANAGER_CONFIG)

    @property
    def estimator_groups(self):
        """List of calibration groups."""
        return self._estimator_groups

    @estimator_groups.setter
    def estimator_groups(self, value):
        self._estimator_groups = value

    def load_config(self, config: dict):
        """Load calibration config from config Dict.

        each calibration config is a list of calibration groups which consist of multiple estimators.
        For each estimator the `model` and `model_args` are used to request a model from the calibration_model_provider and to initialize it.
        The estimator is then initialized with the `Calibration` class and added to the group.

        Parameters
        ----------

        config : dict
            Calibration config dict

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
                        'model_args': {
                            'n_kernels': 2
                        },
                        'input_columns': ['mz_library'],
                        'target_columns': ['mz_observed'],
                        'output_columns': ['mz_calibrated'],
                        'transform_deviation': 1e6
                    },

                ]
            }])

        """
        self.reporter.log_string("Loading calibration config")
        self.reporter.log_string(f"Calibration config: {config}")
        for group in config:
            self.reporter.log_string(
                f'Calibration group :{group["name"]}, found {len(group["estimators"])} estimator(s)'
            )
            for estimator in group["estimators"]:
                if not self._has_mobility and estimator["name"] == "mobility":
                    self.reporter.log_string(
                        f'Skipping mobility estimator in group {group["name"]} as mobility is not available',
                    )
                    group["estimators"].remove(estimator)
                    continue
                try:
                    template = calibration_model_provider.get_model(estimator["model"])
                    model_args = estimator.get("model_args", {})
                    estimator["function"] = template(**model_args)
                except Exception as e:
                    self.reporter.log_string(
                        f'Could not load estimator {estimator["name"]}: {e}',
                        verbosity="error",
                    )

            group_copy = {"name": group["name"]}
            group_copy["estimators"] = [Calibration(**x) for x in group["estimators"]]
            self.estimator_groups.append(group_copy)

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

        self.reporter.log_string(
            f"could not get_group: {group_name}", verbosity="error"
        )
        return None

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
        if group is not None:
            return [x.name for x in group["estimators"]]
        self.reporter.log_string(
            f"could not get_estimator_names: {group_name}", verbosity="error"
        )
        return None

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
        if group is not None:
            for estimator in group["estimators"]:
                if estimator.name == estimator_name:
                    return estimator

        self.reporter.log_string(
            f"could not get_estimator: {group_name}, {estimator_name}",
            verbosity="error",
        )
        return None

    def fit(
        self,
        df: pd.DataFrame,
        group_name: str,
        skip: list | None = None,
        *args,
        **kwargs,
    ):
        """Fit all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        skip: TODO

        """

        if skip is None:
            skip = []
        if len(self.estimator_groups) == 0:
            raise ValueError("No estimators defined")

        group_idx = [
            i for i, x in enumerate(self.estimator_groups) if x["name"] == group_name
        ]
        if len(group_idx) == 0:
            raise ValueError(f"No group named {group_name} found")

        # only iterate over the first group with the given name
        for group in group_idx:
            for estimator in self.estimator_groups[group]["estimators"]:
                if estimator.name in skip:
                    continue
                self.reporter.log_string(
                    f"calibration group: {group_name}, fitting {estimator.name} estimator "
                )
                estimator.fit(df, *args, **kwargs)

        is_fitted = True
        # check if all estimators are fitted
        for group in self.estimator_groups:
            for estimator in group["estimators"]:
                is_fitted = is_fitted and estimator.is_fitted

        self.is_fitted = is_fitted and len(self.estimator_groups) > 0

    def predict(self, df: pd.DataFrame, group_name: str, *args, **kwargs):
        """Predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        """

        if len(self.estimator_groups) == 0:
            raise ValueError("No estimators defined")

        group_idx = [
            i for i, x in enumerate(self.estimator_groups) if x["name"] == group_name
        ]
        if len(group_idx) == 0:
            raise ValueError(f"No group named {group_name} found")
        for group in group_idx:
            for estimator in self.estimator_groups[group]["estimators"]:
                self.reporter.log_string(
                    f"calibration group: {group_name}, predicting {estimator.name}"
                )
                estimator.predict(df, inplace=True, *args, **kwargs)  # noqa: B026 Star-arg unpacking after a keyword argument is strongly discouraged

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

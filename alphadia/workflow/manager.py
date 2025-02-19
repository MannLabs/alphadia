# native imports
import logging
import os
import pickle
import traceback
import typing
from collections import defaultdict
from copy import deepcopy

import numpy as np

# alpha family imports
# third party imports
import pandas as pd
import torch
import xxhash

# alphadia imports
import alphadia
from alphadia import fdr
from alphadia.calibration.property import Calibration, calibration_model_provider
from alphadia.fdrx.models.two_step_classifier import TwoStepClassifier
from alphadia.workflow import reporting
from alphadia.workflow.config import Config

logger = logging.getLogger()

# TODO move all managers to dedicated modules

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


class BaseManager:
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
        figure_path: None | str = None,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """Base class for all managers which handle parts of the workflow.

        Parameters
        ----------

        path : str, optional
            Path to the manager pickle on disk.

        load_from_file : bool, optional
            If True, the manager will be loaded from file if it exists.
        """

        self._path = path
        self.is_loaded_from_file = False
        self.is_fitted = False
        self.figure_path = figure_path
        self._version = alphadia.__version__
        self.reporter = reporting.LogBackend() if reporter is None else reporter

        if load_from_file:
            # Note: be careful not to overwrite loaded values by initializing them in child classes after calling super().__init__()
            self.load()

    @property
    def path(self):
        """Path to the manager pickle on disk."""
        return self._path

    @property
    def is_loaded_from_file(self):
        """Check if the calibration manager was loaded from file."""
        return self._is_loaded_from_file

    @is_loaded_from_file.setter
    def is_loaded_from_file(self, value):
        self._is_loaded_from_file = value

    @property
    def is_fitted(self):
        """Check if all estimators in all calibration groups are fitted."""
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value):
        self._is_fitted = value

    def save(self):
        """Save the state to pickle file."""
        if self.path is None:
            return

        try:
            with open(self.path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            self.reporter.log_string(
                f"Failed to save {self.__class__.__name__} to {self.path}: {str(e)}",
                verbosity="error",
            )

            self.reporter.log_string(
                f"Traceback: {traceback.format_exc()}", verbosity="error"
            )

    def load(self):
        """Load the state from pickle file."""
        if self.path is None:
            self.reporter.log_string(
                f"{self.__class__.__name__} not loaded from disk.",
            )
            return
        elif not os.path.exists(self.path):
            self.reporter.log_string(
                f"{self.__class__.__name__} not found at: {self.path}",
                verbosity="warning",
            )
            return

        try:
            with open(self.path, "rb") as f:
                loaded_state = pickle.load(f)

                if loaded_state._version == self._version:
                    self.__dict__.update(loaded_state.__dict__)
                    self.is_loaded_from_file = True
                    self.reporter.log_string(
                        f"Loaded {self.__class__.__name__} from {self.path}"
                    )
                else:
                    self.reporter.log_string(
                        f"Version mismatch while loading {self.__class__}: {loaded_state._version} != {self._version}. Will not load.",
                        verbosity="warning",
                    )
        except Exception:
            self.reporter.log_string(
                f"Failed to load {self.__class__.__name__} from {self.path}",
                verbosity="error",
            )

    def fit(self):
        """Fit the workflow property of the manager."""
        raise NotImplementedError(
            f"fit() not implemented for {self.__class__.__name__}"
        )

    def predict(self):
        """Return the predictions of the workflow property of the manager."""
        raise NotImplementedError(
            f"predict() not implemented for {self.__class__.__name__}"
        )

    def fit_predict(self):
        """Fit and return the predictions of the workflow property of the manager."""
        raise NotImplementedError(
            f"fit_predict() not implemented for {self.__class__.__name__}"
        )


class CalibrationManager(BaseManager):
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
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

        """

        super().__init__(path=path, load_from_file=load_from_file, **kwargs)

        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            self.estimator_groups = []
            self.load_config(CALIBRATION_MANAGER_CONFIG)

    @property
    def estimator_groups(self):
        """List of calibration groups."""
        return self._estimator_groups

    @estimator_groups.setter
    def estimator_groups(self, value):
        self._estimator_groups = value

    def disable_mobility_calibration(self):
        """Iterate all estimators and remove the mobility estimator from each group."""
        for group in self.estimator_groups:
            for estimator in group["estimators"]:
                if estimator.name == "mobility":
                    group["estimators"].remove(estimator)
                    self.reporter.log_string(
                        f'removed mobility estimator from group {group["name"]}'
                    )

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
                estimator.fit(
                    df, *args, neptune_key=f"{group_name}_{estimator.name}", **kwargs
                )

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


class OptimizationManager(BaseManager):
    def __init__(
        self,
        config: None | Config = None,
        gradient_length: None | float = None,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            rt_error = (
                config["search_initial"]["initial_rt_tolerance"]
                if config["search_initial"]["initial_rt_tolerance"] > 1
                else config["search_initial"]["initial_rt_tolerance"] * gradient_length
            )
            initial_parameters = {
                "ms1_error": config["search_initial"]["initial_ms1_tolerance"],
                "ms2_error": config["search_initial"]["initial_ms2_tolerance"],
                "rt_error": rt_error,
                "mobility_error": config["search_initial"][
                    "initial_mobility_tolerance"
                ],
                "column_type": "library",
                "num_candidates": config["search_initial"]["initial_num_candidates"],
                "classifier_version": -1,
                "fwhm_rt": config["optimization_manager"]["fwhm_rt"],
                "fwhm_mobility": config["optimization_manager"]["fwhm_mobility"],
                "score_cutoff": config["optimization_manager"]["score_cutoff"],
            }
            self.__dict__.update(
                initial_parameters
            )  # TODO either store this as a dict or in individual instance variables

            for key, value in initial_parameters.items():
                self.reporter.log_string(f"initial parameter: {key} = {value}")

    def fit(
        self, update_dict
    ):  # TODO siblings' implementations have different signatures
        """Update the parameters dict with the values in update_dict."""
        self.__dict__.update(update_dict)
        self.is_fitted = True

    def predict(self):
        """Return the parameters dict."""
        return self.parameters

    def fit_predict(self, update_dict):
        """Update the parameters dict with the values in update_dict and return the parameters dict."""
        self.fit(update_dict)
        return self.predict()


def get_group_columns(competetive: bool, group_channels: bool) -> list[str]:
    """
    Determine the group columns based on competitiveness and channel grouping.

    competitive : bool
        If True, group candidates eluting at the same time by grouping them under the same 'elution_group_idx'.
    group_channels : bool
        If True and 'competitive' is also True, further groups candidates by 'channel'.

    Returns
    -------
    list
        A list of column names to be used for grouping in the analysis. If competitive, this could be either
        ['elution_group_idx', 'channel'] or ['elution_group_idx'] depending on the `group_channels` flag.
        If not competitive, the list will always be ['precursor_idx'].

    """
    if competetive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]
    return group_columns


class FDRManager(BaseManager):
    def __init__(
        self,
        feature_columns: list,
        classifier_base,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        """Contains, updates and applies classifiers for target-decoy competitio-based false discovery rate (FDR) estimation.

        Parameters
        ----------
        feature_columns: list
            List of feature columns to use for the classifier
        classifier_base: object
            Base classifier object to use for the FDR estimation

        """
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            self.feature_columns = feature_columns
            self.classifier_store = defaultdict(list)
            self.classifier_base = classifier_base
            self.is_two_step_classifier = isinstance(classifier_base, TwoStepClassifier)

        self._current_version = -1
        self.load_classifier_store()

    def fit_predict(
        self,
        features_df: pd.DataFrame,
        decoy_strategy: typing.Literal[
            "precursor", "precursor_channel_wise", "channel"
        ] = "precursor",
        competetive: bool = True,
        df_fragments: None | pd.DataFrame = None,
        dia_cycle: None | np.ndarray = None,
        decoy_channel: int = -1,
        version: int = -1,
    ):
        """Fit the classifier and perform FDR estimation.

        Notes
        -----
            The classifier_hash must be identical for every call of fit_predict for self._current_version to give the right index in self.classifier_store.
        """
        available_columns = list(
            set(features_df.columns).intersection(set(self.feature_columns))
        )

        # perform sanity checks
        if len(available_columns) == 0:
            raise ValueError("No feature columns found in features_df")

        strategy_requires_decoy_column = (
            decoy_strategy == "precursor" or decoy_strategy == "precursor_channel_wise"
        )
        if strategy_requires_decoy_column and "decoy" not in features_df.columns:
            raise ValueError("decoy column not found in features_df")

        strategy_requires_channel_column = (
            decoy_strategy == "precursor_channel_wise" or decoy_strategy == "channel"
        )
        if strategy_requires_channel_column and "channel" not in features_df.columns:
            raise ValueError("channel column not found in features_df")

        if decoy_strategy == "channel" and decoy_channel == -1:
            raise ValueError("decoy_channel must be set if decoy_type is channel")

        if (
            decoy_strategy == "precursor" or decoy_strategy == "precursor_channel_wise"
        ) and decoy_channel > -1:
            self.reporter.log_string(
                "decoy_channel is ignored if decoy_type is precursor",
                verbosity="warning",
            )
            decoy_channel = -1

        if (
            decoy_strategy == "channel"
            and decoy_channel > -1
            and decoy_channel not in features_df["channel"].unique()
        ):
            raise ValueError(f"decoy_channel {decoy_channel} not found in features_df")

        self.reporter.log_string(
            f"performing {decoy_strategy} FDR with {len(available_columns)} features"
        )
        self.reporter.log_string(f"Decoy channel: {decoy_channel}")
        self.reporter.log_string(f"Competetive: {competetive}")

        classifier = self.get_classifier(available_columns, version)
        if decoy_strategy == "precursor":
            if not self.is_two_step_classifier:
                psm_df = fdr.perform_fdr(
                    classifier,
                    available_columns,
                    features_df[features_df["decoy"] == 0].copy(),
                    features_df[features_df["decoy"] == 1].copy(),
                    competetive=competetive,
                    group_channels=True,
                    df_fragments=df_fragments,
                    dia_cycle=dia_cycle,
                    figure_path=self.figure_path,
                )
            else:
                group_columns = get_group_columns(competetive, group_channels=True)

                psm_df = classifier.fit_predict(
                    features_df,
                    available_columns + ["score"],
                    group_columns=group_columns,
                )

        elif decoy_strategy == "precursor_channel_wise":
            channels = features_df["channel"].unique()
            psm_df_list = []
            for channel in channels:
                channel_df = features_df[
                    features_df["channel"].isin([channel, decoy_channel])
                ].copy()
                psm_df_list.append(
                    fdr.perform_fdr(
                        classifier,
                        available_columns,
                        channel_df[channel_df["decoy"] == 0].copy(),
                        channel_df[channel_df["decoy"] == 1].copy(),
                        competetive=competetive,
                        group_channels=True,
                        df_fragments=df_fragments,
                        dia_cycle=dia_cycle,
                        figure_path=self.figure_path,
                    )
                )
            psm_df = pd.concat(psm_df_list)
        elif decoy_strategy == "channel":
            channels = list(set(features_df["channel"].unique()) - set([decoy_channel]))
            psm_df_list = []
            for channel in channels:
                channel_df = features_df[
                    features_df["channel"].isin([channel, decoy_channel])
                ].copy()
                psm_df_list.append(
                    fdr.perform_fdr(
                        classifier,
                        available_columns,
                        channel_df[channel_df["channel"] != decoy_channel].copy(),
                        channel_df[channel_df["channel"] == decoy_channel].copy(),
                        competetive=competetive,
                        group_channels=False,
                        figure_path=self.figure_path,
                    )
                )

            psm_df = pd.concat(psm_df_list)
            psm_df.loc[psm_df["channel"] == decoy_channel, "decoy"] = 1
        else:
            raise ValueError(f"Invalid decoy_strategy: {decoy_strategy}")

        self.is_fitted = True

        self._current_version += 1
        self.classifier_store[column_hash(available_columns)].append(classifier)

        self.save()

        return psm_df

    def save_classifier_store(self, path: None | str = None, version: int = -1):
        """Saves the classifier store to disk.

        Parameters
        ----------
        path: None | str
            Where to save the classifier. Saves to alphadia/constants/classifier if None.
        version: int
            Version of the classifier to save. Takes the last classifier if -1 (default)

        """
        if path is None:
            path = os.path.join(
                os.path.dirname(alphadia.__file__), "constants", "classifier"
            )

        logger.info(f"Saving classifier store to {path}")

        for classifier_hash, classifier_list in self.classifier_store.items():
            torch.save(
                classifier_list[version].to_state_dict(),
                os.path.join(path, f"{classifier_hash}.pth"),
            )

    def load_classifier_store(self, path: None | str = None):
        """Loads the classifier store from disk.

        Parameters
        ----------
        path: None | str
            Location of the classifier to load. Loads from alphadia/constants/classifier if None.

        """
        if path is None:
            path = os.path.join(
                os.path.dirname(alphadia.__file__), "constants", "classifier"
            )

        logger.info(f"Loading classifier store from {path}")

        if (
            not self.is_two_step_classifier
        ):  # TODO add pretrained model for TwoStepClassifier
            for file in os.listdir(path):
                if file.endswith(".pth"):
                    classifier_hash = file.split(".")[0]

                    if classifier_hash not in self.classifier_store:
                        classifier = deepcopy(self.classifier_base)
                        classifier.from_state_dict(
                            torch.load(os.path.join(path, file), weights_only=False)
                        )
                        self.classifier_store[classifier_hash].append(classifier)

    def get_classifier(self, available_columns: list, version: int = -1):
        """Gets the classifier for a given set of feature columns and version. If the classifier is not found in the store, gets the base classifier instead.

        Parameters
        ----------
        available_columns: list
            List of feature columns
        version: int
            Version of the classifier to get

        Returns
        ----------
        object
            Classifier object

        """
        classifier_hash = column_hash(available_columns)
        if classifier_hash in self.classifier_store:
            classifier = self.classifier_store[classifier_hash][version]
        else:
            classifier = self.classifier_base
        return deepcopy(classifier)

    @property
    def current_version(self):
        return self._current_version

    def predict(self):
        """Return the parameters dict."""
        raise NotImplementedError(
            f"predict() not implemented for {self.__class__.__name__}"
        )

    def fit(self, update_dict):
        """Update the parameters dict with the values in update_dict and return the parameters dict."""
        raise NotImplementedError(
            f"fit() not implemented for {self.__class__.__name__}"
        )


def column_hash(columns):
    columns.sort()
    return xxhash.xxh64_hexdigest("".join(columns))


class TimingManager(BaseManager):
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        """Contains and updates timing information for the portions of the workflow."""
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})
        if not self.is_loaded_from_file:
            self.timings = {}

    def set_start_time(self, workflow_stage: str):
        """Stores the start time of the given stage of the workflow in the timings attribute. Also saves the timing manager to disk.

        Parameters
        ----------
        workflow_stage : str
            The name under which the timing will be stored in the timings dict
        """
        self.timings.update({workflow_stage: {"start": pd.Timestamp.now()}})

    def set_end_time(self, workflow_stage: str):
        """Stores the end time of the given stage of the workflow in the timings attribute and calculates the duration. Also saves the timing manager to disk.
        Parameters
        ----------
        workflow_stage : str
            The name under which the timing will be stored in the timings dict

        """
        self.timings[workflow_stage]["end"] = pd.Timestamp.now()
        self.timings[workflow_stage]["duration"] = (
            self.timings[workflow_stage]["end"] - self.timings[workflow_stage]["start"]
        ).total_seconds() / 60

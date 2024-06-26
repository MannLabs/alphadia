# native imports
import logging
import os
import pickle
import typing
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
from alphadia.workflow import reporting

logger = logging.getLogger()


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
        if self.path is not None:
            try:
                with open(self.path, "wb") as f:
                    pickle.dump(self, f)
            except Exception:
                self.reporter.log_string(
                    f"Failed to save {self.__class__.__name__} to {self.path}",
                    verbosity="error",
                )

    def load(self):
        """Load the state from pickle file."""
        if self.path is not None:
            if os.path.exists(self.path):
                try:
                    with open(self.path, "rb") as f:
                        loaded_state = pickle.load(f)

                        if loaded_state._version == self._version:
                            self.__dict__.update(loaded_state.__dict__)
                            self.is_loaded_from_file = True
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
                else:
                    self.reporter.log_string(
                        f"Loaded {self.__class__.__name__} from {self.path}"
                    )
            else:
                self.reporter.log_string(
                    f"{self.__class__.__name__} not found at: {self.path}",
                    verbosity="warning",
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
        config: None | dict = None,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        """Contains, updates and applies all calibrations for a single run.

        Calibrations are grouped into calibration groups. Each calibration group is applied to a single data structure (precursor dataframe, fragment fataframe, etc.). Each calibration group contains multiple estimators which each calibrate a single property (mz, rt, etc.). Each estimator is a `Calibration` object which contains the estimator function.

        Parameters
        ----------

        config : typing.Union[None, dict], default=None
            Calibration config dict. If None, the default config is used.

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
            self.load_config(config)

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
        initial_parameters: dict,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            self.__dict__.update(initial_parameters)

            for key, value in initial_parameters.items():
                self.reporter.log_string(f"initial parameter: {key} = {value}")

    def fit(self, update_dict):
        """Update the parameters dict with the values in update_dict."""
        self.__dict__.update(update_dict)
        self.is_fitted = True
        self.save()

    def predict(self):
        """Return the parameters dict."""
        return self.parameters

    def fit_predict(self, update_dict):
        """Update the parameters dict with the values in update_dict and return the parameters dict."""
        self.fit(update_dict)
        return self.predict()


class FDRManager(BaseManager):
    def __init__(
        self,
        feature_columns: list,
        classifier_base,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            self.feature_columns = feature_columns
            self.classifier_store = {}
            self.classifier_base = classifier_base

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
    ):
        """Update the parameters dict with the values in update_dict."""
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

        classifier = self.get_classifier(available_columns)
        if decoy_strategy == "precursor":
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
                        reuse_fragments=True,
                        figure_path=self.figure_path,
                    )
                )

            psm_df = pd.concat(psm_df_list)
            psm_df.loc[psm_df["channel"] == decoy_channel, "decoy"] = 1
        else:
            raise ValueError(f"Invalid decoy_strategy: {decoy_strategy}")

        self.is_fitted = True
        self.classifier_store[column_hash(available_columns)] = classifier
        self.save()

        return psm_df

    def save_classifier_store(self, path=None):
        if path is None:
            path = os.path.join(
                os.path.dirname(alphadia.__file__), "constants", "classifier"
            )

        logger.info(f"Saving classifier store to {path}")

        for classifier_hash, classifier in self.classifier_store.items():
            torch.save(
                classifier.to_state_dict(), os.path.join(path, f"{classifier_hash}.pth")
            )

    def load_classifier_store(self, path=None):
        if path is None:
            path = os.path.join(
                os.path.dirname(alphadia.__file__), "constants", "classifier"
            )

        logger.info(f"Loading classifier store from {path}")

        for file in os.listdir(path):
            if file.endswith(".pth"):
                classifier_hash = file.split(".")[0]

                if classifier_hash not in self.classifier_store:
                    self.classifier_store[classifier_hash] = deepcopy(
                        self.classifier_base
                    )
                    self.classifier_store[classifier_hash].from_state_dict(
                        torch.load(os.path.join(path, file))
                    )

    def get_classifier(self, available_columns):
        classifier_hash = column_hash(available_columns)
        if classifier_hash in self.classifier_store:
            classifier = self.classifier_store[classifier_hash]
        else:
            classifier = deepcopy(self.classifier_base)
        return classifier

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

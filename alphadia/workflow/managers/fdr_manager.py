import logging
import os
from collections import defaultdict
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xxhash

import alphadia
from alphadia.fdr import fdr
from alphadia.fdr._fdrx.models.two_step_classifier import TwoStepClassifier
from alphadia.fdr.classifiers import Classifier
from alphadia.workflow.config import Config
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


def get_group_columns(competetive: bool, group_channels: bool) -> list[str]:
    """
    Determine the group columns based on competetiveness and channel grouping.

    competetive : bool
        If True, group candidates eluting at the same time by grouping them under the same 'elution_group_idx'.
    group_channels : bool
        If True and 'competetive' is also True, further groups candidates by 'channel'.

    Returns
    -------
    list
        A list of column names to be used for grouping in the analysis. If competetive, this could be either
        ['elution_group_idx', 'channel'] or ['elution_group_idx'] depending on the `group_channels` flag.
        If not competetive, the list will always be ['precursor_idx'].

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


def column_hash(columns):
    columns.sort()
    return xxhash.xxh64_hexdigest("".join(columns))


class FDRManager(BaseManager):
    def __init__(
        self,
        feature_columns: list,
        classifier_base: Classifier | TwoStepClassifier,
        config: Config,
        dia_cycle: None | np.ndarray = None,
        path: None | str = None,
        load_from_file: bool = True,
        random_state: int | None = None,
        **kwargs,
    ):
        """Contains, updates and applies classifiers for target-decoy competition-based false discovery rate (FDR) estimation.

        Parameters
        ----------
        feature_columns: list
            List of feature columns to use for the classifier
        classifier_base: object
            Base classifier object to use for the FDR estimation
        config: Config
            The workflow configuration object
        dia_cycle: None | np.ndarray
            DIA cycle information, if applicable. If None, no DIA cycle information is used.
        path : str, optional
            Path to the manager pickle on disk.
        load_from_file : bool, optional
            If True, the manager will be loaded from file if it exists.
        random_state: int, optional
            Random state for reproducibility.
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

        self._compete_for_fragments = config["search"]["compete_for_fragments"]

        self._dia_cycle = dia_cycle

        self._np_rng = (
            None if random_state is None else np.random.default_rng(random_state)
        )

    def fit_predict(
        self,
        features_df: pd.DataFrame,
        decoy_strategy: Literal["precursor", "precursor_channel_wise", "channel"],
        competetive: bool,
        df_fragments: pd.DataFrame | None = None,
        decoy_channel: int = -1,
        version: int = -1,
    ):
        """Fit the classifier and perform FDR estimation.

        Parameters
        ----------
        features_df: pd.DataFrame
            DataFrame containing the features to use for the classifier. Must contain the columns specified in self.feature_columns.
        decoy_strategy: Literal["precursor", "precursor_channel_wise", "channel"]
            The decoy strategy.
        competetive: bool
            Whether competetive scoring should be used.
        df_fragments: None | pd.DataFrame
            Dataframe containing the fragments to use for the classifier. If None, no fragments are used.
        decoy_channel: int
            Channel to use for decoy competition if decoy_strategy is "channel". Defaults to -1, which means no decoy channel is used.
        version: int
            Version of the classifier to use. If -1, uses the latest version. Defaults to -1.

        Notes
        -----
            The classifier_hash must be identical for every call of fit_predict for self._current_version to give the right index in self.classifier_store.
        """
        available_columns = list(
            set(features_df.columns).intersection(set(self.feature_columns))
        )

        self._check_valid_input(
            available_columns, decoy_channel, decoy_strategy, features_df
        )

        if (
            decoy_strategy == "precursor" or decoy_strategy == "precursor_channel_wise"
        ) and decoy_channel > -1:
            self.reporter.log_string(
                "decoy_channel is ignored if decoy_strategy is 'precursor' or 'precursor_channel_wise'.",
                verbosity="warning",
            )
            decoy_channel = -1

        self.reporter.log_string(
            f"performing {decoy_strategy} FDR with {len(available_columns)} features"
        )
        self.reporter.log_string(f"Decoy channel: {decoy_channel}")
        self.reporter.log_string(f"Competetive: {competetive}")

        classifier = self.get_classifier(available_columns, version)
        random_state = (
            None if self._np_rng is None else self._np_rng.integers(0, 1_000_000)
        )

        if decoy_strategy == "precursor":
            if not self.is_two_step_classifier:
                psm_df = fdr.perform_fdr(
                    classifier,
                    available_columns,
                    features_df[features_df["decoy"] == 0].copy(),
                    features_df[features_df["decoy"] == 1].copy(),
                    competetive=competetive,
                    group_channels=True,
                    # TODO move this logic to perform_fdr():
                    df_fragments=df_fragments if self._compete_for_fragments else None,
                    dia_cycle=self._dia_cycle,
                    figure_path=self.figure_path,
                    random_state=random_state,
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
                        df_fragments=df_fragments
                        if self._compete_for_fragments
                        else None,
                        dia_cycle=self._dia_cycle,
                        figure_path=self.figure_path,
                        random_state=random_state,
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
                        random_state=random_state,
                    )
                )

            psm_df = pd.concat(psm_df_list)
            psm_df.loc[psm_df["channel"] == decoy_channel, "decoy"] = 1
        else:
            raise ValueError(f"Invalid decoy_strategy: {decoy_strategy}")

        self._current_version += 1
        self.classifier_store[column_hash(available_columns)].append(classifier)

        self.save()

        return psm_df

    def _check_valid_input(
        self,
        available_columns: list[str],
        decoy_channel: int,
        decoy_strategy: str,
        features_df: pd.DataFrame,
    ):
        """Checks if the input features_df is valid for the FDR estimation.

        Raises ValueError if the input is not valid.
        """
        if len(available_columns) == 0:
            raise ValueError("No feature columns found in features_df")

        strategy_requires_decoy_column = (
            decoy_strategy == "precursor" or decoy_strategy == "precursor_channel_wise"
        )
        if strategy_requires_decoy_column and "decoy" not in features_df.columns:
            raise ValueError("Column 'decoy' not found in features_df")

        strategy_requires_channel_column = (
            decoy_strategy == "precursor_channel_wise" or decoy_strategy == "channel"
        )
        if strategy_requires_channel_column and "channel" not in features_df.columns:
            raise ValueError("Column 'channel' not found in features_df")

        if decoy_strategy == "channel" and decoy_channel == -1:
            raise ValueError("decoy_channel must be set if decoy_type is channel")

        if (
            decoy_strategy == "channel"
            and decoy_channel > -1
            and decoy_channel not in features_df["channel"].unique()
        ):
            raise ValueError(f"decoy_channel {decoy_channel} not found in features_df")

    def save_classifier_store(
        self, path: None | str = None, version: int = -1
    ):  # TODO: unused?
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

    def get_classifier(
        self, available_columns: list, version: int = -1
    ) -> Classifier | TwoStepClassifier:
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

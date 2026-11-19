import logging
from collections.abc import Callable, Collection
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import xxhash

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import Classifier
from alphadia.workflow.config import Config
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


def get_group_columns(competitive: bool, group_channels: bool) -> list[str]:
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
    if competitive:
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
    return xxhash.xxh64_hexdigest("".join(columns).encode())


class FDRManager(BaseManager):
    def __init__(
        self,
        feature_columns: list,
        classifier_base: Classifier,
        config: Config,
        dia_cycle: None | np.ndarray = None,
        dia_mobility_values: None | np.ndarray = None,
        path: None | str = None,
        load_from_file: bool = True,
        random_state: int | None = None,
        **kwargs,
    ):
        """Contains, updates and applies classifiers for target-decoy competition-based false discovery rate (FDR) estimation.

        The manager keeps one classifier per set of feature columns. Each fit on the same feature columns continues
        training that classifier, so the calibration rounds of a raw file train one classifier further. A fit on other
        feature columns starts from a copy of `classifier_base`.

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
        dia_mobility_values: None | np.ndarray
            1/K0 of every scan of `dia_cycle` for fragment competition on ion mobility data; None skips fragment
            competition there.
        path : str, optional
            Path to the manager pickle on disk.
        load_from_file : bool, optional
            If True, the manager will be loaded from file if it exists.
        random_state: int, optional
            Random state for reproducibility.
        """
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)

        if not self.is_loaded_from_file:
            self.feature_columns = feature_columns
            self.classifier_base = classifier_base

        # Classifiers are not reused across runs, so a manager loaded from file also starts without one.
        self._classifiers: dict[str, Classifier] = {}

        self._compete_for_fragments = config["search"]["compete_for_fragments"]

        self._dia_cycle = dia_cycle
        self._dia_mobility_values = dia_mobility_values

        self._np_rng = (
            None if random_state is None else np.random.default_rng(random_state)
        )

    def fit_predict(
        self,
        features_df: pd.DataFrame,
        decoy_strategy: Literal["precursor", "precursor_channel_wise", "channel"],
        competitive: bool,
        df_fragments: pd.DataFrame | None = None,
        decoy_channel: int = -1,
        fragment_provider: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        excluded_columns: Collection[str] = (),
        train_on_best_candidate: bool = False,
    ):
        """Fit the classifier and perform FDR estimation.

        Parameters
        ----------
        features_df: pd.DataFrame
            DataFrame containing the features to use for the classifier. Must contain the columns specified in self.feature_columns.
        decoy_strategy: Literal["precursor", "precursor_channel_wise", "channel"]
            The decoy strategy.
        competitive: bool
            Whether competitive scoring should be used.
        df_fragments: None | pd.DataFrame
            Dataframe containing the fragments to use for the classifier. If None, no fragments are used.
        decoy_channel: int
            Channel to use for decoy competition if decoy_strategy is "channel". Defaults to -1, which means no decoy channel is used.
        fragment_provider: None | Callable[[pd.DataFrame], pd.DataFrame]
            Returns the fragments of the PSMs it is given, for fragment competition when `df_fragments` is not
            available up front. Only used with decoy_strategy "precursor".
        excluded_columns: Collection[str]
            Feature columns the classifier does not use in this call, although `features_df` carries them.
        train_on_best_candidate: bool
            Train the classifier only on the most target-like candidate of each precursor. Only used with
            decoy_strategy "precursor".
        """
        available_columns = list(
            set(features_df.columns)
            .intersection(set(self.feature_columns))
            .difference(excluded_columns)
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
        self.reporter.log_string(f"competitive: {competitive}")

        classifier = self._classifier(available_columns)
        random_state = (
            None if self._np_rng is None else self._np_rng.integers(0, 1_000_000)
        )

        if decoy_strategy == "precursor":
            decoy = features_df["decoy"].to_numpy()
            is_target_or_decoy = (decoy == 0) | (decoy == 1)
            # Filtering copies the PSMs, so only do it when there is something to drop.
            if not is_target_or_decoy.all():
                features_df = features_df[is_target_or_decoy]
                decoy = decoy[is_target_or_decoy]
            psm_df = fdr.perform_fdr(
                classifier,
                available_columns,
                features_df,
                decoy == 1,
                competitive=competitive,
                group_channels=True,
                # TODO move this logic to perform_fdr():
                df_fragments=df_fragments if self._compete_for_fragments else None,
                fragment_provider=fragment_provider
                if self._compete_for_fragments
                else None,
                dia_cycle=self._dia_cycle,
                dia_mobility_values=self._dia_mobility_values,
                train_on_best_candidate=train_on_best_candidate,
                figure_path=self.figure_path,
                random_state=random_state,
            )

        elif decoy_strategy == "precursor_channel_wise":
            channels = features_df["channel"].unique()
            psm_df_list = []
            for channel in channels:
                channel_df = features_df[
                    features_df["channel"].isin([channel, decoy_channel])
                    & features_df["decoy"].isin([0, 1])
                ]
                psm_df_list.append(
                    fdr.perform_fdr(
                        classifier,
                        available_columns,
                        channel_df,
                        channel_df["decoy"].to_numpy() == 1,
                        competitive=competitive,
                        group_channels=True,
                        df_fragments=df_fragments
                        if self._compete_for_fragments
                        else None,
                        dia_cycle=self._dia_cycle,
                        dia_mobility_values=self._dia_mobility_values,
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
                ]
                psm_df_list.append(
                    fdr.perform_fdr(
                        classifier,
                        available_columns,
                        channel_df,
                        channel_df["channel"].to_numpy() == decoy_channel,
                        competitive=competitive,
                        group_channels=False,
                        figure_path=self.figure_path,
                        random_state=random_state,
                    )
                )

            psm_df = pd.concat(psm_df_list)
            psm_df.loc[psm_df["channel"] == decoy_channel, "decoy"] = 1
        else:
            raise ValueError(f"Invalid decoy_strategy: {decoy_strategy}")

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

    def _classifier(self, available_columns: list[str]) -> Classifier:
        """Return the classifier of these feature columns, which a fit trains further in place.

        The first fit on a set of feature columns starts from a copy of `classifier_base`.
        """
        classifier_hash = column_hash(available_columns)
        if classifier_hash not in self._classifiers:
            self._classifiers[classifier_hash] = deepcopy(self.classifier_base)
        return self._classifiers[classifier_hash]

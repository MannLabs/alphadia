"""
This module is responsible for a reporting feature to allow updating the default configuration
using a list of experiment configurations in an interactive way by visualizing to the user the modifications/updates
that will be applied to the default configuration by every experiment configuration in the list.

Experiment Configuration Update Process

In our analysis, we focus on discerning changes between the default configuration and all experiments, rather than within individual experiments. To achieve this, we adopt the following approach:
- Assuming that the default config will always have all possible config keys and no new config keys will be introduced by the experiments configs at any level of the config. i.e Experiment configs only update values.

1. Loop over default config:
   - We will recursively loop over all keys in the default config and look for updates in experiments configs, if found update the value and store in a new translated form.

2. Maintaining Source Information:
   - To retain the source of changes or modifications, we employ a special representation of the regular config.
   - Each leaf node in the config is transformed into a tuple format: `((value, source experiment name))`.
   - This allows users to trace the origin of a particular value, indicating which experiment triggered the change.
   - See translate_config() and translate_config_back() for more details.

**Note:**
- The order of experiments holds significance, with configurations later in the sequence taking precedence in terms of their impact on changes.
- But we still define the source of the update to be the first experiment that triggered the change.
"""

import copy
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger()


def get_tree_structure(last_item_arr: list[bool], update=False):
    tree_structure = ""
    for i in last_item_arr[:-1]:
        if i:
            tree_structure += "    "
        else:
            tree_structure += "│   "
    if len(last_item_arr) > 0:
        if last_item_arr[-1]:
            tree_structure += "└──"
        else:
            tree_structure += "├──"
    return tree_structure


def print_w_style(
    string: str, style: str = "auto", last_item_arr: list[bool] | None = None
) -> None:
    """
    Print string with tree structure and uses ANSI color codes to color the string base on the style:
    - update: green color
    - new: green color but add +++ to the beginning
    - default: no color
    - old: red color

    Parameters
    ----------
    string : str
        String to be printed

    style : str
        Style of the string

    last_item_arr : list[bool], optional
        If the string is the last item in the list or dict, by default [False]

    """
    if last_item_arr is None:
        last_item_arr = [False]
    if style == "auto":
        # Check what the config name in string inside the brackets ( )
        # If the source is default, remove the brackets and set style to default
        # Else set style to new
        if string.find("(") != -1:
            config_name = string[string.find("(") + 1 : string.find(")")]
            if config_name == "default":
                string = string[: string.find("(")] + string[string.find(")") + 1 :]
                style = "default"
            else:
                style = "new"
        else:
            style = "default"

    if style == "update":
        # Green color
        style = "\x1b[32;20m"
        reset = "\x1b[0m"
    elif style == "new":
        # green color
        style = "\x1b[32;20m"
        reset = "\x1b[0m"
    elif style == "default":
        # no color
        style = ""
        reset = ""
    # Print with tree structure using level and color

    tree_structure = get_tree_structure(last_item_arr, update=style == "update")

    logger.info(f"{tree_structure}{style}{string}{reset}")


def print_recursively(
    config: dict[str, Any] | list[Any],
    level: int = 0,
    style: str = "auto",
    last_item: bool = False,
    last_item_arr: list | None = None,
) -> None:
    """
    Recursively print any config with tree structure and uses ANSI color codes to color the string based on the style.

    Parameters
    ----------
    config : dict or list
        Config data to be printed

    level : int
        Level of the config in the tree structure

    style : str
        Style of the config

    last_item : bool, optional
        If the config is the last item in the list or dict, by default False.

    last_item_arr : TODO
    """

    if last_item_arr is None:
        last_item_arr = []
    if isinstance(config, tuple):
        print_w_style(
            f"{config[0]} ({config[1]})", style=style, last_item_arr=last_item_arr
        )
        return

    if isinstance(config, list):
        for i, value in enumerate(config):
            is_last_item_list = i == len(config) - 1
            print_recursively(
                value, style=style, last_item_arr=last_item_arr + [is_last_item_list]
            )
        return

    if isinstance(config, dict):
        for key, value in config.items():
            is_last_item_dict = key == list(config.keys())[-1]

            if isinstance(value, tuple):
                print_w_style(
                    f"{key}: {value[0]} ({value[1]})",
                    style=style,
                    last_item_arr=last_item_arr + [is_last_item_dict],
                )
                continue

            elif isinstance(value, list | dict):
                print_w_style(
                    f"{key}",
                    style=style,
                    last_item_arr=last_item_arr + [is_last_item_dict],
                )
                print_recursively(
                    value,
                    style=style,
                    last_item_arr=last_item_arr + [is_last_item_dict],
                )
            else:
                print_w_style(
                    f"{key}: {value}",
                    style=style,
                    last_item_arr=last_item_arr + [is_last_item_dict],
                )
        return

    print_w_style(f"{config}", style=style, last_item_arr=last_item_arr)


def translate_config(
    default_config: dict[str, Any] | list[Any], name: str
) -> dict[str, Any] | list[Any]:
    """
    Takes as input a dictionary or list of dictianry that contains config values and a name of experiment
    and changes every leaf value to a tuple (value, name)

    Parameters
    ----------
    default_config : dict or list
        Config data to be translated

    name : str
        Name of the experiment

    Returns
    -------
    dict or list
        Translated config data with leaf values as tuple (value, experiment name)
    """
    if not isinstance(default_config, dict) and not isinstance(default_config, list):
        return (default_config, name)

    if isinstance(default_config, dict):
        for key, value in default_config.items():
            if not isinstance(value, dict) and not isinstance(value, list):
                default_config[key] = (value, name)
            else:
                default_config[key] = translate_config(value, name)
    elif isinstance(default_config, list):
        for i, value in enumerate(default_config):
            if not isinstance(value, dict) and not isinstance(value, list):
                default_config[i] = (value, name)
            else:
                default_config[i] = translate_config(value, name)
    return default_config


def translate_config_back(config: dict[str, Any] | list[Any]):
    """
    Takes as input a dictionary or list of dictionary that contains config values and changes every leaf value from a tuple (value, name) to value

    Parameters
    ----------
    config : dict or list
        Config data to be translated

    Returns
    -------
    dict or list
        Translated config data with leaf values as value only i.e original format
    """
    if (
        not isinstance(config, dict)
        and not isinstance(config, list)
        and isinstance(config, tuple)
    ):
        return config

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, tuple):
                config[key] = value[0]
            else:
                config[key] = translate_config_back(value)
    elif isinstance(config, list):
        for i, value in enumerate(config):
            if isinstance(value, tuple):
                config[i] = value[0]
            else:
                config[i] = translate_config_back(value)
    return config


def update_recursive(
    config: dict[str, Any],
    experiment_configs: list[dict[str, Any] | list[Any]],
    level: int = 0,
    print_output: bool = True,
    is_leaf_node: bool = False,
    last_item_arr: list | None = None,
) -> dict[str, Any] | list[Any]:
    """
    Recursively update the default config with the experiments config
    print the config in a tree structure using pipes and dashes and colors to indicate the changes

    Parameters
    ----------
    default_config : dict
        Default config data

    experiment_config : dict or list
        Experiment config data (updates)

    level : int, optional
        Level of the config in tree stucture, by default 0

    print_output : bool, optional
        Whether to print the modifications or not, by default True

    is_leaf_node : bool, optional
        Whether the config is a leaf node or not, by default False
        This is used to determine the style of the config only does not affect the update process

    last_item_arr: TODO
    """
    if last_item_arr is None:
        last_item_arr = []
    parent_key = config["key"]
    default_config = config["value"]
    # If the default config is a leaf node, then we can update it
    if isinstance(default_config, tuple):
        if print_output:
            if parent_key is None:
                print_w_style(
                    f"{default_config[0]} ({default_config[1]})",
                    style="auto",
                    last_item_arr=last_item_arr,
                )
            else:
                print_w_style(
                    f"{parent_key}: {default_config[0]} ({default_config[1]})",
                    style="auto",
                    last_item_arr=last_item_arr,
                )
        # Find the latest update
        new_value = default_config
        for experiment_config in experiment_configs:
            if default_config[0] != experiment_config[0]:
                # Only update if the new value is "New" this is to have the source of information as the first experiment update to that value
                new_value = (
                    experiment_config
                    if experiment_config[0] != new_value[0]
                    else new_value
                )  #
        # If we have a new value, print it
        if new_value != default_config and print_output:
            if parent_key is None:
                print_w_style(
                    f"{new_value[0]} ({new_value[1]})",
                    style="update",
                    last_item_arr=last_item_arr,
                )
            else:
                print_w_style(
                    f"{parent_key}: {new_value[0]} ({new_value[1]})",
                    style="update",
                    last_item_arr=last_item_arr,
                )
        return new_value

    # If the default config is a list, then we need to update each item in the list
    if isinstance(default_config, list):
        for i, default_value in enumerate(default_config):
            is_last_item = i == len(default_config) - 1

            if not isinstance(
                default_value, tuple
            ):  # If the default value is not a leaf node, print it's key on separate line
                print_w_style(
                    f"{i}", style="auto", last_item_arr=last_item_arr + [is_last_item]
                )

            # Collect potential updates for this item
            potential_config_updates = []
            for experiment_config in experiment_configs:
                if i < len(experiment_config):
                    potential_config_updates.append(experiment_config[i])

            default_config[i] = update_recursive(
                {"key": None, "value": default_value},
                potential_config_updates,
                level,
                print_output,
                is_last_item,
                last_item_arr=last_item_arr + [is_last_item],
            )
        return default_config

    for default_key, default_value in default_config.items():
        is_last_item = default_key == list(default_config.keys())[-1]

        if not isinstance(
            default_value, tuple
        ):  # If the default value is not a leaf node, print it's key on separate line
            print_w_style(
                f"{default_key}",
                style="auto",
                last_item_arr=last_item_arr + [is_last_item],
            )

        # Collect potential updates for this item
        potential_config_updates = []
        for experiment_config in experiment_configs:
            if default_key in experiment_config:
                potential_config_updates.append(experiment_config[default_key])

        default_config[default_key] = update_recursive(
            {"key": default_key, "value": default_value},
            potential_config_updates,
            level + 1,
            print_output,
            is_last_item,
            last_item_arr=last_item_arr + [is_last_item],
        )
    return default_config


def recursive_fill_table(
    df: pd.DataFrame, experiment_name: str, parent_key: str, value: Any
) -> None:
    """
    Recursively fill the table with the modifications happening to the config.
    if the value is a dict or list then it will recursively call itself with the value as the new value
    else it will check if the value is different from the last recorded value and if it is then it will add it to the table
    if not then it will do nothing, so that for each key the last value on the y axis is the last value that key was set to.

    Parameters
    ----------
    df : pandas.DataFrame
        Table of modifications

    experiment_name : str
        Name of the experiment

    parent_key : str
        Parent key of the value

    value : Any
        Value of the key
    """
    if isinstance(value, dict):
        for key, value_ in value.items():
            recursive_fill_table(df, experiment_name, parent_key + "." + key, value_)
    elif isinstance(value, list):
        for i, value_ in enumerate(value):
            recursive_fill_table(
                df, experiment_name, parent_key + "[" + str(i) + "]", value_
            )
    else:
        # Check if t he value is different from the last recorded value
        experiment_index = df.columns.get_loc(experiment_name)
        key_idx = np.where(df.index == parent_key)[0]
        if len(key_idx) == 0:
            # If the key doesn't exist then add it
            df.loc[parent_key, experiment_name] = value
            return
        else:
            key_idx = key_idx[0]

        while df.isnull().iloc[key_idx, experiment_index] and experiment_index >= 0:
            experiment_index -= 1
        last_recorded_value = df.iloc[key_idx, experiment_index]
        if last_recorded_value != value:
            df.loc[parent_key, experiment_name] = value


def get_update_table(
    default_config: "Config", configs: list["Config"]
) -> "pd.DataFrame":
    """
    Returns a table of the modifications happening to the config
    such that the rows are the keys and the columns are the experiments
    levels of the config are represeneted by . in the keys so for example
    the key
        a:
            b:
                c:
    would be represented as a.b.c

    Parameters
    ----------
    default_config : Config
        Default config

    configs : list of Config
        List of experiment configs

    Returns
    -------
    df : pandas.DataFrame
        Table of modifications
    """

    columns = [config.experiment_name for config in configs]

    columns.insert(0, default_config.experiment_name)

    df = pd.DataFrame(columns=columns)

    # Add the default config to the modifications
    recursive_fill_table(df, default_config.experiment_name, "", default_config.config)

    for c in configs:
        recursive_fill_table(df, c.experiment_name, "", c.config)

    # Fill Nan with '-' to make it look nicer
    df = df.fillna("-")
    # remove the first dot from the index
    df.index = df.index.str[1:]

    return df


class Config:
    """
    Config class that can read from and write to yaml and json files
    and can be used to update the config with experiment configs
    and print the config with tree structure and uses ANSI color codes to color the string.
    """

    def __init__(self, experiment_name: str = "default") -> None:
        self.experiment_name = experiment_name
        self.config = {}
        self.translated_config = {}

    def from_yaml(self, path: str) -> None:
        with open(path) as f:
            self.config = yaml.safe_load(f)

    def from_json(self, path: str) -> None:
        with open(path) as f:
            self.config = json.load(f)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.config, f, sort_keys=False)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.config, f)

    def from_dict(self, config: dict[str, Any]) -> None:
        self.config = config

    def to_dict(self) -> dict[str, Any]:
        return self.config

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.config

    def __repr__(self) -> str:
        if len(self.translated_config) > 0:
            print_recursively(self.translated_config, 0, "auto")
        else:
            print_recursively(self.config, 0, "auto")
        return ""

    def translate(self):
        """
        Translate the config dict so that every leaf node is a tuple (value, experiment_name), instead of just value

        and sets the translate_config attribute, uses the general translate_config function
        """
        temp = copy.deepcopy(self.config)
        self.translated_config = translate_config(temp, self.experiment_name)
        # Let's make sure that the main config was not translated
        return self.translated_config

    def align_config_w_translation(self) -> None:
        """
        Translate the config dict back so that every leaf node is a value, instead of a tuple (value, experiment_name)
        uses the general translate_config_back function
        """
        temp = copy.deepcopy(self.translated_config)
        self.config = translate_config_back(temp)

    def update(self, experiments: list["Config"], print_modifications: bool = True):
        """
        Updates the config with the experiment configs,
        and allow for multiple experiment configs to be added.

        Parameters
        ----------
        experiments : list of configs
            List of experiment configs

        print_modifications : bool, optional
            Whether to print the modifications or not, either way the updated config will be printed. When set to True,
            the modifications will be first printed to show old, updated, new values where as the updated config
            contains updated and unmodifed values.
        """

        # Translate the config dict first.
        self.translate()

        if len(experiments) > 0:
            translated_experiments = []
            for config in experiments:
                translated_experiments.append(config.translate())
            self.translated_config = update_recursive(
                {"key": "", "value": self.translated_config},
                translated_experiments,
                print_output=print_modifications,
            )
        # Translate the config dict back
        self.align_config_w_translation()

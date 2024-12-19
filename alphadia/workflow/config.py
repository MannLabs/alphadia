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

import json
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger()

USER_DEFINED = "user defined"
MULTISTEP_SEARCH = "multistep search"


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
        style = (
            "new"
            if any([s in string for s in [USER_DEFINED, MULTISTEP_SEARCH]])
            else "default"
        )

    if style in ["update", "new"]:
        # Green color
        style = "\x1b[32;20m"
        reset = "\x1b[0m"
    else:
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

    all_keys = list(default_config.keys())
    for experiment_config in experiment_configs:
        all_keys += [key for key in experiment_config if key not in all_keys]

    for key in all_keys:
        style = "auto"
        if (
            key not in default_config
        ):  # TODO either this is obsolete or the module docstring needs an update
            style = "new"
            for experiment_config in experiment_configs:
                if key in experiment_config:
                    default_config[key] = experiment_config[key]
                    break

        default_value = default_config[key]

        is_last_item = key == all_keys[-1]

        if not isinstance(
            default_value, tuple
        ):  # If the default value is not a leaf node, print it's key on separate line
            print_w_style(
                f"{key}",
                style=style,
                last_item_arr=last_item_arr + [is_last_item],
            )

        # Collect potential updates for this item
        potential_config_updates = []
        for experiment_config in experiment_configs:
            if key in experiment_config:
                potential_config_updates.append(experiment_config[key])

        default_config[key] = update_recursive(
            {"key": key, "value": default_value},
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

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

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

    def update(self, experiments: list["Config"], print_modifications: bool = True):
        """
        Updates the config with the experiment configs, and allow for multiple experiment configs to be added.

        The order of experiments holds significance, with configurations later in the sequence taking precedence in terms of their impact on changes.

        Parameters
        ----------
        experiments : list of configs
            List of experiment configs

        print_modifications : bool, optional
            Whether to print the modifications or not, either way the updated config will be printed.
        """

        default_config = deepcopy(self.config)
        tracking_config = deepcopy(self.config)
        # this is a bit of a hack to initialize the tracking config
        update(deepcopy(self.config), default_config, tracking_config, "default")

        initial_config = deepcopy(self.config)
        for config in experiments:
            update(
                initial_config,
                config.to_dict(),
                tracking_config,
                config.experiment_name,
            )

        self.config = initial_config

        pretty_print_config(self.config, default_config, tracking_config)


def update(
    target_dict: dict, update_dict: dict, tracking_dict: dict, experiment_name: str
) -> None:
    """
    Recursively update target_dict with values from update_dict, following specific rules for different types.

    Args:
        target_dict: The dictionary to be modified
        update_dict: The dictionary containing update values
        tracking_dict: The dictionary containing the source of a value
        experiment_name: The name of the current experiment

    Notes:
        - Nested dictionaries are recursively updated
        - Only updates existing keys (no new keys added)
        - Complex type lists (lists of dicts) are updated by matching 'name' field
        - Simple type lists are overwritten

    Raises:
        ValueError in these cases:
        - a key is not found in the target_dict
        - the type of the update value does not match the type of the target value
        - a complex type list does not contain a 'name' field for each item
        - an item is not found in the target_dict
        - the type of an item in a complex type list does not match the type of the corresponding item in the update list
    """
    for key, update_value in update_dict.items():
        if key not in target_dict:
            raise ValueError(f"Key not found in target_dict: '{key}'")

        target_value = target_dict[key]
        tracking_value = tracking_dict[key]

        if not type(update_value) == type(target_value):
            raise ValueError(
                f"Type mismatch for key '{key}': {type(update_value)} != {type(target_value)}"
            )

        if isinstance(target_value, dict):
            update(target_value, update_value, tracking_value, experiment_name)

        elif isinstance(target_value, list):
            if target_value and isinstance(target_value[0], dict):
                _update_nested_list(
                    target_value, update_value, tracking_value, experiment_name
                )
            else:
                # For simple type lists, overwrite completely
                target_dict[key] = update_value
                tracking_dict[key] = experiment_name

        # Handle simple values
        else:
            target_dict[key] = update_value
            if key != "name":  # TODO remove with nested lists
                tracking_dict[key] = experiment_name


def _update_nested_list(
    target_value: list, update_value: list, tracking_value: list, experiment_name: str
) -> None:
    """Update a list of dictionaries (complex type list)."""
    _check_all_have_name_attributes(target_value)
    _check_all_have_name_attributes(update_value)
    # Create a map of name to item for quick lookup
    target_map = {item["name"]: item for item in target_value}
    tracking_map = {item["name"]: item for item in tracking_value}
    for update_item in update_value:
        if not isinstance(update_item, dict):
            raise ValueError(
                f"Complex type list items must be dictionaries, found {type(update_item)} '{update_item}' (update_item)"
            )

        # add new item to list
        if update_item["name"] not in target_map:
            target_item = update_item.copy()
            target_value.append(target_item)

            tracking_item = update_item.copy()
            tracking_value.append(tracking_item)
        else:
            target_item = target_map[update_item["name"]]
            tracking_item = tracking_map[update_item["name"]]
            # remove if "name" is the only tag
            if len(update_item) == 1:
                target_value.remove(target_item)
                tracking_value.remove(tracking_item)
            # update
            else:
                if not isinstance(target_item, dict):
                    raise ValueError(
                        f"Complex type list items must be dictionaries, found {type(target_item)} '{target_item}' (target_item)"
                    )

        update(target_item, update_item, tracking_item, experiment_name)


def _check_all_have_name_attributes(values):
    if any(["name" not in v for v in values]):
        raise ValueError(
            f"Complex type lists must contain a 'name' field for each item: {values}"
        )


def pretty_print_config(
    config: dict, default_config: dict, tracking_config: dict
) -> None:
    """
    Pretty print a configuration dictionary in a tree-like structure.

    Args:
        config: The configuration dictionary to print
        prefix: Current line prefix for proper indentation
    """

    _pretty_print(
        config, default_config=default_config, tracking_config=tracking_config
    )


def _pretty_print(
    config: dict,
    *,
    default_config: dict,
    tracking_config: dict,
    prefix: str = "",
):
    for i, (key, value) in enumerate(config.items()):
        is_last_item = i == len(config.items()) - 1

        # Determine the current line's prefix
        current_prefix = "└──" if is_last_item else "├──"

        # Determine the next level's prefix
        next_prefix = prefix + ("    " if is_last_item else "│   ")

        try:
            default_config_value = default_config[key]
        except TypeError:
            default_config_value = default_config[i]
        try:
            tracking_config_value = tracking_config[key]
        except TypeError:
            tracking_config_value = tracking_config[i]

        if isinstance(value, dict):
            # Print dictionary key and continue with nested values
            print(f"{prefix}{current_prefix}{key}")
            _pretty_print(
                value,
                default_config=default_config_value,
                tracking_config=tracking_config_value,
                prefix=next_prefix,
            )
        elif isinstance(value, list):
            # Handle lists
            print(f"{prefix}{current_prefix}{key}:")
            for j, item in enumerate(value):
                if isinstance(item, dict):
                    next_prefix = prefix + ("    " if is_last_item else "│   ")
                    try:
                        _pretty_print(
                            item,
                            default_config=default_config_value[j],
                            tracking_config=tracking_config_value[j],
                            prefix=next_prefix,
                        )
                    except Exception:
                        # in case something was added
                        _pretty_print(
                            item,
                            default_config=defaultdict(dict),
                            tracking_config=tracking_config_value[j],
                            prefix=next_prefix,
                        )
                else:
                    # For simple lists
                    try:
                        print(
                            f"{next_prefix}- {_pp(item, default_config_value[j], tracking_config_value)}"
                        )
                    except Exception:
                        # in case something was added
                        print(
                            f"{next_prefix}- {_pp(item, None, tracking_config_value)}"
                        )
        else:
            # Print leaf node (key-value pair)
            print(
                f"{prefix}{current_prefix}{key}: {_pp(value, default_config_value, tracking_config_value)}"
            )


def _pp(actual_value, default_value, tracking_value):
    if default_value == actual_value:
        msg = f"{actual_value}"
    else:
        msg = f"{actual_value} [{tracking_value}, default: {default_value}]"
    return msg

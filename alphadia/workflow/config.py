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
from copy import deepcopy
from typing import Any

import yaml

logger = logging.getLogger()

USER_DEFINED = "user defined"
MULTISTEP_SEARCH = "multistep search"


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
        return str(self.config)

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
        # we assume that the config that is updated is the default config
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
        tracking_dict: A dictionary with the same structure as target_dict, whose leaf values will be overwritten with experiment_name
        experiment_name: The name of the current experiment

    Notes:
        - Nested dictionaries are recursively updated
        - Only updates existing keys (no new keys added)
        - All lists are overwritten

    Raises:
        ValueError in these cases:
        - a key is not found in the target_dict
        - the type of the update value does not match the type of the target value
        - an item is not found in the target_dict
    """
    for key, update_value in update_dict.items():
        if key not in target_dict:
            raise ValueError(f"Key not found in target_dict: '{key}'")

        target_value = target_dict[key]
        tracking_value = tracking_dict[key]

        if (
            target_value is not None
            and type(update_value) != type(target_value)
            and not (
                isinstance(update_value, int | float)
                and isinstance(target_value, int | float)
            )
        ):
            raise ValueError(
                f"Type mismatch for key '{key}': {type(update_value)} != {type(target_value)}"
            )

        if isinstance(target_value, dict):
            update(target_value, update_value, tracking_value, experiment_name)

        elif isinstance(target_value, list):
            # overwrite lists completely
            target_dict[key] = update_value
            tracking_dict[key] = experiment_name

        # Handle simple values
        else:
            target_dict[key] = update_value
            tracking_dict[key] = experiment_name


def pretty_print_config(
    config: dict, default_config: dict, tracking_config: dict
) -> None:
    """
    Pretty print a configuration dictionary in a tree-like structure.

    Args:
        config: The configuration dictionary to print
        default_config: The default configuration dictionary to print
        tracking_config: A dictionary with the same structure as config, whose leaf values contain the experiment name that last updated the value
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
    """Recursively pretty print a configuration dictionary in a tree-like structure."""
    for i, (key, value) in enumerate(config.items()):
        is_last_item = i == len(config.items()) - 1

        # determine the current line's prefix
        current_prefix = "└──" if is_last_item else "├──"

        # determine the next level's prefix
        next_prefix = prefix + ("    " if is_last_item else "│   ")

        try:
            default_config_value = default_config[key]
        except (KeyError, TypeError):
            try:
                default_config_value = default_config[i]
            except KeyError:
                # nested list case
                default_config_value = "(added)"

        try:
            tracking_config_value = tracking_config[key]
        except (KeyError, TypeError):
            tracking_config_value = tracking_config

        if isinstance(value, dict):
            # Print dictionary key and continue with nested values
            logger.info(f"{prefix}{current_prefix}{key}")
            _pretty_print(
                value,
                default_config=default_config_value,
                tracking_config=tracking_config_value,
                prefix=next_prefix,
            )
        elif isinstance(value, list):
            logger.info(f"{prefix}{current_prefix}{key}:")
            for j, item in enumerate(value):
                # Handle nested lists
                if isinstance(item, dict):
                    next_prefix = prefix + ("    " if is_last_item else "│   ")

                    _pretty_print(
                        item,
                        default_config=default_config_value[j],
                        tracking_config=tracking_config_value,
                        prefix=next_prefix,
                    )

                else:
                    # For simple lists
                    default_value = (
                        default_config_value[j]
                        if j < len(default_config_value)
                        else "(added)"
                    )

                    logger.info(
                        f"{next_prefix}- {_make_pretty(item, default_value, tracking_config_value)}"
                    )

        else:
            # Print leaf node (key-value pair)
            logger.info(
                f"{prefix}{current_prefix}{key}: {_make_pretty(value, default_config_value, tracking_config_value)}"
            )


def _make_pretty(
    actual_value: str | int | float,
    default_value: str | int | float,
    tracking_value: str,
) -> str:
    """Create a pretty string representation of a configuration value."""
    if default_value == actual_value:
        msg = f"{actual_value}"
    else:
        style = "\x1b[32;20m"
        reset = "\x1b[0m"
        msg = (
            f"{style}{actual_value} [{tracking_value}, default: {default_value}]{reset}"
        )
    return msg

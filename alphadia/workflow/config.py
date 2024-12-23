"""This module is responsible for creating and storing the configuration.

It allows updating the default configuration with one or more experiment configurations.
The order of experiments holds significance, with configurations later in the sequence overwriting previous values.
Lists are always overwritten.

On demand, the current config can be visualized in a tree-like structure.
"""

import json
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any

import yaml

from alphadia.exceptions import KeyAddedConfigError, TypeMismatchConfigError

logger = logging.getLogger()

DEFAULT = "default"
USER_DEFINED = "user defined"
USER_DEFINED_CLI_PARAM = "user defined (cli)"
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

    def update(self, experiments: list["Config"], do_print: bool = False):
        """
        Updates the config with the experiment configs, and allow for multiple experiment configs to be added.

        The order of experiments holds significance, with configurations later in the sequence taking precedence in terms of their impact on changes.

        Parameters
        ----------
        experiments : list of configs
            List of experiment configs

        do_print : bool, optional
            Whether to print the modified config. Default is False.
        """

        # we assume that self.config holds the default config
        default_config = deepcopy(self.config)

        # initialize the tracking config as infinitely nested dictionary to be able to map all changes
        def _recursive_defaultdict():
            return defaultdict(_recursive_defaultdict)

        tracking_dict = defaultdict(_recursive_defaultdict)

        current_config = deepcopy(self.config)
        for experiment_config in experiments:
            logger.info(f"Updating config with '{experiment_config.experiment_name}'")
            _update(
                current_config,
                experiment_config.to_dict(),
                tracking_dict,
                experiment_config.experiment_name,
            )

        self.config = current_config

        if do_print:
            try:
                self._pretty_print(current_config, default_config, tracking_dict)
            except Exception as e:
                logger.warning(f"Could not print config: {e}")
                logger.info(f"{(yaml.dump(current_config))}")

    @staticmethod
    def _pretty_print(config: dict, default_config: dict, tracking_dict: dict) -> None:
        """
        Pretty print a configuration dictionary in a tree-like structure.

        Args:
            config: The configuration dictionary to print
            default_config: The default configuration dictionary to print
            tracking_dict: A dictionary with the same structure as config, whose leaf values contain the experiment name that last updated the value
        """

        _pretty_print(
            config, default_config=default_config, tracking_dict=tracking_dict
        )


def _update(
    target_config: dict,
    update_config: dict,
    tracking_dict: dict,
    experiment_name: str,
) -> None:
    """
    Recursively update target_dict in-place with values from update_dict, following specific rules for different types.

    For each value that gets updated, the corresponding value in tracking_dict is updated with experiment_name.

    Args:
        target_config: The config dictionary to be modified
        update_config: The config dictionary containing update values
        tracking_dict: A dictionary of nested dictionaries.
            If a value target_config gets overwritten, the same value in tracking_dict will be overwritten with `experiment_name`.
        experiment_name: The name of the current experiment

    Notes:
        - Nested dictionaries are recursively updated
        - Only updates existing keys (adding new keys not allowed)
        - lists are always overwritten

    Raises:
        - KeyAddedConfigError: a key is not found in the target_config
        - ValueTypeMismatchConfigError: the type of the update value does not match the type of the target value
    """
    for key, update_value in update_config.items():
        if key not in target_config:
            raise KeyAddedConfigError(key, experiment_name)

        target_value = target_config[key]
        tracking_value = tracking_dict[key]

        if (
            target_value is not None
            and type(target_value) != type(update_value)
            and not (
                isinstance(target_value, int | float)
                and isinstance(update_value, int | float)
            )
        ):
            raise TypeMismatchConfigError(
                key, experiment_name, f"{type(update_value)} != {type(target_value)}"
            )

        if isinstance(target_value, dict):
            _update(target_value, update_value, tracking_value, experiment_name)

        elif isinstance(target_value, list):
            # overwrite lists completely
            target_config[key] = update_value
            tracking_dict[key] = experiment_name

        # handle simple values
        else:
            target_config[key] = update_value
            tracking_dict[key] = experiment_name


def _pretty_print(
    config: dict,
    *,
    default_config: dict | list | None,
    tracking_dict: dict | str,
    prefix: str = "",
):
    """Recursively pretty print a configuration dictionary in a tree-like structure."""
    for i, (key, value) in enumerate(config.items()):
        is_last_item = i == len(config.items()) - 1

        # determine the current line's prefix
        current_prefix = "└──" if is_last_item else "├──"

        # determine the next level's prefix
        next_prefix = prefix + ("    " if is_last_item else "│   ")

        if default_config is None:
            # in case something was added
            default_config_value = None
        elif isinstance(default_config, dict):
            try:
                default_config_value = default_config[key]
            except KeyError:
                # in case a key was added
                default_config_value = None
        else:
            # we can assume it's a list here, as simple types are printed right away
            default_config_value = default_config[i]

        if isinstance(tracking_dict, str):
            # we have a leaf node (e.g. "default")
            tracking_dict_value = tracking_dict
        else:
            # tracking configs values are either dict or str (not lists: those are overwritten by experiment_name)
            tracking_dict_value = tracking_dict[key]

        if isinstance(value, dict):
            logger.info(f"{prefix}{current_prefix}{key}")
            _pretty_print(
                value,
                default_config=default_config_value,
                tracking_dict=tracking_dict_value,
                prefix=next_prefix,
            )
        elif isinstance(value, list):
            logger.info(f"{prefix}{current_prefix}{key}:")
            for j, value_ in enumerate(value):
                default_value = (
                    default_config_value[j]
                    if default_config_value is not None
                    and j < len(default_config_value)
                    else None
                )

                # complex lists
                if isinstance(value_, dict):
                    next_prefix = prefix + ("    " if is_last_item else "│   ")
                    _pretty_print(
                        value_,
                        default_config=default_value,
                        tracking_dict=tracking_dict_value,
                        prefix=next_prefix,
                    )

                # simple lists
                else:
                    color_on, color_off = _get_color_tokens(
                        value_, default_config_value
                    )
                    logger.info(
                        f"{next_prefix}{color_on}- {_expand(value_, default_value, tracking_dict_value)}{color_off}"
                    )

        # simple value
        else:
            color_on, color_off = _get_color_tokens(value, default_config_value)
            logger.info(
                f"{prefix}{color_on}{current_prefix}{key}: {_expand(value, default_config_value, tracking_dict_value)}{color_off}"
            )


def _get_color_tokens(
    actual_value: str | int | float,
    default_value: str | int | float,
) -> tuple[str, str]:
    """Get color on/off tokens if values differ, else empty strings."""

    if default_value != actual_value:
        style = "\x1b[32;20m"
        reset = "\x1b[0m"
        return style, reset
    return "", ""


def _expand(
    actual_value: str | int | float,
    default_value: str | int | float,
    tracking_value: str,
) -> str:
    """Create an expanded string representation of a configuration value in case it differs from the default."""
    msg = str(actual_value)

    if default_value != actual_value:
        return f"{msg} [{tracking_value}, default: {default_value}]"

    return msg

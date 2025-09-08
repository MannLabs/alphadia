import tempfile
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from alphabase.constants.modification import MOD_DF

from alphadia import search_step
from alphadia.search_step import SearchStep
from alphadia.workflow.config import Config


def test_custom_modifications():
    temp_directory = tempfile.gettempdir()

    config = {
        "custom_modifications": [
            {
                "name": "ThisModDoesNotExists@K",
                "composition": "H(10)",
            },
        ]
    }

    step = search_step.SearchStep(temp_directory, config=config)  # noqa F841
    assert "ThisModDoesNotExists@K" in MOD_DF["mod_name"].values


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_initializes_with_default_config(mock_load_default_config):
    """Test that the config is initialized with default values."""
    config = Config(
        {"key1": "value1", "key2": "value2"}, "default"
    )  # not using a mock here as working with the real object is much simpler
    mock_load_default_config.return_value = deepcopy(
        config
    )  # copy required here as we want to compare changes to a mutable object below

    # when
    result = SearchStep._init_config(None, None, None, "/output")

    mock_load_default_config.assert_called_once()
    assert result == config | {"output_directory": "/output"}


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_user_config_object(mock_load_default_config):
    """Test that the config is updated with user config object."""
    config = Config({"key1": "value1", "key2": "value2"})
    mock_load_default_config.return_value = deepcopy(config)

    user_config = Config({"key2": "value2b"})
    # when
    result = SearchStep._init_config(user_config, None, None, "/output")

    assert result == {
        "key1": "value1",
        "key2": "value2b",
        "output_directory": "/output",
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_user_and_cli_and_extra_config_dicts(
    mock_load_default_config,
):
    """Test that the config is updated with user, cli and extra config dicts."""
    config = Config(
        {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "key4": "value4",
            "output_directory": None,
        }
    )
    mock_load_default_config.return_value = deepcopy(config)

    user_config = {"key2": "value2b"}
    cli_config = {"key3": "value3b"}
    extra_config = {"key4": "value4b"}
    # when
    result = SearchStep._init_config(user_config, cli_config, extra_config, "/output")

    mock_load_default_config.assert_called_once()

    assert result == {
        "key1": "value1",
        "key2": "value2b",
        "key3": "value3b",
        "key4": "value4b",
        "output_directory": "/output",
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_cli_config_no_overwrite_output_path(
    mock_load_default_config,
):
    """Test that the output directory is not overwritten if provided by config."""
    config = Config({"key1": "value1", "output_directory": None})
    mock_load_default_config.return_value = deepcopy(config)

    user_config = {"key1": "value1b", "output_directory": "/output"}
    # when
    result = SearchStep._init_config(user_config, None, None, "/another_output")

    mock_load_default_config.assert_called_once()

    assert result == {"key1": "value1b", "output_directory": "/output"}


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_extra_config_overwrite_output_path(
    mock_load_default_config,
):
    """Test that the output directory is overwritten by extra_config."""
    config = Config({"key1": "value1", "output_directory": "/default_output"})
    mock_load_default_config.return_value = deepcopy(config)

    extra_config = {"key1": "value1b"}
    # when
    result = SearchStep._init_config(None, None, extra_config, "/extra_output")

    mock_load_default_config.assert_called_once()

    assert result == {"key1": "value1b", "output_directory": "/extra_output"}


@pytest.mark.parametrize(
    ("config1", "config2", "config3"),
    [
        ("not_dict_nor_config_object", None, None),
        (None, "not_dict_nor_config_object", None),
        (None, None, "not_dict_nor_config_object"),
    ],
)
@patch("alphadia.search_step.SearchStep._load_default_config")
def test_raises_value_error_for_invalid_config(
    mock_load_default_config, config1, config2, config3
):
    """Test that a TypeError is raised if the config is not a dict or Config object."""
    mock_load_default_config.return_value = MagicMock(spec=Config)

    with pytest.raises(TypeError, match="'str' object is not a mapping"):
        # when
        SearchStep._init_config(config1, config2, config3, "/output")

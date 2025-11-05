import tempfile
from copy import deepcopy
from unittest.mock import MagicMock, call, patch

import pytest
from alphabase.constants.modification import MOD_DF

import alphadia
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
    default_config = Config(
        {
            "key1": "value1",
            "key2": "value2",
            "search": {"extraction_backend": "python"},
        },
        "default",
    )  # not using a mock here as working with the real object is much simpler
    mock_load_default_config.return_value = deepcopy(
        default_config
    )  # copy required here as we want to compare changes to a mutable object below

    # when
    result = SearchStep._init_config(None, None, None, "/output")

    mock_load_default_config.assert_called_once()
    assert result == default_config | {"output_directory": "/output", "version": alphadia.__version__}


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_user_config_object(mock_load_default_config):
    """Test that the config is updated with user config object."""
    default_config = Config(
        {
            "key1": "value1",
            "key2": "value2",
            "search": {"extraction_backend": "python"},
        }
    )
    mock_load_default_config.return_value = deepcopy(default_config)

    user_config = Config({"key2": "NEW_value2"})
    # when
    result = SearchStep._init_config(user_config, None, None, "/output")

    assert result == {
        "key1": "value1",
        "key2": "NEW_value2",
        "output_directory": "/output",
        "search": {"extraction_backend": "python"},
        "version": alphadia.__version__,
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_user_and_cli_and_extra_config_dicts(
    mock_load_default_config,
):
    """Test that the config is updated with user, cli and extra config dicts."""
    default_config = Config(
        {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "key4": "value4",
            "output_directory": None,
            "search": {"extraction_backend": "python"},
        }
    )
    mock_load_default_config.return_value = deepcopy(default_config)

    user_config = {
        "key2": "NEW_value2",
        "key3": "GET_OVERWRITTEN_value3",
        "key4": "GETS_OVERWRITTEN_value4",
    }
    cli_config = {"key3": "NEW_value3", "key4": "GETS_OVERWRITTEN_value4"}
    extra_config = {"key4": "NEW_value4"}
    # when
    result = SearchStep._init_config(user_config, cli_config, extra_config, "/output")

    mock_load_default_config.assert_called_once()

    assert result == {
        "key1": "value1",
        "key2": "NEW_value2",
        "key3": "NEW_value3",
        "key4": "NEW_value4",
        "output_directory": "/output",
        "search": {"extraction_backend": "python"},
        "version": alphadia.__version__,
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_cli_config_overwrite_output_path(
    mock_load_default_config,
):
    """Test that the output directory is not overwritten if provided by config."""
    default_config = Config(
        {
            "key1": "value1",
            "output_directory": None,
            "search": {"extraction_backend": "python"},
        }
    )
    mock_load_default_config.return_value = deepcopy(default_config)

    user_config = {"key1": "NEW_value1", "output_directory": "/output"}

    # when
    result = SearchStep._init_config(
        user_config, None, None, "/actual_output_directory"
    )

    mock_load_default_config.assert_called_once()

    assert result == {
        "key1": "NEW_value1",
        "output_directory": "/actual_output_directory",
        "search": {"extraction_backend": "python"},
        "version": alphadia.__version__,
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_extra_config_overwrite_output_path(
    mock_load_default_config,
):
    """Test that the output directory is overwritten by extra_config."""
    default_config = Config(
        {
            "key1": "value1",
            "output_directory": "/default_output",
            "search": {"extraction_backend": "python"},
        }
    )
    mock_load_default_config.return_value = deepcopy(default_config)

    extra_config = {"key1": "NEW_value1"}
    # when
    result = SearchStep._init_config(None, None, extra_config, "/extra_output")

    mock_load_default_config.assert_called_once()

    assert result == {
        "key1": "NEW_value1",
        "output_directory": "/extra_output",
        "search": {"extraction_backend": "python"},
        "version": alphadia.__version__,
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_user_config_object_ng_backend(mock_load_default_config):
    """Test that the correct defaults are loaded if extraction backend is "rust"."""
    default_config = Config(
        {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "search": {"extraction_backend": "python"},
        }
    )
    default_config_ng = Config(
        {
            "key1": "NEW_NG_DEFAULT1",
            "key2": "NEW_NG_DEFAULT2",
        }
    )
    mock_load_default_config.side_effect = [
        deepcopy(default_config),
        deepcopy(default_config_ng),
    ]

    user_config = Config(
        {"search": {"extraction_backend": "rust"}, "key2": "some_user_value"}
    )

    # when
    result = SearchStep._init_config(user_config, None, None, "/output")

    assert result == {
        "key1": "NEW_NG_DEFAULT1",  # taken from ng default
        "key2": "some_user_value",  # overwritten by user although ng default exists
        "key3": "value3",
        "output_directory": "/output",
        "search": {"extraction_backend": "rust"},
        "version": alphadia.__version__,
    }
    mock_load_default_config.assert_has_calls(
        [call(), call(file_name="default_rust.yaml")]
    )


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

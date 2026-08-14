import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from alphabase.constants.modification import MOD_DF

from alphadia import __version__ as alphadia_version
from alphadia import search_step
from alphadia.exceptions import ConfigError
from alphadia.search_step import SearchStep
from alphadia.workflow.config import Config

QUANT_FILE_NAMES = ("psm.parquet", "frag.parquet")


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
            "search": {"extraction_backend": "rust"},
            "library_prediction": {"peptdeep_model_path": None},
            "general": {"reuse_quant_from": []},
        },
        "default",
    )  # not using a mock here as working with the real object is much simpler
    mock_load_default_config.return_value = deepcopy(
        default_config
    )  # copy required here as we want to compare changes to a mutable object below

    # when
    result = SearchStep._init_config(None, None, None, "/output")

    mock_load_default_config.assert_called_once()
    assert result == default_config | {
        "output_directory": "/output",
        "version": alphadia_version,
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_user_config_object(mock_load_default_config):
    """Test that the config is updated with user config object."""
    default_config = Config(
        {
            "key1": "value1",
            "key2": "value2",
            "search": {"extraction_backend": "rust"},
            "library_prediction": {"peptdeep_model_path": None},
            "general": {"reuse_quant_from": []},
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
        "search": {"extraction_backend": "rust"},
        "version": alphadia_version,
        "library_prediction": {"peptdeep_model_path": None},
        "general": {"reuse_quant_from": []},
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
            "search": {"extraction_backend": "rust"},
            "library_prediction": {"peptdeep_model_path": None},
            "general": {"reuse_quant_from": []},
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
        "search": {"extraction_backend": "rust"},
        "version": alphadia_version,
        "library_prediction": {"peptdeep_model_path": None},
        "general": {"reuse_quant_from": []},
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
            "search": {"extraction_backend": "rust"},
            "library_prediction": {"peptdeep_model_path": None},
            "general": {"reuse_quant_from": []},
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
        "search": {"extraction_backend": "rust"},
        "version": alphadia_version,
        "library_prediction": {"peptdeep_model_path": None},
        "general": {"reuse_quant_from": []},
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
            "search": {"extraction_backend": "rust"},
            "library_prediction": {"peptdeep_model_path": None},
            "general": {"reuse_quant_from": []},
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
        "search": {"extraction_backend": "rust"},
        "version": alphadia_version,
        "library_prediction": {"peptdeep_model_path": None},
        "general": {"reuse_quant_from": []},
    }


@patch("alphadia.search_step.SearchStep._load_default_config")
def test_updates_with_user_config_object_python_backend(mock_load_default_config):
    """Test that the correct defaults are loaded if extraction backend is "python"."""
    default_config = Config(
        {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "search": {"extraction_backend": "rust"},
            "library_prediction": {"peptdeep_model_path": None},
            "general": {"reuse_quant_from": []},
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
        {"search": {"extraction_backend": "python"}, "key2": "some_user_value"}
    )

    # when
    result = SearchStep._init_config(user_config, None, None, "/output")

    assert result == {
        "key1": "NEW_NG_DEFAULT1",  # taken from ng default
        "key2": "some_user_value",  # overwritten by user although ng default exists
        "key3": "value3",
        "output_directory": "/output",
        "search": {"extraction_backend": "python"},
        "version": alphadia_version,
        "library_prediction": {"peptdeep_model_path": None},
        "general": {"reuse_quant_from": []},
    }
    mock_load_default_config.assert_has_calls(
        [call(), call(file_name="default_python.yaml")]
    )


def _create_quant_folder(
    quant_directory: Path, raw_name: str, file_names: tuple[str, ...] = QUANT_FILE_NAMES
) -> Path:
    """Create a quant folder for `raw_name` holding empty `file_names`."""
    folder = quant_directory / raw_name
    folder.mkdir(parents=True)
    for file_name in file_names:
        (folder / file_name).touch()

    return folder


def test_get_reusable_quant_folder_returns_none_if_reuse_not_configured(tmp_path):
    """Test that no quant folder is reused if reuse is not configured."""
    _create_quant_folder(tmp_path / "output" / "quant", "raw1")
    step = SearchStep(str(tmp_path / "output"))

    # when
    assert step._get_reusable_quant_folder("raw1") is None


def test_get_reusable_quant_folder_from_own_quant_directory(tmp_path):
    """Test that the quant folder of the step itself is reused if `reuse_quant` is set."""
    folder = _create_quant_folder(tmp_path / "output" / "quant", "raw1")
    step = SearchStep(
        str(tmp_path / "output"), config={"general": {"reuse_quant": True}}
    )

    # when
    assert step._get_reusable_quant_folder("raw1") == str(folder)


def test_get_reusable_quant_folder_from_other_quant_directory(tmp_path):
    """Test that a quant folder from `reuse_quant_from` is reused without being written to."""
    folder = _create_quant_folder(tmp_path / "previous_run" / "quant", "raw1")
    step = SearchStep(
        str(tmp_path / "output"),
        config={
            "general": {"reuse_quant_from": [str(tmp_path / "previous_run/quant")]}
        },
    )

    # when
    assert step._get_reusable_quant_folder("raw1") == str(folder)
    assert sorted(p.name for p in folder.iterdir()) == sorted(QUANT_FILE_NAMES)


def test_get_reusable_quant_folder_returns_none_for_incomplete_folder(tmp_path):
    """Test that an incomplete quant folder is not reused."""
    _create_quant_folder(
        tmp_path / "previous_run" / "quant", "raw1", file_names=("psm.parquet",)
    )
    step = SearchStep(
        str(tmp_path / "output"),
        config={
            "general": {"reuse_quant_from": [str(tmp_path / "previous_run/quant")]}
        },
    )

    # when
    assert step._get_reusable_quant_folder("raw1") is None


def test_get_reusable_quant_folder_requires_transfer_file(tmp_path):
    """Test that the transfer fragment file is required if the transfer library is enabled."""
    _create_quant_folder(tmp_path / "previous_run" / "quant", "raw1")
    step = SearchStep(
        str(tmp_path / "output"),
        config={
            "general": {"reuse_quant_from": [str(tmp_path / "previous_run/quant")]},
            "transfer_library": {"enabled": True},
        },
    )

    # when
    assert step._get_reusable_quant_folder("raw1") is None


def test_get_reusable_quant_folder_raises_on_duplicate_raw_file(tmp_path):
    """Test that a raw file found in more than one quant directory raises."""
    _create_quant_folder(tmp_path / "previous_run_1" / "quant", "raw1")
    _create_quant_folder(tmp_path / "previous_run_2" / "quant", "raw1")
    step = SearchStep(
        str(tmp_path / "output"),
        config={
            "general": {
                "reuse_quant_from": [
                    str(tmp_path / "previous_run_1/quant"),
                    str(tmp_path / "previous_run_2/quant"),
                ]
            }
        },
    )

    with pytest.raises(ConfigError, match="CONFIG_ERROR"):
        # when
        step._get_reusable_quant_folder("raw1")


def test_raises_for_nonexistent_reuse_quant_from_directory(tmp_path):
    """Test that a nonexistent directory in `reuse_quant_from` raises on initialization."""
    with pytest.raises(ConfigError, match="CONFIG_ERROR"):
        # when
        SearchStep(
            str(tmp_path / "output"),
            config={
                "general": {"reuse_quant_from": [str(tmp_path / "does_not_exist")]}
            },
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

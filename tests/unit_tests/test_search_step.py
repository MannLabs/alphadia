import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
from alphabase.constants.modification import MOD_DF

from alphadia import __version__ as alphadia_version
from alphadia import search_step
from alphadia.exceptions import ConfigError, GenericUserError
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
            "search": {"extraction_backend": "rust"},
            "library_prediction": {"peptdeep_model_path": None},
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
    }
    mock_load_default_config.assert_has_calls(
        [call(), call(file_name="default_python.yaml")]
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


def _get_search_step_for_transfer_library(
    transfer_library_config: dict, quant_directory: str, raw_path_list: list[str]
) -> SearchStep:
    """Get a SearchStep instance with the minimal state required for `_validate_transfer_library`."""
    step = object.__new__(SearchStep)
    step._config = {
        "transfer_library": transfer_library_config,
        "quant_directory": quant_directory,
    }
    step.raw_path_list = raw_path_list
    return step


def test_validate_transfer_library_passes_if_not_reusing_quant():
    """Test that no quantification results are required if reusing is not requested."""
    step = _get_search_step_for_transfer_library(
        {"enabled": False, "reuse_quant": False}, "/does/not/exist", ["/raw/file1.raw"]
    )

    # when
    step._validate_transfer_library()


def test_validate_transfer_library_raises_if_transfer_library_not_enabled():
    """Test that reusing quantification results requires the transfer library to be enabled."""
    step = _get_search_step_for_transfer_library(
        {"enabled": False, "reuse_quant": True}, "/does/not/exist", ["/raw/file1.raw"]
    )

    with pytest.raises(ConfigError, match="transfer_library.enabled"):
        # when
        step._validate_transfer_library()


def test_validate_transfer_library_raises_if_quant_results_are_missing():
    """Test that missing quantification results are reported for all raw files up front."""
    with tempfile.TemporaryDirectory() as quant_directory:
        (Path(quant_directory) / "file1" / "figures").mkdir(parents=True)
        (Path(quant_directory) / "file1" / "psm.parquet").touch()

        step = _get_search_step_for_transfer_library(
            {"enabled": True, "reuse_quant": True},
            quant_directory,
            ["/raw/file1.raw", "/raw/file2.raw"],
        )

        with pytest.raises(ConfigError, match="file1.*file2"):
            # when
            step._validate_transfer_library()


def test_validate_transfer_library_passes_if_quant_results_are_present():
    """Test that complete quantification results for all raw files are accepted."""
    with tempfile.TemporaryDirectory() as quant_directory:
        for raw_name in ["file1", "file2"]:
            (Path(quant_directory) / raw_name).mkdir()
            (Path(quant_directory) / raw_name / "psm.parquet").touch()
            (Path(quant_directory) / raw_name / "frag.parquet").touch()

        step = _get_search_step_for_transfer_library(
            {"enabled": True, "reuse_quant": True},
            quant_directory,
            ["/raw/file1.raw", "/raw/file2.raw"],
        )

        # when
        step._validate_transfer_library()


def test_harmonize_modification_names_translates_legacy_names():
    """Test that terminal modifications of older alphabase versions are translated."""
    mods = pd.Series(
        [
            "Acetyl@Protein N-term",
            "Oxidation@M;Acetyl@Protein N-term",
            "Carbamidomethyl@C",
            "",
        ]
    )

    # when
    result = search_step._harmonize_modification_names(mods)

    pd.testing.assert_series_equal(
        result,
        pd.Series(
            [
                "Acetyl@Protein_N-term",
                "Oxidation@M;Acetyl@Protein_N-term",
                "Carbamidomethyl@C",
                "",
            ]
        ),
    )


def test_harmonize_modification_names_raises_for_unknown_modification():
    """Test that modifications unknown to the installed alphabase version are reported."""
    mods = pd.Series(["Oxidation@M;ThisModDoesNotExist@K"])

    with pytest.raises(GenericUserError, match="ThisModDoesNotExist@K"):
        # when
        search_step._harmonize_modification_names(mods)

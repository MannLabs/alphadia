from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphadia.exceptions import CustomError
from alphadia.search_plan import SearchPlan

MOCK_DEFAULT_CONFIG = {
    "general": {
        "transfer_step_enabled": False,
        "adaptation_method": "transfer",
        "mbr_step_enabled": False,
    },
}

MOCK_MULTISTEP_CONFIG = {
    "transfer": {"some_adaptation_config_key": "some_adaptation_config_value"},
    "library": {"some_library_config_key": "some_library_config_value"},
    "mbr": {"some_mbr_config_key": "some_mbr_config_value"},
}

BASE_USER_CONFIG = {
    "some_user_config_key": "some_user_config_value",
}

BASE_CLI_PARAMS_CONFIG = {
    "raw_paths": ["/raw1"],
    "library_path": "/user_provided_library_path",
    "fasta_paths": ["/fasta1"],
    "quant_directory": "/user_provided_quant_path",
}


def _convert_path(path_str: str) -> str:
    """Conversion to and back from Path necessary for Windows compatibility."""
    return str(Path(path_str))


def get_search_plan(
    config,
    cli_params_config=BASE_CLI_PARAMS_CONFIG,  # noqa: B006
    configs=[MOCK_DEFAULT_CONFIG, MOCK_MULTISTEP_CONFIG],  # noqa: B006
):
    """Helper function to create a SearchPlan object with a given config."""
    with patch("alphadia.search_plan.yaml") as mock_yaml:
        mock_yaml.safe_load.side_effect = configs
        return SearchPlan(
            output_directory="/user_provided_output_path",
            config=config,
            cli_params_config=cli_params_config,
        )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
def test_runs_plan_without_transfer_and_mbr_steps(mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly without adaptation and mbr steps."""
    search_plan = get_search_plan(BASE_USER_CONFIG)

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # could use `mock_plan.assert_has_calls([call(..)])` pattern here but it is harder to read in case of error
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": str(Path("/user_provided_output_path")),
        "config": BASE_USER_CONFIG,
        "extra_config": {},
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "library",
    }

    mock_plan.return_value.run.assert_called_once_with()


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
def test_runs_plan_without_transfer_and_mbr_steps_none_dirs(
    mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly without adaptation and mbr steps when all parameters are none or empty."""

    search_plan = get_search_plan({}, {})

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # could use `mock_plan.assert_has_calls([call(..)])` pattern here but it is harder to read in case of error
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path"),
        "config": {},
        "extra_config": {},
        "cli_config": {},
        "step_name": "library",
    }

    mock_plan.return_value.run.assert_called_once_with()


@pytest.mark.parametrize(
    ("transfer_step_enabled", "adaptation_method", "mbr_step_enabled"),
    [
        (True, "transfer", True),
        (True, "tto", False),
        (False, "transfer", True),
        (False, "transfer", False),
    ],
)
@patch("alphadia.search_plan.reporting.init_logging")
def test_default_correctly_read(
    mock_init_logging,
    transfer_step_enabled,
    adaptation_method,
    mbr_step_enabled,
):
    """Test that the defaults are correctly propagated to the class internals."""

    search_plan = get_search_plan(
        {},
        {},
        [
            {
                "general": {
                    "transfer_step_enabled": transfer_step_enabled,
                    "adaptation_method": adaptation_method,
                    "mbr_step_enabled": mbr_step_enabled,
                }
            },
            {},
        ],
    )

    assert search_plan._transfer_step_enabled == transfer_step_enabled
    assert search_plan._adaptation_method == adaptation_method
    assert search_plan._mbr_step_enabled == mbr_step_enabled


@patch("alphadia.search_plan.reporting.init_logging")
def test_raises_on_invalid_adaptation_method(mock_init_logging):
    """Test that an invalid adaptation_method raises a CustomError."""
    with pytest.raises(CustomError):
        get_search_plan(
            {
                "general": {
                    "transfer_step_enabled": True,
                    "adaptation_method": "invalid_method",
                }
            }
        )


@patch("alphadia.search_plan.reporting.init_logging")
def test_old_config_no_adaptation_method_defaults_to_transfer(mock_init_logging):
    """Backward compat: user config has transfer_step_enabled but no adaptation_method.

    adaptation_method should default to "transfer" from default.yaml so the pipeline
    behaves identically to the pre-adaptation_method era.
    """
    search_plan = get_search_plan(
        {"general": {"transfer_step_enabled": True}},
        {},
        [MOCK_DEFAULT_CONFIG, MOCK_MULTISTEP_CONFIG],
    )

    assert search_plan._adaptation_method == "transfer"
    assert search_plan._transfer_step_enabled is True


@patch("alphadia.search_plan.reporting.init_logging")
def test_old_config_no_transfer_step_defaults_to_disabled(mock_init_logging):
    """Backward compat: completely minimal user config with no step flags at all.

    Both transfer_step_enabled and adaptation_method should fall back to defaults
    (False and "transfer" respectively), so a single library step runs.
    """
    search_plan = get_search_plan({}, {}, [MOCK_DEFAULT_CONFIG, MOCK_MULTISTEP_CONFIG])

    assert search_plan._transfer_step_enabled is False
    assert search_plan._adaptation_method == "transfer"
    assert search_plan._mbr_step_enabled is False


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_old_config_runs_transfer_path_without_adaptation_method(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Backward compat: user config with transfer_step_enabled but no adaptation_method
    should run exactly the same as explicitly setting adaptation_method: "transfer".
    """
    # Old-style config: no adaptation_method key at all
    old_user_config = {"general": {"transfer_step_enabled": True}}

    search_plan = get_search_plan(old_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    search_plan.run_plan()

    # transfer step should use transfer_learning (the pre-TTO default path)
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/transfer"),
        "config": old_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"]
        | {"transfer_learning": {"enabled": True}},
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "transfer",
    }

    # library step should receive the peptdeep model path (transfer method output)
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path"),
        "config": old_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "peptdeep_model_path": _convert_path(
                    "/user_provided_output_path/transfer/peptdeep.transfer"
                ),
                "enabled": True,
                "use_peptdeep_kontext": False,
            },
        }
        | dynamic_config,
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "library",
    }

    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_adaptation_step_transfer(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with the adaptation step (transfer method)."""
    additional_user_config = {
        "general": {
            "transfer_step_enabled": True,
            "adaptation_method": "transfer",
            "mbr_step_enabled": False,
        }
    }

    search_plan = get_search_plan(BASE_USER_CONFIG | additional_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # adaptation_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/transfer"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"]
        | {"transfer_learning": {"enabled": True}},
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "transfer",
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "peptdeep_model_path": _convert_path(
                    "/user_provided_output_path/transfer/peptdeep.transfer"
                ),
                "enabled": True,
                "use_peptdeep_kontext": False,
            },
        }
        | dynamic_config,
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "library",
    }

    assert mock_plan.return_value.run.call_count == 2
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_adaptation_step_tto(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with the adaptation step (tto method)."""
    additional_user_config = {
        "general": {
            "transfer_step_enabled": True,
            "adaptation_method": "tto",
            "mbr_step_enabled": False,
        }
    }

    search_plan = get_search_plan(BASE_USER_CONFIG | additional_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # adaptation_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/transfer"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"]
        | {"context_extraction": {"enabled": True}},
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "transfer",
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "context_path": _convert_path(
                    "/user_provided_output_path/transfer/peptdeep_kontext.context"
                ),
                "enabled": True,
                "use_peptdeep_kontext": True,
            },
        }
        | dynamic_config,
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "library",
    }

    assert mock_plan.return_value.run.call_count == 2
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_mbr_step(mock_get_dyn_config, mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly with the mbr step enabled."""
    additional_user_config = {
        "general": {
            "transfer_step_enabled": False,
            "mbr_step_enabled": True,
        }
    }

    search_plan = get_search_plan(BASE_USER_CONFIG | additional_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # library_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/library"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"],
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "library",
    }

    # mbr_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["mbr"]
        | dynamic_config
        | {
            "library_path": str(
                Path("/user_provided_output_path/library/speclib.mbr.hdf")
            )
        },
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "mbr",
    }

    assert mock_plan.return_value.run.call_count == 2
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/library")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_adaptation_transfer_and_mbr_steps(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with both the adaptation (transfer) and mbr steps enabled."""
    additional_user_config = {
        "general": {
            "transfer_step_enabled": True,
            "adaptation_method": "transfer",
            "mbr_step_enabled": True,
        }
    }

    search_plan = get_search_plan(BASE_USER_CONFIG | additional_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # adaptation_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/transfer"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"]
        | {"transfer_learning": {"enabled": True}},
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "transfer",
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/library"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "peptdeep_model_path": str(
                    Path("/user_provided_output_path/transfer/peptdeep.transfer")
                ),
                "enabled": True,
                "use_peptdeep_kontext": False,
            },
        }
        | dynamic_config,
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "library",
    }

    # mbr_step
    assert mock_plan.call_args_list[2].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["mbr"]
        | dynamic_config
        | {
            "library_path": str(
                Path("/user_provided_output_path/library/speclib.mbr.hdf")
            ),
        },
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "mbr",
    }

    assert mock_plan.return_value.run.call_count == 3
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_adaptation_tto_and_mbr_steps(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with both the adaptation (tto) and mbr steps enabled."""
    additional_user_config = {
        "general": {
            "transfer_step_enabled": True,
            "adaptation_method": "tto",
            "mbr_step_enabled": True,
        }
    }

    search_plan = get_search_plan(BASE_USER_CONFIG | additional_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # adaptation_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/transfer"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"]
        | {"context_extraction": {"enabled": True}},
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "transfer",
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path/library"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "context_path": str(
                    Path("/user_provided_output_path/transfer/peptdeep_kontext.context")
                ),
                "enabled": True,
                "use_peptdeep_kontext": True,
            },
        }
        | dynamic_config,
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "library",
    }

    # mbr_step
    assert mock_plan.call_args_list[2].kwargs == {
        "output_folder": _convert_path("/user_provided_output_path"),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["mbr"]
        | dynamic_config
        | {
            "library_path": str(
                Path("/user_provided_output_path/library/speclib.mbr.hdf")
            ),
        },
        "cli_config": BASE_CLI_PARAMS_CONFIG,
        "step_name": "mbr",
    }

    assert mock_plan.return_value.run.call_count == 3
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


@pytest.mark.parametrize(
    ("input_data", "expected_output"),
    [
        (
            ([10, 20, np.nan], [20, np.nan, 30]),
            {"search": {"target_ms1_tolerance": 15.0, "target_ms2_tolerance": 25.0}},
        ),
        (
            ([np.nan, np.nan, np.nan], [20, np.nan, 30]),
            {"search": {"target_ms2_tolerance": 25.0}},
        ),
        (
            ([10, 20, np.nan], [np.nan, np.nan, np.nan]),
            {"search": {"target_ms1_tolerance": 15.0}},
        ),
        (
            ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]),
            {},
        ),
    ],
)
def test_get_optimized_values_config(input_data, expected_output):
    """Test that the SearchPlan object updates the config correct data, incl. handling NaNs."""

    df = pd.DataFrame(
        {
            "optimization.ms1_error": input_data[0],
            "optimization.ms2_error": input_data[1],
        }
    )

    output_dir = MagicMock(wraps=Path)

    # when
    with patch("alphadia.search_plan.pd.read_csv", return_value=df) as mock_read_csv:
        extra_config = SearchPlan._get_optimized_values_config(output_dir)

    assert extra_config == expected_output
    mock_read_csv.assert_called_once_with(output_dir / "stat_output.tsv", sep="\t")

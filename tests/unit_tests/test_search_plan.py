from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from alphadia.search_plan import SearchPlan

MOCK_MULTISTEP_CONFIG = {
    "transfer": {"some_transfer_config_key": "some_transfer_config_value"},
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


def get_search_plan(config):
    """Helper function to create a SearchPlan object with a given config."""
    with patch(
        "alphadia.search_plan.yaml.safe_load", return_value=MOCK_MULTISTEP_CONFIG
    ):
        return SearchPlan(
            output_directory="/user_provided_output_path",
            config=config,
            cli_params_config=BASE_CLI_PARAMS_CONFIG,
        )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
def test_runs_plan_without_transfer_and_mbr_steps(mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly without transfer and mbr steps."""
    search_plan = get_search_plan(BASE_USER_CONFIG)

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # could use `mock_plan.assert_has_calls([call(..)])` pattern here but it is harder to read in case of error
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": str(
            Path("/user_provided_output_path")
        ),  # conversion to and from Path necessary for windows compatibility
        "config": BASE_USER_CONFIG,
        "extra_config": {},
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    mock_plan.return_value.run.assert_called_once_with()


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
def test_runs_plan_without_transfer_and_mbr_steps_none_dirs(
    mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly without transfer and mbr steps when all parameters are none or empty."""

    search_plan = SearchPlan(
        output_directory="/user_provided_output_path", config={}, cli_params_config={}
    )

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # could use `mock_plan.assert_has_calls([call(..)])` pattern here but it is harder to read in case of error
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": str(Path("/user_provided_output_path")),
        "config": {},
        "extra_config": {},
        "cli_config": {},
    }

    mock_plan.return_value.run.assert_called_once_with()


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_transfer_step(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with the transfer step enabled."""
    additional_user_config = {
        "multistep_search": {
            "transfer_step_enabled": True,
            "mbr_step_enabled": False,
        }
    }

    search_plan = get_search_plan(BASE_USER_CONFIG | additional_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # transfer_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": str(Path("/user_provided_output_path/transfer")),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"],
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": str(Path("/user_provided_output_path")),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "peptdeep_model_path": "/user_provided_output_path/transfer/peptdeep.transfer",
                "predict": True,
            },
        }
        | dynamic_config,
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    mock_plan.return_value.run.assert_has_calls([call(), call()])
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_mbr_step(mock_get_dyn_config, mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly with the mbr step enabled."""
    additional_user_config = {
        "multistep_search": {
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
        "output_folder": str(Path("/user_provided_output_path/library")),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"],
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    # mbr_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": str(Path("/user_provided_output_path")),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["mbr"]
        | dynamic_config
        | {
            "library_path": str(
                Path("/user_provided_output_path/library/speclib.mbr.hdf")
            )
        },
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    mock_plan.return_value.run.assert_has_calls([call(), call()])
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/library")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.SearchStep")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_transfer_and_mbr_steps(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with both the transfer and mbr steps enabled."""
    additional_user_config = {
        "multistep_search": {
            "transfer_step_enabled": True,
            "mbr_step_enabled": True,
        }
    }

    search_plan = get_search_plan(BASE_USER_CONFIG | additional_user_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # transfer_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": str(Path("/user_provided_output_path/transfer")),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"],
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": str(Path("/user_provided_output_path/library")),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "peptdeep_model_path": str(
                    Path("/user_provided_output_path/transfer/peptdeep.transfer")
                ),
                "predict": True,
            },
        }
        | dynamic_config,
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    # mbr_step
    assert mock_plan.call_args_list[2].kwargs == {
        "output_folder": str(Path("/user_provided_output_path")),
        "config": BASE_USER_CONFIG | additional_user_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["mbr"]
        | dynamic_config
        | {
            "library_path": str(
                Path("/user_provided_output_path/library/speclib.mbr.hdf")
            ),
        },
        "cli_config": BASE_CLI_PARAMS_CONFIG,
    }

    mock_plan.return_value.run.assert_has_calls([call(), call(), call()])
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

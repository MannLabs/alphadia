from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd

from alphadia.search_plan import SearchPlan

MOCK_MULTISTEP_CONFIG = {
    "transfer": {"some_transfer_config_key": "some_transfer_config_value"},
    "library": {"some_library_config_key": "some_library_config_value"},
    "mbr": {"some_mbr_config_key": "some_mbr_config_value"},
}

USER_CONFIG = {
    "some_user_config_key": "some_user_config_value",
}


def get_search_plan(config):
    """Helper function to create a SearchPlan object with a given config."""
    with patch(
        "alphadia.search_plan.yaml.safe_load", return_value=MOCK_MULTISTEP_CONFIG
    ):
        return SearchPlan(
            output_directory="/user_provided_output_path",
            raw_path_list=["/raw1"],
            library_path="/user_provided_library_path",
            fasta_path_list=["/fasta1"],
            config=config,
            quant_dir="/user_provided_quant_path",
        )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
def test_runs_plan_without_transfer_and_mbr_steps(mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly without transfer and mbr steps."""
    search_plan = get_search_plan(USER_CONFIG)

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # could use `mock_plan.assert_has_calls([call(..)])` pattern here but it is harder to read in case of error
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": "/user_provided_output_path",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_library_path",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG,
        "extra_config": {},
        "quant_path": "/user_provided_quant_path",
    }

    mock_plan.return_value.run.assert_called_once_with()


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
def test_runs_plan_without_transfer_and_mbr_steps_none_dirs(
    mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly without transfer and mbr steps when all parameters are none or empty."""

    search_plan = SearchPlan(
        output_directory="/user_provided_output_path",
        raw_path_list=[],
        library_path=None,
        fasta_path_list=[],
        config={},
        quant_dir=None,
    )

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # could use `mock_plan.assert_has_calls([call(..)])` pattern here but it is harder to read in case of error
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": "/user_provided_output_path",
        "raw_path_list": [],
        "library_path": None,
        "fasta_path_list": [],
        "config": {},
        "extra_config": {},
        "quant_path": None,
    }

    mock_plan.return_value.run.assert_called_once_with()


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_transfer_step(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with the transfer step enabled."""
    multistep_search_config = {
        "multistep_search": {
            "transfer_step_enabled": True,
            "mbr_step_enabled": False,
        }
    }

    search_plan = get_search_plan(USER_CONFIG | multistep_search_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # transfer_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": "/user_provided_output_path/transfer",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_library_path",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG | multistep_search_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"],
        "quant_path": "/user_provided_quant_path",
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": "/user_provided_output_path",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_library_path",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG | multistep_search_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "peptdeep_model_path": "/user_provided_output_path/transfer/peptdeep.transfer",
                "predict": True,
            },
        }
        | dynamic_config,
        "quant_path": "/user_provided_output_path/transfer/quant",
    }

    mock_plan.return_value.run.assert_has_calls([call(), call()])
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_mbr_step(mock_get_dyn_config, mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly with the mbr step enabled."""
    multistep_search_config = {
        "multistep_search": {
            "transfer_step_enabled": False,
            "mbr_step_enabled": True,
        }
    }

    search_plan = get_search_plan(USER_CONFIG | multistep_search_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # library_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": "/user_provided_output_path/library",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_library_path",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG | multistep_search_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"],
        "quant_path": "/user_provided_quant_path",
    }

    # mbr_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": "/user_provided_output_path",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_output_path/library/speclib.mbr.hdf",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG | multistep_search_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["mbr"] | dynamic_config,
        "quant_path": "/user_provided_output_path/library/quant",
    }

    mock_plan.return_value.run.assert_has_calls([call(), call()])
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/library")
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
@patch("alphadia.search_plan.SearchPlan._get_optimized_values_config")
def test_runs_plan_with_transfer_and_mbr_steps(
    mock_get_dyn_config, mock_plan, mock_init_logging
):
    """Test that the SearchPlan object runs the plan correctly with both the transfer and mbr steps enabled."""
    multistep_search_config = {
        "multistep_search": {
            "transfer_step_enabled": True,
            "mbr_step_enabled": True,
        }
    }

    search_plan = get_search_plan(USER_CONFIG | multistep_search_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/user_provided_output_path")

    # transfer_step
    assert mock_plan.call_args_list[0].kwargs == {
        "output_folder": "/user_provided_output_path/transfer",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_library_path",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG | multistep_search_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["transfer"],
        "quant_path": "/user_provided_quant_path",
    }

    # library_step
    assert mock_plan.call_args_list[1].kwargs == {
        "output_folder": "/user_provided_output_path/library",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_library_path",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG | multistep_search_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["library"]
        | {
            "library_prediction": {
                "peptdeep_model_path": "/user_provided_output_path/transfer/peptdeep.transfer",
                "predict": True,
            },
        }
        | dynamic_config,
        "quant_path": "/user_provided_output_path/transfer/quant",
    }

    # mbr_step
    assert mock_plan.call_args_list[2].kwargs == {
        "output_folder": "/user_provided_output_path",
        "raw_path_list": ["/raw1"],
        "library_path": "/user_provided_output_path/library/speclib.mbr.hdf",
        "fasta_path_list": ["/fasta1"],
        "config": USER_CONFIG | multistep_search_config,
        "extra_config": MOCK_MULTISTEP_CONFIG["mbr"] | dynamic_config,
        "quant_path": "/user_provided_output_path/library/quant",
    }

    mock_plan.return_value.run.assert_has_calls([call(), call(), call()])
    mock_get_dyn_config.assert_called_once_with(
        Path("/user_provided_output_path/transfer")
    )


def test_get_optimized_values_config():
    """Test that the SearchPlan object updates the config correct data, incl. handling NaNs."""

    df = pd.DataFrame({"ms1_error": [10, 20, np.nan], "ms2_error": [20, np.nan, 30]})

    output_dir = MagicMock(wraps=Path)

    # when
    with patch("alphadia.search_plan.pd.read_csv", return_value=df) as mock_read_csv:
        extra_config = SearchPlan._get_optimized_values_config(output_dir)

    assert extra_config == {
        "search": {"target_ms1_tolerance": 15.0, "target_ms2_tolerance": 25.0}
    }
    mock_read_csv.assert_called_once_with(output_dir / "stat_output.tsv", sep="\t")

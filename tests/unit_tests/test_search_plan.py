from unittest.mock import MagicMock, call, patch

from alphadia.planning import Plan
from alphadia.search_plan import SearchPlan

MOCK_MULTISTEP_CONFIG = {
    "transfer": {"some_transfer_config_key": "some_transfer_config_value"},
    "library": {"some_library_config_key": "some_library_config_value"},
    "mbr": {"some_mbr_config_key": "some_mbr_config_value"},
}


def get_search_plan(config):
    """Helper function to create a SearchPlan object with a given config."""
    with patch(
        "alphadia.search_plan.yaml.safe_load", return_value=MOCK_MULTISTEP_CONFIG
    ):
        return SearchPlan(
            output_directory="/output",
            raw_path_list=["/raw1"],
            library_path="/library",
            fasta_path_list=["/fasta1"],
            config=config,
            quant_dir="/quant",
        )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
def test_runs_plan_without_transfer_and_mbr_steps(mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly without transfer and mbr steps."""
    search_plan = get_search_plan({"some_user_config_key": "some_user_config_value"})

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/output")

    mock_plan.assert_called_once_with(
        "/output",
        raw_path_list=["/raw1"],
        library_path="/library",
        fasta_path_list=["/fasta1"],
        config={"some_user_config_key": "some_user_config_value"},
        extra_config={},
        quant_path="/quant",
    )

    mock_plan.return_value.run.assert_called_once_with()


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
@patch("alphadia.search_plan.SearchPlan._get_dynamic_config_from_step")
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

    user_config = {
        "some_user_config_key": "some_user_config_value",
    }
    search_plan = get_search_plan(user_config | multistep_search_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/output")

    mock_plan.assert_has_calls(
        [
            call(
                "/output/transfer",
                raw_path_list=["/raw1"],
                library_path="/library",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config=MOCK_MULTISTEP_CONFIG["transfer"],
                quant_path="/quant",
            ),
            call().run(),
            call(
                "/output",
                raw_path_list=["/raw1"],
                library_path="/output/transfer/speclib.hdf",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config=MOCK_MULTISTEP_CONFIG["library"]
                | {
                    "library_prediction": {
                        "peptdeep_model_path": "/output/transfer/peptdeep.transfer"
                    },
                }
                | dynamic_config,
                quant_path="/output/transfer/quant",
            ),
            call().run(),
        ]
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
@patch("alphadia.search_plan.SearchPlan._get_dynamic_config_from_step")
def test_runs_plan_with_mbr_step(mock_get_dyn_config, mock_plan, mock_init_logging):
    """Test that the SearchPlan object runs the plan correctly with the mbr step enabled."""
    multistep_search_config = {
        "multistep_search": {
            "transfer_step_enabled": False,
            "mbr_step_enabled": True,
        }
    }

    user_config = {
        "some_user_config_key": "some_user_config_value",
    }
    search_plan = get_search_plan(user_config | multistep_search_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/output")

    mock_plan.assert_has_calls(
        [
            call(
                "/output/library",
                raw_path_list=["/raw1"],
                library_path="/library",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config={},  # TODO should this be MOCK_MULTISTEP_CONFIG["library"]?
                quant_path="/quant",
            ),
            call().run(),
            call(
                "/output",
                raw_path_list=["/raw1"],
                library_path="/output/library/speclib.hdf",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config=MOCK_MULTISTEP_CONFIG["mbr"] | dynamic_config,
                quant_path="/output/library/quant",
            ),
            call().run(),
        ],
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
@patch("alphadia.search_plan.SearchPlan._get_dynamic_config_from_step")
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

    user_config = {
        "some_user_config_key": "some_user_config_value",
    }
    search_plan = get_search_plan(user_config | multistep_search_config)

    dynamic_config = {"some_dynamic_config_key": "some_dynamic_config_value"}
    mock_get_dyn_config.return_value = dynamic_config

    # when
    search_plan.run_plan()

    mock_init_logging.assert_called_once_with("/output")

    mock_plan.assert_has_calls(
        [
            call(
                "/output/transfer",
                raw_path_list=["/raw1"],
                library_path="/library",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config=MOCK_MULTISTEP_CONFIG["transfer"],
                quant_path="/quant",
            ),
            call().run(),
            call(
                "/output/library",
                raw_path_list=["/raw1"],
                library_path="/output/transfer/speclib.hdf",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config=MOCK_MULTISTEP_CONFIG["library"]
                | {
                    "library_prediction": {
                        "peptdeep_model_path": "/output/transfer/peptdeep.transfer"
                    },
                }
                | dynamic_config,
                quant_path="/output/transfer/quant",
            ),
            call().run(),
            call(
                "/output",
                raw_path_list=["/raw1"],
                library_path="/output/library/speclib.hdf",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config=MOCK_MULTISTEP_CONFIG["mbr"] | dynamic_config,
                quant_path="/output/library/quant",
            ),
            call().run(),
        ],
    )


def test_get_dynamic_config_from_step():
    """Test that the SearchPlan object updates the config with the library step."""
    library_step = MagicMock(spec=Plan)
    library_step.estimators = {
        "optimization:ms1_error": 10,
        "optimization:ms2_error": 20,
    }

    # when
    extra_config = SearchPlan._get_dynamic_config_from_step(library_step)
    assert extra_config == {
        "search": {"target_ms1_tolerance": 10, "target_ms2_tolerance": 20}
    }

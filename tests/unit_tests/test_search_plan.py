from pathlib import Path
from unittest.mock import call, patch

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


def test_initializes_correctly():
    """Test that the SearchPlan object initializes correctly."""
    search_plan = get_search_plan({})
    assert search_plan._output_dir == Path("/output")
    assert search_plan._library_path == Path("/library")
    assert search_plan._fasta_path_list == ["/fasta1"]
    assert search_plan._quant_dir == Path("/quant")
    assert search_plan._raw_path_list == ["/raw1"]


def test_initialize_correctly_transfer_only():
    """Test that the SearchPlan object initializes correctly with only the transfer step enabled."""
    search_plan = get_search_plan(
        {"multistep_search": {"transfer_step_enabled": True, "mbr_step_enabled": False}}
    )
    assert search_plan._transfer_step_output_dir == Path("/output/transfer")
    assert search_plan._library_step_quant_dir == Path("/output/transfer/quant")
    assert search_plan._library_step_library_path == Path(
        "/output/transfer/speclib.hdf"
    )
    assert search_plan._library_step_output_dir == Path("/output")
    assert search_plan._mbr_step_quant_dir is None
    assert search_plan._mbr_step_library_path is None


def test_initialize_correctly_mbr_only():
    """Test that the SearchPlan object initializes correctly with only the mbr step enabled."""
    search_plan = get_search_plan(
        {"multistep_search": {"transfer_step_enabled": False, "mbr_step_enabled": True}}
    )
    assert search_plan._transfer_step_output_dir is None
    assert search_plan._library_step_quant_dir == Path("/quant")
    assert search_plan._library_step_library_path == Path("/library")
    assert search_plan._library_step_output_dir == Path("/output/library")
    assert search_plan._mbr_step_quant_dir == Path("/output/library/quant")
    assert search_plan._mbr_step_library_path == Path("/output/library/speclib.hdf")


def test_initialize_correctly_transfer_and_mbr():
    search_plan = get_search_plan(
        {"multistep_search": {"transfer_step_enabled": True, "mbr_step_enabled": True}}
    )
    assert search_plan._transfer_step_output_dir == Path("/output/transfer")
    assert search_plan._library_step_quant_dir == Path("/output/transfer/quant")
    assert search_plan._library_step_library_path == Path(
        "/output/transfer/speclib.hdf"
    )
    assert search_plan._library_step_output_dir == Path("/output/library")
    assert search_plan._mbr_step_quant_dir == Path("/output/library/quant")
    assert search_plan._mbr_step_library_path == Path("/output/library/speclib.hdf")


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
def test_runs_plan_with_transfer_step(mock_plan, mock_init_logging):
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
                },
                quant_path="/output/transfer/quant",
            ),
            call().run(),
        ]
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
def test_runs_plan_with_mbr_step(mock_plan, mock_init_logging):
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
                extra_config=MOCK_MULTISTEP_CONFIG["mbr"],
                quant_path="/output/library/quant",
            ),
            call().run(),
        ],
    )


@patch("alphadia.search_plan.reporting.init_logging")
@patch("alphadia.search_plan.Plan")
def test_runs_plan_with_transfer_and_mbr_steps(mock_plan, mock_init_logging):
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
                },
                quant_path="/output/transfer/quant",
            ),
            call().run(),
            call(
                "/output",
                raw_path_list=["/raw1"],
                library_path="/output/library/speclib.hdf",
                fasta_path_list=["/fasta1"],
                config=user_config | multistep_search_config,
                extra_config=MOCK_MULTISTEP_CONFIG["mbr"],
                quant_path="/output/library/quant",
            ),
            call().run(),
        ],
    )

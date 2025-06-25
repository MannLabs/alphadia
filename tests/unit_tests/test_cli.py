"""This module provides unit tests for alphadia.cli."""

from unittest.mock import MagicMock, mock_open, patch

import yaml

from alphadia.cli import _get_config_from_args, _get_from_args_or_config, run

# TODO add tests for _get_raw_path_list_from_args_and_config


def test_get_config_from_args_nothing_provided():
    """Test the _get_config_from_args function correctly returns if nothing is provided."""
    mock_args = MagicMock(config=None, config_dict={})

    result = _get_config_from_args(mock_args)

    assert result == ({}, None, {})


def test_get_config_from_args():
    """Test the _get_config_from_args function correctly parses config file."""
    mock_args = MagicMock(config="config.yaml", config_dict={})

    yaml_content = {"key1": "value1", "key2": "value2"}
    mock_yaml = yaml.dump(yaml_content)

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        result = _get_config_from_args(mock_args)

    assert result == ({"key1": "value1", "key2": "value2"}, "config.yaml", {})


def test_get_config_from_config_dict():
    """Test the _get_config_from_args function correctly parses config dict."""
    mock_args = MagicMock(config=None, config_dict='{"key3": "value3"}')

    result = _get_config_from_args(mock_args)

    assert result == ({"key3": "value3"}, None, '{"key3": "value3"}')


def test_get_config_from_args_and_config_dict():
    """Test the _get_config_from_args function correctly merges config file and dict."""
    mock_args = MagicMock(config="config.yaml", config_dict='{"key3": "value3"}')

    yaml_content = {"key1": "value1", "key2": "value2"}
    mock_yaml = yaml.dump(yaml_content)

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        result = _get_config_from_args(mock_args)

    assert result == (
        {"key1": "value1", "key2": "value2", "key3": "value3"},
        "config.yaml",
        '{"key3": "value3"}',
    )


def test_get_from_args_or_config_returns_value_from_args():
    """Test that the function returns the value from the args when it is not None."""
    args = MagicMock(output="cli_output")
    config = {"output_directory": "config_output"}

    # when
    result = _get_from_args_or_config(
        args, config, args_key="output", config_key="output_directory"
    )

    assert result == "cli_output"


def test_get_from_args_or_config_returns_value_from_config_when_args_none():
    """Test that the function returns the value from the config when the args value is None."""
    args = MagicMock(output=None)
    config = {"output_directory": "config_output"}

    # when
    result = _get_from_args_or_config(
        args, config, args_key="output", config_key="output_directory"
    )

    assert result == "config_output"


@patch("alphadia.cli.parser.parse_known_args")
@patch("builtins.print")
def test_cli_unknown_args(
    mock_print,
    mock_parse_known_args,
):
    mock_parse_known_args.return_value = (MagicMock, ["unknown_arg"])

    mock_search_plan = MagicMock()

    # when
    with patch.dict("sys.modules", SearchPlan=mock_search_plan):
        run()

    mock_print.assert_called_once_with("Unknown arguments: ['unknown_arg']")
    mock_search_plan.assert_not_called()


@patch("alphadia.cli.parser.parse_known_args")
def test_cli_minimal_args(mock_parse_known_args):
    """Test the run function of the CLI with minimal arguments maps correctly to SearchPlan."""
    mock_args = MagicMock(config=None, version=None, check=None, output="/output")
    mock_parse_known_args.return_value = (mock_args, [])

    mock_search_plan = MagicMock()

    mock_reporting = MagicMock()

    # when
    with patch.dict(
        "sys.modules",
        {
            "alphadia.search_plan": MagicMock(SearchPlan=mock_search_plan),
            "alphadia.reporting": MagicMock(reporting=mock_reporting),
        },
    ):
        run()

    mock_search_plan.assert_called_once_with(
        "/output",
        {},
        {
            # TODO raw_paths missing here
            "library_path": mock_args.library,
            "fasta_paths": mock_args.fasta,
            "quant_directory": mock_args.quant_dir,
        },
    )
    mock_search_plan.return_value.run_plan.assert_called_once()

    mock_reporting.init_logging.assert_called_once_with("/output")


@patch("alphadia.cli.parser.parse_known_args")
def test_cli_minimal_args_all_none(mock_parse_known_args):
    """Test the run function of the CLI with minimal arguments maps correctly to SearchPlan if nothing given."""
    mock_args = MagicMock(
        config=None,
        version=None,
        check=None,
        output="/output",
        fasta=None,
        library=None,
        quant_dir=None,
    )
    mock_parse_known_args.return_value = (mock_args, [])

    mock_search_plan = MagicMock()
    mock_reporting = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "alphadia.search_plan": MagicMock(SearchPlan=mock_search_plan),
            "alphadia.reporting": MagicMock(reporting=mock_reporting),
        },
    ):
        run()

    mock_search_plan.assert_called_once_with(
        "/output",
        {},
        {},
    )

    mock_search_plan.return_value.run_plan.assert_called_once()

    mock_reporting.init_logging.assert_called_once_with("/output")

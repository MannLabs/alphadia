"""This module provides unit tests for alphadia.cli."""

from unittest.mock import patch, MagicMock
from alphadia.cli import run


@patch("alphadia.cli.parser.parse_known_args")
@patch("alphadia.cli.reporting.init_logging")
@patch("alphadia.cli.SearchPlan")
def test_cli_minimal_args(mock_search_plan, mock_init_logging, mock_parse_known_args):
    """Test the run function of the CLI with minimal arguments."""
    mock_args = MagicMock(config=None, version=None, output="/output")
    mock_parse_known_args.return_value = (mock_args, [])

    # when
    run()

    mock_search_plan.assert_called_once_with("/output", {}, {
        'library_path': mock_args.library,
                                                        'fasta_paths': mock_args.fasta,
                                                        'quant_directory': mock_args.quant_dir}
                                             )

    mock_init_logging.assert_called_once_with("/output")

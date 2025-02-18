#!python
"""CLI for alphaDIA.

Ideally the CLI module should have as little logic as possible so that the search behaves the same from the CLI or a jupyter notebook.
"""

import argparse
import json
import logging
import os
import re
import sys

import matplotlib
import yaml

import alphadia
from alphadia import utils
from alphadia.constants.keys import ConfigKeys
from alphadia.exceptions import CustomError
from alphadia.search_plan import SearchPlan
from alphadia.workflow import reporting

logger = logging.getLogger()

epilog = "Parameters passed via CLI will overwrite parameters from config file (except for  '--file': will be merged)."

parser = argparse.ArgumentParser(
    description="Search DIA experiments with alphaDIA", epilog=epilog
)
parser.add_argument(
    "--version",
    "-v",
    action="store_true",
    help="Print version and exit",
)
parser.add_argument(
    "--output",
    "--output-directory",
    "-o",
    type=str,
    help="Output directory.",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--file",
    "--raw-path",
    "-f",
    type=str,
    help="Path to raw data input file. Can be passed multiple times.",
    action="append",
    default=[],
)
parser.add_argument(
    "--directory",
    "-d",
    type=str,
    help="Directory containing raw data input files.",
    action="append",
    default=[],
)
parser.add_argument(
    "--regex",
    "-r",
    type=str,
    help="Regex to match raw files in 'directory'.",
    nargs="?",
    default=".*",
)
parser.add_argument(
    "--library",
    "--library-path",
    "-l",
    type=str,
    help="Path to spectral library file.",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--fasta",
    "--fasta-path",
    help="Path to fasta file used to generate or annotate the spectral library. Can be passed multiple times.",
    action="append",
    default=[],
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    help="Path to config yaml file which will be used to update the default config.",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--config-dict",
    type=str,
    help="Python dictionary which will be used to update the default config.",
    nargs="?",
    default="{}",
)
parser.add_argument(
    "--quant-dir",  # TODO deprecate
    "--quant-directory",
    type=str,
    help="Directory to save the quantification results (psm & frag parquet files) to be reused in a distributed search.",
    nargs="?",
    default=None,
)


def _get_config_from_args(args: argparse.Namespace) -> dict:
    """Parse config file from `args.config` if given and update with optional JSON string `args.config_dict`."""

    config = {}
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    try:
        utils.recursive_update(config, json.loads(args.config_dict))
    except Exception as e:
        print(f"Could not parse config update: {e}")

    return config


def _get_from_args_or_config(
    args: argparse.Namespace, config: dict, *, args_key: str, config_key: str
) -> str:
    """Get a value from command line arguments (key: `args_key`) or config file (key: `config_key`), the former taking precedence."""
    value_from_args = args.__dict__.get(args_key)
    return value_from_args if value_from_args is not None else config.get(config_key)


def _get_raw_path_list_from_args_and_config(
    args: argparse.Namespace, config: dict
) -> list:
    """
    Generate a list of raw file paths based on command-line arguments and configuration.

    This function combines file paths specified in the configuration and command-line
    arguments, including files from specified directories. It filters the resulting
    list of file paths using a regular expression provided in the arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing file and directory
        paths, as well as a regex pattern for filtering.
    config : dict
        Configuration dictionary that may include a list of raw paths
        and a directory to search for files.

    Returns
    -------
        list: a list of file paths that match the specified regex pattern.
    """

    raw_path_list = config.get(ConfigKeys.RAW_PATHS, [])
    raw_path_list += args.file

    if (config_directory := config.get("directory")) is not None:
        raw_path_list += [
            os.path.join(config_directory, f) for f in os.listdir(config_directory)
        ]

    for directory in args.directory:
        raw_path_list += [os.path.join(directory, f) for f in os.listdir(directory)]

    # filter raw files by regex
    len_before = len(raw_path_list)
    raw_path_list = [
        f
        for f in raw_path_list
        if re.search(args.regex, os.path.basename(f)) is not None
    ]
    len_after = len(raw_path_list)

    if len_removed := len_before - len_after:
        print(
            f"Ignoring {len_removed} / {len_before} file(s) from arguments list due to --regex."
        )

    return raw_path_list


def run(*args, **kwargs):
    # parse command line arguments
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Unknown arguments: {unknown}")
        parser.print_help()
        return

    if args.version:
        print(f"{alphadia.__version__}")
        return

    user_config = _get_config_from_args(args)

    output_directory = _get_from_args_or_config(
        args, user_config, args_key="output", config_key="output_directory"
    )
    if output_directory is None:
        parser.print_help()

        print("No output directory specified. Please do so via CL-argument or config.")
        return
    reporting.init_logging(output_directory)

    # TODO revisit the multiple sources of raw files (cli, config, regex, ...)
    raw_paths = _get_raw_path_list_from_args_and_config(args, user_config)
    cli_params_config = {
        **({ConfigKeys.RAW_PATHS: raw_paths} if raw_paths else {}),
        **({ConfigKeys.LIBRARY_PATH: args.library} if args.library is not None else {}),
        **({ConfigKeys.FASTA_PATHS: args.fasta} if args.fasta else {}),
        **(
            {ConfigKeys.QUANT_DIRECTORY: args.quant_dir}
            if args.quant_dir is not None
            else {}
        ),
    }

    # TODO rename all output_directory, output_folder => output_path, quant_dir->quant_path (except cli parameter)

    # important to suppress matplotlib output
    matplotlib.use("Agg")

    try:
        SearchPlan(output_directory, user_config, cli_params_config).run_plan()

    except Exception as e:
        if isinstance(e, CustomError):
            exit_code = 1
        else:
            import traceback

            logger.info(traceback.format_exc())
            exit_code = 127

        logger.error(e)
        sys.exit(exit_code)


# uncomment for debugging:
# if __name__ == "__main__":
#     run()

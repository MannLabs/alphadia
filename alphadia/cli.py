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
from alphadia.exceptions import CustomError
from alphadia.search_plan import SearchPlan
from alphadia.workflow import reporting

logger = logging.getLogger()


parser = argparse.ArgumentParser(description="Search DIA experiments with alphaDIA")
parser.add_argument(
    "--version",
    "-v",
    action="store_true",
    help="Print version and exit",
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    help="Output directory",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--file",
    "-f",
    type=str,
    help="Raw data input files.",
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
    help="Regex to match raw files in directory.",
    nargs="?",
    default=".*",
)
parser.add_argument(
    "--library",
    "-l",
    type=str,
    help="Spectral library.",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--fasta",
    help="Fasta file(s) used to generate or annotate the spectral library.",
    action="append",
    default=[],
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    help="Config yaml which will be used to update the default config.",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--config-dict",
    type=str,
    help="Python Dict which will be used to update the default config.",
    nargs="?",
    default="{}",
)
parser.add_argument(
    "--quant-dir",
    type=str,
    help="Directory to save the quantification results (psm & frag parquet files) to be reused in a distributed search",
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

    Args:
        args (argparse.Namespace): Command-line arguments containing file and directory
            paths, as well as a regex pattern for filtering.
        config (dict): Configuration dictionary that may include a list of raw paths
            and a directory to search for files.

    Returns:
        list: A list of file paths that match the specified regex pattern.
    """

    raw_path_list = config.get("raw_path_list", [])
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
    print(f"Removed {len_before - len_after} of {len_before} files.")

    return raw_path_list


def _get_fasta_list_from_args_and_config(
    args: argparse.Namespace, config: dict
) -> list:
    """Parse fasta file list from command line arguments and config file, merging them if both are given."""

    fasta_path_list = config.get("fasta_list", [])
    fasta_path_list += args.fasta

    return fasta_path_list


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
        print("No output directory specified.")
        return
    reporting.init_logging(output_directory)

    quant_dir = _get_from_args_or_config(
        args, user_config, args_key="quant_dir", config_key="quant_dir"
    )
    raw_path_list = _get_raw_path_list_from_args_and_config(args, user_config)
    library_path = _get_from_args_or_config(
        args, user_config, args_key="library", config_key="library"
    )
    fasta_path_list = _get_fasta_list_from_args_and_config(args, user_config)

    # TODO rename all output_directory, output_folder => output_path, quant_dir->quant_path (except cli parameter)

    # important to suppress matplotlib output
    matplotlib.use("Agg")

    try:
        SearchPlan(
            output_directory,
            raw_path_list=raw_path_list,
            library_path=library_path,
            fasta_path_list=fasta_path_list,
            config=user_config,
            quant_path=quant_dir,
        ).run_plan()

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

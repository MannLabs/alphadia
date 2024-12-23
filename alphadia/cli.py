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
    "--wsl",
    "-w",
    action="store_true",
    help="Set if running on Windows Subsystem for Linux.",
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


def parse_config(args: argparse.Namespace) -> dict:
    """Parse config file and config update JSON string.
    1. Load config file if specified.
    2. Update config with config update JSON string.

    Parameters
    ----------

    args : argparse.Namespace
        Command line arguments.

    Returns
    -------

    config : dict
        Updated config dictionary.
    """

    config = {}
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    try:
        utils.recursive_update(config, json.loads(args.config_dict))
    except Exception as e:
        print(f"Could not parse config update: {e}")

    return config


def parse_output_directory(args: argparse.Namespace, config: dict) -> str:
    """Parse output directory.
    1. Use output directory from config file if specified.
    2. Use output directory from command line if specified.

    Parameters
    ----------

    args : argparse.Namespace
        Command line arguments.

    config : dict
        Config dictionary.

    Returns
    -------

    output_directory : str
        Output directory.
    """

    output_directory = None
    if config.get(ConfigKeys.OUTPUT_DIRECTORY) is not None:
        output_directory = (
            utils.windows_to_wsl(config[ConfigKeys.OUTPUT_DIRECTORY])
            if args.wsl
            else config[ConfigKeys.OUTPUT_DIRECTORY]
        )

    if args.output is not None:
        output_directory = (
            utils.windows_to_wsl(args.output) if args.wsl else args.output
        )

    return output_directory


def parse_raw_path_list(args: argparse.Namespace, config: dict) -> list:
    """Parse raw file list.
    1. Use raw file list from config file if specified.
    2. Use raw file list from command line if specified.

    Parameters
    ----------

    args : argparse.Namespace
        Command line arguments.

    config : dict
        Config dictionary.

    Returns
    -------

    raw_path_list : list
        List of raw files.
    """
    config_raw_path_list = config.get("raw_paths", [])
    raw_path_list = (
        utils.windows_to_wsl(config_raw_path_list) if args.wsl else config_raw_path_list
    )
    raw_path_list += utils.windows_to_wsl(args.file) if args.wsl else args.file

    config_directory = config.get("directory")
    directory = utils.windows_to_wsl(config_directory) if args.wsl else config_directory
    if directory is not None:
        raw_path_list += [os.path.join(directory, f) for f in os.listdir(directory)]

    directory_list = (
        utils.windows_to_wsl(args.directory) if args.wsl else args.directory
    )
    for directory in directory_list:
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

    config = parse_config(args)

    output_directory = parse_output_directory(args, config)
    if output_directory is None:
        # print help message if no output directory specified
        parser.print_help()

        print("No output directory specified. Please do so via CL-argument or config.")
        return
    reporting.init_logging(output_directory)

    # TODO revisit the multiple sources of raw files (cli, config, regex, ...)
    raw_path_list = parse_raw_path_list(args, config)
    cli_params_config = {
        **({ConfigKeys.RAW_PATHS: raw_path_list} if raw_path_list else {}),
        **({ConfigKeys.LIBRARY_PATH: args.library} if args.library is not None else {}),
        **({ConfigKeys.FASTA_PATHS: args.library} if args.fasta else {}),
        **(
            {ConfigKeys.QUANT_DIRECTORY: args.library}
            if args.quant_dir is not None
            else {}
        ),
    }

    # TODO rename all output_directory, output_folder => output_path, quant_dir->quant_path (except cli parameter)

    # important to suppress matplotlib output
    matplotlib.use("Agg")

    try:
        SearchPlan(output_directory, config, cli_params_config).run_plan()

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
if __name__ == "__main__":
    run()

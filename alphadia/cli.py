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

import yaml

import alphadia
from alphadia import utils
from alphadia.exceptions import CustomError
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
    "--wsl",
    "-w",
    action="store_true",
    help="Set if running on Windows Subsystem for Linux.",
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
    if "output_directory" in config:
        output_directory = (
            utils.windows_to_wsl(config["output_directory"])
            if args.wsl
            else config["output_directory"]
        )

    if args.output is not None:
        output_directory = (
            utils.windows_to_wsl(args.output) if args.wsl else args.output
        )

    return output_directory


def parse_quant_dir(args: argparse.Namespace, config: dict) -> str:
    """Parse custom quant path.
    1. Use custom quant path from config file if specified.
    2. Use custom quant path from command line if specified.

    Parameters
    ----------

    args : argparse.Namespace
        Command line arguments.

    config : dict
        Config dictionary.

    Returns
    -------

    quant_dir : str
        path to quant directory.
    """

    quant_dir = None
    if "quant_dir" in config:
        quant_dir = (
            utils.windows_to_wsl(config["quant_dir"])
            if args.wsl
            else config["quant_dir"]
        )

    if args.quant_dir is not None:
        quant_dir = utils.windows_to_wsl(args.quant_dir) if args.wsl else args.quant_dir

    return quant_dir


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
    config_raw_path_list = config.get("raw_path_list", [])
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


def parse_library(args: argparse.Namespace, config: dict) -> str:
    """Parse spectral library.
    1. Use spectral library from config file if specified.
    2. Use spectral library from command line if specified.

    Parameters
    ----------

    args : argparse.Namespace
        Command line arguments.

    config : dict
        Config dictionary.

    Returns
    -------

    library : str
        Spectral library.
    """

    library = None
    if "library" in config:
        library = (
            utils.windows_to_wsl(config["library"]) if args.wsl else config["library"]
        )

    if args.library is not None:
        library = utils.windows_to_wsl(args.library) if args.wsl else args.library

    return library


def parse_fasta(args: argparse.Namespace, config: dict) -> list:
    """Parse fasta file list.
    1. Use fasta file list from config file if specified.
    2. Use fasta file list from command line if specified.

    Parameters
    ----------

    args : argparse.Namespace
        Command line arguments.

    config : dict
        Config dictionary.

    Returns
    -------

    fasta_path_list : list
        List of fasta files.
    """

    config_fasta_path_list = config.get("fasta_list", [])
    fasta_path_list = (
        utils.windows_to_wsl(config_fasta_path_list)
        if args.wsl
        else config_fasta_path_list
    )
    fasta_path_list += utils.windows_to_wsl(args.fasta) if args.wsl else args.fasta

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

    config = parse_config(args)

    output_directory = parse_output_directory(args, config)
    if output_directory is None:
        # print help message if no output directory specified
        parser.print_help()

        print("No output directory specified.")
        return

    quant_dir = parse_quant_dir(args, config)

    reporting.init_logging(output_directory)
    raw_path_list = parse_raw_path_list(args, config)

    library_path = parse_library(args, config)
    fasta_path_list = parse_fasta(args, config)

    logger.progress(f"Searching {len(raw_path_list)} files:")
    for f in raw_path_list:
        logger.progress(f"  {os.path.basename(f)}")

    logger.progress(f"Using library: {library_path}")

    logger.progress(f"Using {len(fasta_path_list)} fasta files:")
    for f in fasta_path_list:
        logger.progress(f"  {f}")

    # TODO rename all output_directory, output_folder => output_path, quant_dir->quant_path (except cli parameter)
    logger.progress(f"Saving output to: {output_directory}")
    if quant_dir is not None:
        logger.progress(f"Saving quantification output to {quant_dir=}")

    try:
        import matplotlib

        # important to suppress matplotlib output
        matplotlib.use("Agg")

        from alphadia.planning import Plan

        plan = Plan(
            output_directory,
            raw_path_list=raw_path_list,
            library_path=library_path,
            fasta_path_list=fasta_path_list,
            config=config,
            quant_path=quant_dir,
        )

        plan.run()

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

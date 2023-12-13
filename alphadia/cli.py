#!python

# native imports
import logging
import time
import yaml
import os

# alphadia imports
import alphadia
from alphadia.workflow import reporting
from alphadia import utils

from alphabase.constants import modification

modification.add_new_modifications(
    {
        "Dimethyl:d12@Protein N-term": {"composition": "H(-2)2H(8)13C(2)"},
        "Dimethyl:d12@Any N-term": {
            "composition": "H(-2)2H(8)13C(2)",
        },
        "Dimethyl:d12@R": {
            "composition": "H(-2)2H(8)13C(2)",
        },
        "Dimethyl:d12@K": {
            "composition": "H(-2)2H(8)13C(2)",
        },
    }
)

# alpha family imports

# third party imports
import click


@click.group(
    context_settings=dict(
        help_option_names=["-h", "--help"],
    ),
    invoke_without_command=True,
)
@click.pass_context
@click.version_option(alphadia.__version__, "-v", "--version", message="%(version)s")
def run(ctx, **kwargs):
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("gui", help="Start graphical user interface.")
def gui():
    import alphadia.gui

    alphadia.gui.run()


@run.command(
    "extract",
    help="Extract DIA precursors from a list of raw files using a spectral library.",
)
@click.argument(
    "output-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
)
@click.option(
    "--file",
    "-f",
    help="Raw data input files.",
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
)
@click.option(
    "--directory",
    "-d",
    help="Directory containing raw data input files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--regex",
    "-r",
    help="Regex to match raw files in directory.",
    type=str,
    default=".*",
    show_default=True,
)
@click.option(
    "--library",
    "-l",
    help="Spectral library in AlphaBase hdf5 format.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--wsl",
    "-w",
    help="Run alphadia using WSL. Windows paths will be converted to WSL paths.",
    type=bool,
    default=False,
    is_flag=True,
)
@click.option(
    "--fdr",
    help="False discovery rate for the final output.",
    type=float,
    default=0.01,
    show_default=True,
)
@click.option(
    "--keep-decoys",
    help="Keep decoys in the final output.",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--config",
    help="Config yaml which will be used to update the default config.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--config-base",
    help="DO NOT TOUCH - Default config yaml. If not specified, the default config will be used.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--config-update",
    help="Dict which will be used to update the default config.",
    type=str,
    default={},
)
@click.option(
    "--neptune-token",
    help="Neptune.ai token for continous logging.",
    type=str,
    default=None,
    show_default=False,
)
@click.option(
    "--neptune-tag",
    help="Neptune.ai tag for continous logging.",
    type=str,
    multiple=True,
)
@click.option(
    "--figure-path",
    help="If specified, directory will be used to store calibration figures.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def extract(**kwargs):
    kwargs["neptune_tag"] = list(kwargs["neptune_tag"])

    # load config file if specified
    config_update = {}
    if kwargs["config"] is not None:
        with open(kwargs["config"], "r") as f:
            config_update = yaml.safe_load(f)

    # update output directory based on config file
    output_directory = None
    if kwargs["output_directory"] is not None:
        if kwargs["wsl"]:
            kwargs["output_directory"] = utils.windows_to_wsl(
                kwargs["output_directory"]
            )
        output_directory = kwargs["output_directory"]

    if "output_directory" in config_update:
        if kwargs["wsl"]:
            config_update["output_directory"] = utils.windows_to_wsl(
                config_update["output_directory"]
            )
        output_directory = config_update["output_directory"]

    if output_directory is None:
        logging.error("No output directory specified.")
        return

    reporting.init_logging(output_directory)
    logger = logging.getLogger()

    # assert input files have been specified
    files = []
    if kwargs["file"] is not None:
        files = list(kwargs["file"])
        if kwargs["wsl"]:
            files = [utils.windows_to_wsl(f) for f in files]

    # load whole directory if specified
    if kwargs["directory"] is not None:
        if kwargs["wsl"]:
            kwargs["directory"] = utils.windows_to_wsl(kwargs["directory"])
        files += [
            os.path.join(kwargs["directory"], f)
            for f in os.listdir(kwargs["directory"])
        ]

    # load list of raw files from config file
    if "raw_file_list" in config_update:
        if kwargs["wsl"]:
            config_update["raw_file_list"] = [
                utils.windows_to_wsl(f) for f in config_update["raw_file_list"]
            ]
        files += (
            config_update["raw_file_list"]
            if type(config_update["raw_file_list"]) is list
            else [config_update["raw_file_list"]]
        )
    
    # filter files based on regex
    logger.info(f"Filtering files based on regex: {kwargs['regex']}")
    len_before = len(files)
    files = [f for f in files if utils.match_regex(f, kwargs["regex"])]
    len_after = len(files)
    logger.info(f"Removed {len_before - len_after} of {len_before} files.")

    if (files is None) or (len(files) == 0):
        logging.error("No raw files specified.")
        return

    # assert library has been specified
    library = None
    if kwargs["library"] is not None:
        if kwargs["wsl"]:
            kwargs["library"] = utils.windows_to_wsl(kwargs["library"])
        library = kwargs["library"]

    if "library" in config_update:
        if kwargs["wsl"]:
            config_update["library"] = utils.windows_to_wsl(config_update["library"])
        library = config_update["library"]

    if library is None:
        logging.error("No library specified.")
        return

    logger.progress(f"Extracting from {len(files)} files:")
    for f in files:
        logger.progress(f"  {f}")
    logger.progress(f"Using library {library}.")

    if kwargs["wsl"]:
        config_update["general"]["wsl"] = True

    logger.progress(f"Saving output to {output_directory}.")

    try:
        import matplotlib

        # important to supress matplotlib output
        matplotlib.use("Agg")

        from alphadia.planning import Plan

        plan = Plan(output_directory, files, library, config_update=config_update)

        plan.run(
            keep_decoys=kwargs["keep_decoys"],
            fdr=kwargs["fdr"],
            figure_path=kwargs["figure_path"],
        )

    except Exception as e:
        logger.error(e)

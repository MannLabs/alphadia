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

# alpha family imports

# third party imports
import click

@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
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
    help="Extract DIA precursors from a list of raw files using a spectral library."
)
@click.argument(
    "output-location",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
)
@click.option(
    '--file',
    '-f',
    help="Raw data input files.",
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
)
@click.option(
    '--directory',
    '-d',
    help="Directory containing raw data input files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    '--library',
    '-l',
    help="Spectral library in AlphaBase hdf5 format.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    '--wsl',
    '-w',
    help="Run alphadia using WSL. Windows paths will be converted to WSL paths.",
    type=bool,
    default=False,
    is_flag=True,
)
@click.option(
    "--fdr",
    help='False discovery rate for the final output.',
    type=float,
    default=0.01,
    show_default=True,
)
@click.option(
    "--keep-decoys",
    help='Keep decoys in the final output.',
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--config",
    help='Config yaml which will be used to update the default config.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--config-base",
    help='DO NOT TOUCH - Default config yaml. If not specified, the default config will be used.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--config-update",
    help='Dict which will be used to update the default config.',
    type=str,
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
    multiple=True
)
@click.option(
    "--figure-path",
    help="If specified, directory will be used to store calibration figures.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def extract(**kwargs):

    kwargs['neptune_tag'] = list(kwargs['neptune_tag'])

    # load config file if specified
    config_update = None
    if kwargs['config'] is not None:
        with open(kwargs['config'], 'r') as f:
            config_update = yaml.safe_load(f)

    output_location = None
    if kwargs['output_location'] is not None:
        output_location = kwargs['output_location']

    if "output" in config_update:
        output_location = config_update['output']

    if output_location is None:
        logging.error("No output location specified.")
        return

    reporting.init_logging(output_location)
    logger = logging.getLogger()
    
    # assert input files have been specified
    files = None
    if kwargs['file'] is not None:
        files = list(kwargs['file'])
        if kwargs['wsl']:
            files = [utils.windows_to_wsl(f) for f in files]

    if kwargs['directory'] is not None:
        if kwargs['wsl']:
            kwargs['directory'] = utils.windows_to_wsl(kwargs['directory'])
        files += [os.path.join(kwargs['directory'], f) for f in os.listdir(kwargs['directory'])]
    
    if "files" in config_update:
        if kwargs['wsl']:
            config_update['files'] = [utils.windows_to_wsl(f) for f in config_update['files']]
        files += config_update['files'] if type(config_update['files']) is list else [config_update['files']]

    if (files is None) or (len(files) == 0):
        logging.error("No files specified.")
        return
    
    # assert library has been specified
    library = None
    if kwargs['library'] is not None:
        if kwargs['wsl']:
            kwargs['library'] = utils.windows_to_wsl(kwargs['library'])
        library = kwargs['library']

    if "library" in config_update:
        if kwargs['wsl']:
            config_update['library'] = utils.windows_to_wsl(config_update['library'])
        library = config_update['library']

    if library is None:
        logging.error("No library specified.")
        return
 
    logger.progress(f"Extracting from {len(files)} files:")
    for f in files:
        logger.progress(f"  {f}")
    logger.progress(f"Using library {library}.")
    logger.progress(f"Saving output to {output_location}.")
    
    try:

        import matplotlib
        # important to supress matplotlib output
        matplotlib.use('Agg')

        from alphadia.planning import Plan
        #lib._precursor_df['elution_group_idx'] = lib._precursor_df['precursor_idx']

        #config_update = eval(kwargs['config_update']) if kwargs['config_update'] else None

        plan = Plan(
            output_location,
            files,
            library,
            config_update = config_update
            )

        plan.run(
            keep_decoys = kwargs['keep_decoys'], 
            fdr = kwargs['fdr'], 
            figure_path = kwargs['figure_path'],
        )

    except Exception as e:
        logging.exception(e)
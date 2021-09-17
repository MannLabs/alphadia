#!python


# builtin
import logging
import time

# external
import click

# local
import alphadia


@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
)
@click.pass_context
@click.version_option(alphadia.__version__, "-v", "--version")
def run(ctx, **kwargs):
    name = f"AlphaDIA {alphadia.__version__}"
    click.echo("*" * (len(name) + 4))
    click.echo(f"* {name} *")
    click.echo("*" * (len(name) + 4))
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("gui", help="Start graphical user interface.")
def gui():
    import alphadia.gui
    alphadia.gui.run()


@run.command(
    "annotate",
    help="Annotate a DIA file with an AlphaPept result library."
)
@click.argument(
    "dia_file_name",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
)
@click.argument(
    "alphapept_library_file_name",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.argument(
    "output_file_name",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--ppm_tolerance",
    help="The ppm tolerance",
    type=float,
    default=20,
    show_default=True,
)
@click.option(
    "--rt_tolerance",
    help="The rt tolerance in seconds",
    type=float,
    default=30,
    show_default=True,
)
@click.option(
    "--mobility_tolerance",
    help="The mobility tolerance in 1/k0 (ignored for Thermo)",
    type=float,
    default=0.05,
    show_default=True,
)
@click.option(
    "--fdr_rate",
    help="The FDR",
    type=float,
    default=0.01,
    show_default=True,
)
@click.option(
    "--thread_count",
    help="The number of threads to use",
    type=int,
    default=-1,
    show_default=True,
)
@click.option(
    "--max_scan_difference",
    help="KEEP DEFAULT",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--max_cycle_difference",
    help="KEEP DEFAULT",
    type=int,
    default=1,
    show_default=True,
)
def annotate(**kwargs):
    logging.basicConfig(
        format='%(asctime)s> %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    try:
        start_time = time.time()
        kwargs = {key: arg for key, arg in kwargs.items() if arg is not None}
        logging.info("Creating new AlphaTemplate with parameters:")
        max_len = max(len(key) + 1 for key in kwargs)
        for key, value in sorted(kwargs.items()):
            logging.info(f"{key:<{max_len}} - {value}")
        logging.info("")
        import alphadia.dia
        alphadia.dia.run_analysis(
            dia_file_name=kwargs["dia_file_name"],
            alphapept_library_file_name=kwargs["alphapept_library_file_name"],
            output_file_name=kwargs["output_file_name"],
            ppm=kwargs["ppm_tolerance"],
            rt_tolerance=kwargs["rt_tolerance"],
            mobility_tolerance=kwargs["mobility_tolerance"],
            max_scan_difference=kwargs["max_scan_difference"],
            max_cycle_difference=kwargs["max_cycle_difference"],
            thread_count=kwargs["thread_count"],
            fdr_rate=kwargs["fdr_rate"],
        )
    except Exception:
        logging.exception("Something went wrong, execution incomplete!")
    else:
        logging.info(
            f"Analysis done in {time.time() - start_time:.2f} seconds."
        )

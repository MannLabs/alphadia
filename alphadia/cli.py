#!python


# builtin
import logging
import time
import sys
import yaml

# external
import click

# local
import alphadia
from alphadia.extraction.workflow import reporting

@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
)

@click.pass_context
@click.version_option(alphadia.__version__, "-v", "--version")
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
    '--library',
    '-l',
    help="Spectral library in AlphaBase hdf5 format.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
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

    reporting.init_logging(kwargs['output_location'])
    logger = logging.getLogger()

    # assert input files have been specified
    files = None
    if kwargs['file'] is not None:
        files = list(kwargs['file'])
    
    if "files" in config_update:
        files = config_update['files'] if type(config_update['files']) is list else [config_update['files']]

    if (files is None) or (len(files) == 0):
        logging.error("No files specified.")
        return
    
    # assert library has been specified
    library = None
    if kwargs['library'] is not None:
        library = kwargs['library']

    if "library" in config_update:
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

        from alphabase.spectral_library.base import SpecLibBase
        from alphadia.extraction.planning import Plan

        lib = SpecLibBase()
        lib.load_hdf(library, load_mod_seq=True)
        #lib._precursor_df['elution_group_idx'] = lib._precursor_df['precursor_idx']

        #config_update = eval(kwargs['config_update']) if kwargs['config_update'] else None

        plan = Plan(
            output_location,
            files,
            lib,
            config_update = config_update
            )

        plan.run(
            keep_decoys = kwargs['keep_decoys'], 
            fdr = kwargs['fdr'], 
            figure_path = kwargs['figure_path'],
        )

    except Exception as e:
        logging.exception(e)

@run.group(
    "spectrum",
    help="Process DIA data spectrum-centric."
)
def spectrum(*args, **kwargs):
    pass

@spectrum.command(
    "create",
    help="Create pseudo MSMS spectra from a DIA file."
)
@click.argument(
    "file_names",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    nargs=-1,
)
@click.option(
    "--folder",
    help="If set, the input arguments are considered folders and all .d folders in them will be processed",
    is_flag=True,
    default=False,
)
@click.option(
    "--thread_count",
    help="The number of threads to use. 0 for all, negative to keep available.",
    type=int,
    default=-1,
    show_default=True,
)
def create_spectrum(file_names, folder, thread_count, **kwargs):
    logging.basicConfig(
        format='%(asctime)s> %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    start_time = time.time()
    try:
        import os
        import alphatims.bruker
        import alphadia.preprocessing
        import alphatims.utils
        alphatims.utils.set_threads(thread_count)
        # alphatims.utils.set_logger()
        if folder:
            dia_file_names = []
            for directory in file_names:
                for file_name in os.listdir(directory):
                    full_file_name = os.path.join(directory, file_name)
                    dia_file_names.append(full_file_name)
        else:
            dia_file_names = file_names
        for dia_file_name in dia_file_names:
            if not dia_file_name.endswith(".d") or os.path.isfile(dia_file_name):
                continue
            dia_data = alphatims.bruker.TimsTOF(dia_file_name)
            preprocessing_workflow = alphadia.preprocessing.Workflow()
            preprocessing_workflow.set_dia_data(dia_data)
            preprocessing_workflow.run_default()
            # preprocessing_workflow.save_to_hdf()
            # preprocessing_workflow.load_from_hdf()
            preprocessing_workflow.msms_generator.write_to_hdf_file()
    except Exception:
        logging.exception("Something went wrong, execution incomplete!")
    else:
        logging.info(
            f"Analysis done in {time.time() - start_time:.2f} seconds."
        )



@spectrum.command(
    "annotate",
    help="Annotate pseudo MSMS spectra from a DIA file."
)
@click.argument(
    "library_file_name",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.argument(
    "file_names",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    nargs=-1,
)
@click.option(
    "--folder",
    help="If set, the input arguments are considered folders and all .d folders in them will be processed",
    is_flag=True,
    default=False,
)
@click.option(
    "--thread_count",
    help="The number of threads to use. 0 for all, negative to keep available.",
    type=int,
    default=-1,
    show_default=True,
)
def annotate_spectrum(library_file_name, file_names, folder, thread_count, **kwargs):
    logging.basicConfig(
        format='%(asctime)s> %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    start_time = time.time()
    try:
        import os
        import alphatims.utils
        import alphadia.annotation
        import alphabase.io.hdf
        import numpy as np
        import pandas as pd
        alphatims.utils.set_threads(thread_count)
        # alphatims.utils.set_logger()
        if folder:
            dia_file_names = []
            for directory in file_names:
                for file_name in os.listdir(directory):
                    full_file_name = os.path.join(directory, file_name)
                    dia_file_names.append(full_file_name)
        else:
            dia_file_names = file_names
        library = None
        for dia_file_name in dia_file_names:
            print(dia_file_name)
            if not (dia_file_name.endswith("pseudo_spectra.hdf") and os.path.isfile(dia_file_name)):
                continue
            if library is None:
                library = alphadia.annotation.library.Library()
                library.import_from_file(
                    # "/mnt/a54a8df1-78df-4788-bd29-6fca4115f5c0/software_development_data/fastas/phospho_jon/predict.speclib.hdf",
                    # "/mnt/a54a8df1-78df-4788-bd29-6fca4115f5c0/software_development_data/fastas/diann_entrapment/predict.speclib.hdf",
                    library_file_name,
                    is_already_mmapped=False,
                )
                all_seqs = library.lib.library.mod_seq_df.sequence.values
                all_mods = library.lib.library.mod_seq_df.mods.values
                all_mod_sites = library.lib.library.mod_seq_df.mod_sites.values
                all_msch = library.lib.library.mod_seq_df.mod_seq_charge_hash.values
                all_prot_idxs = library.lib.library.mod_seq_df.protein_idxes.values
                all_prots = library.lib.library.protein_df.values
            hdf = alphabase.io.hdf.HDF_File(
                # f"{self.dia_data.sample_name}.pseudo_spectra.hdf",
                dia_file_name,
            #     read_only=False,
            )
            fragment_df = hdf.fragments.values
            precursor_df = hdf.precursors.values
            annotator = alphadia.annotation.Annotator()
            # annotator.set_preprocessor(preprocessing_workflow)
            annotator.set_ions(precursor_df, fragment_df)
            annotator.set_library(library)
            annotator.run_default()
            mod_seq_charge_hash_pointer = annotator.percolator.score_df.original_index.values
            annotator.percolator.score_df["sequence"] = all_seqs[mod_seq_charge_hash_pointer].astype('U')
            annotator.percolator.score_df["mods"] = all_mods[mod_seq_charge_hash_pointer].astype('U')
            annotator.percolator.score_df["mod_sites"] = all_mod_sites[mod_seq_charge_hash_pointer].astype('U')
            annotator.percolator.score_df["protein_idxs"] = all_prot_idxs[mod_seq_charge_hash_pointer].astype('U')
            vals = annotator.percolator.score_df.protein_idxs.str.split(";", expand=True)[0]
            # Might be more
            annotated_prots = all_prots.iloc[vals.values.astype(np.int64)].reset_index(drop=True)
            annotator.percolator.score_df = pd.concat([annotator.percolator.score_df, annotated_prots], axis=1)
            annotator.percolator.score_df["unique_protein"] = ~annotator.percolator.score_df.protein_idxs.str.contains(";").values
            annotations = annotator.percolator.score_df
            annotations = annotator.percolator.score_df[
                annotator.percolator.score_df.target_type==1
            #     :
            ]

            annotations.to_csv(
            #     f"{dia_data.bruker_d_folder_name[:-2]}_annotation.csv",
                f"{hdf.file_name[:-4]}_annotation.csv",
                index=False,
            )
    except Exception:
        logging.exception("Something went wrong, execution incomplete!")
    else:
        logging.info(
            f"Analysis done in {time.time() - start_time:.2f} seconds."
        )

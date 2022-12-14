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
    "extract",
    help="Extract DIA precursors from an AlphaPept result library."
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
def extract(**kwargs):
    logging.basicConfig(
        format='%(asctime)s> %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    start_time = time.time()
    try:
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

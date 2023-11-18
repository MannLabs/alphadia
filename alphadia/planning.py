# native imports
import logging

logger = logging.getLogger()
import socket
from pathlib import Path
import yaml
import os
from datetime import datetime
import typing

# alphadia imports
from alphadia import utils, libtransform, outputtransform
from alphadia.workflow import peptidecentric, base, reporting
import alphadia

# alpha family imports
from alphabase.spectral_library.flat import SpecLibFlat

# third party imports
import numpy as np
import pandas as pd
import os, psutil
import torch


class Plan:
    def __init__(
        self,
        output_folder: str,
        raw_file_list: typing.List,
        spec_lib_path: typing.Union[str, None] = None,
        config_path: typing.Union[str, None] = None,
        config_update_path: typing.Union[str, None] = None,
        config_update: typing.Union[typing.Dict, None] = None,
    ) -> None:
        """Highest level class to plan a DIA Search.
        Owns the input file list, speclib and the config.
        Performs required manipulation of the spectral library like transforming RT scales and adding columns.

        Parameters
        ----------
        raw_data : list
            list of input file locations

        config_path : str, optional
            yaml file containing the default config.

        config_update_path : str, optional
           yaml file to update the default config.

        config_update : dict, optional
            dict to update the default config. Can be used for debugging purposes etc.

        """
        self.output_folder = output_folder
        reporting.init_logging(self.output_folder)

        logger.progress("      _   _      _         ___ ___   _   ")
        logger.progress("     /_\ | |_ __| |_  __ _|   \_ _| /_\  ")
        logger.progress("    / _ \| | '_ \\ ' \/ _` | |) | | / _ \ ")
        logger.progress("   /_/ \_\_| .__/_||_\__,_|___/___/_/ \_\\")
        logger.progress("           |_|                            ")
        logger.progress("")

        self.raw_file_list = raw_file_list

        # default config path is not defined in the function definition to account for for different path separators on different OS
        if config_path is None:
            # default yaml config location under /misc/config/config.yaml
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "misc", "config", "default.yaml"
            )

        # 1. load default config
        with open(config_path, "r") as f:
            logger.info(f"loading default config from {config_path}")
            self.config = yaml.safe_load(f)

        # 2. load update config from yaml file
        if config_update_path is not None:
            logger.info(f"loading config update from {config_update_path}")
            with open(config_update_path, "r") as f:
                config_update_fromyaml = yaml.safe_load(f)
            utils.recursive_update(self.config, config_update_fromyaml)

        # 3. load update config from dict
        if config_update is not None:
            logger.info(f"Applying config update from dict")
            utils.recursive_update(self.config, config_update)

        if not "output" in self.config:
            self.config["output"] = output_folder

        logger.progress(f"version: {alphadia.__version__}")
        # print hostname, date with day format and time
        logger.progress(f"hostname: {socket.gethostname()}")
        now = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        logger.progress(f"date: {now}")

        self.load_library(spec_lib_path)

        torch.set_num_threads(self.config['general']['thread_count'])


    @property
    def raw_file_list(self) -> typing.List[str]:
        """List of input files locations."""
        return self._raw_file_list

    @raw_file_list.setter
    def raw_file_list(self, raw_file_list: typing.List[str]):
        self._raw_file_list = raw_file_list

    @property
    def config(self) -> typing.Dict:
        """Dict with all configuration parameters for the extraction."""
        return self._config

    @config.setter
    def config(self, config: typing.Dict) -> None:
        self._config = config

    @property
    def spectral_library(self) -> SpecLibFlat:
        """Flattened Spectral Library."""
        return self._spectral_library

    @spectral_library.setter
    def spectral_library(self, spectral_library: SpecLibFlat) -> None:
        self._spectral_library = spectral_library

    def load_library(self, spec_lib_path):
        if "fasta_list" in self.config:
            fasta_files = self.config["fasta_list"]
        else:
            fasta_files = []

        # the import pipeline is used to transform arbitrary spectral libraries into the alphabase format
        # afterwards, the library can be saved as hdf5 and used for further processing
        import_pipeline = libtransform.ProcessingPipeline(
            [
                libtransform.DynamicLoader(),
                libtransform.PrecursorInitializer(),
                libtransform.AnnotateFasta(fasta_files),
                libtransform.IsotopeGenerator(n_isotopes=4),
                libtransform.RTNormalization(),
            ]
        )

        # the prepare pipeline is used to prepare an alphabase compatible spectral library for extraction
        prepare_pipeline = libtransform.ProcessingPipeline([
            libtransform.DecoyGenerator(decoy_type='diann'),
            libtransform.FlattenLibrary(self.config['search_advanced']['top_k_fragments']),
            libtransform.InitFlatColumns(),
            libtransform.LogFlatLibraryStats(),
        ])

        speclib = import_pipeline(spec_lib_path)
        if self.config["library_loading"]["save_hdf"]:
            speclib.save_hdf(os.path.join(self.output_folder, "speclib.hdf"))

        self.spectral_library = prepare_pipeline(speclib)

    def get_run_data(self):
        """Generator for raw data and spectral library."""

        if self.spectral_library is None:
            raise ValueError("no spectral library loaded")

        # iterate over raw files and yield raw data and spectral library
        for i, raw_location in enumerate(self.raw_file_list):
            raw_name = Path(raw_location).stem
            logger.progress(
                f"Loading raw file {i+1}/{len(self.raw_file_list)}: {raw_name}"
            )

            yield raw_name, raw_location, self.spectral_library

    def run(
        self,
        figure_path=None,
        neptune_token=None,
        neptune_tags=[],
        keep_decoys=False,
        fdr=0.01,
    ):
        logger.progress("Starting Search Workflows")

        workflow_folder_list = []

        for raw_name, dia_path, speclib in self.get_run_data():
            workflow = None
            try:
                workflow = peptidecentric.PeptideCentricWorkflow(
                    raw_name,
                    self.config,
                )

                workflow_folder_list.append(workflow.path)

                # check if the raw file is already processed
                psm_location = os.path.join(workflow.path, "psm.tsv")
                frag_location = os.path.join(workflow.path, "frag.tsv")

                if self.config["general"]["reuse_quant"]:
                    if os.path.exists(
                        psm_location
                    ):  # and os.path.exists(frag_location):
                        logger.info(f"Found existing quantification for {raw_name}")
                        continue
                    logger.info(f"No existing quantification found for {raw_name}")

                workflow.load(dia_path, speclib)
                workflow.calibration()

                df = workflow.extraction()
                df = df[df["qval"] <= self.config["fdr"]["fdr"]]

                if self.config["multiplexing"]["multiplexed_quant"]:
                    df = workflow.requantify(df)
                    df = df[df["qval"] <= self.config["fdr"]["fdr"]]
                df["run"] = raw_name
                df.to_csv(psm_location, sep="\t", index=False)

                workflow.reporter.log_string(f"Finished workflow for {raw_name}")
                workflow.reporter.context.__exit__(None, None, None)
                del workflow

            except Exception as e:
                # get full traceback
                import traceback

                traceback.print_exc()

                print(e)
                logger.error(f"Workflow failed for {raw_name} with error {e}")
                continue

        output = outputtransform.SearchPlanOutput(self.config, self.output_folder)
        output.build_output(workflow_folder_list)

# native imports
import logging
import os
import socket
from datetime import datetime
from pathlib import Path

import alphabase
import alpharaw
import alphatims
import directlfq
import peptdeep

# third party imports
import torch
from alphabase.constants import modification
from alphabase.spectral_library.base import SpecLibBase

# alpha family imports
from alphabase.spectral_library.flat import SpecLibFlat

import alphadia

# alphadia imports
from alphadia import libtransform, outputtransform
from alphadia.workflow import peptidecentric, reporting
from alphadia.workflow.config import Config

logger = logging.getLogger()

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
        "Label:13C(12)@K": {"composition": "C(12)"},
    }
)


class Plan:
    def __init__(
        self,
        output_folder: str,
        raw_path_list: list[str] | None = None,
        library_path: str | None = None,
        fasta_path_list: list[str] | None = None,
        config: dict | None = None,
        config_base_path: str | None = None,
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
        if config is None:
            config = {}
        if fasta_path_list is None:
            fasta_path_list = []
        if raw_path_list is None:
            raw_path_list = []
        self.output_folder = output_folder
        reporting.init_logging(self.output_folder)

        logger.progress("          _      _         ___ ___   _   ")
        logger.progress("     __ _| |_ __| |_  __ _|   \_ _| /_\  ")
        logger.progress("    / _` | | '_ \ ' \\/ _` | |) | | / _ \ ")
        logger.progress("    \__,_|_| .__/_||_\__,_|___/___/_/ \_\\")
        logger.progress("           |_|                           ")
        logger.progress("")

        self.spectral_library = None
        self.raw_path_list = raw_path_list
        self.library_path = library_path
        self.fasta_path_list = fasta_path_list

        logger.progress(f"version: {alphadia.__version__}")

        # print hostname, date with day format and time
        logger.progress(f"hostname: {socket.gethostname()}")
        now = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        logger.progress(f"date: {now}")

        # print environment
        self.log_environment()

        # 1. default config path is not defined in the function definition to account for for different path separators on different OS
        if config_base_path is None:
            # default yaml config location under /misc/config/config.yaml
            config_base_path = os.path.join(
                os.path.dirname(__file__), "constants", "default.yaml"
            )

        self.config = Config()

        logger.info(f"loading default config from {config_base_path}")
        self.config.from_yaml(config_base_path)

        # 2. load update config from dict
        if isinstance(config, dict):
            update_config = Config("user defined")
            update_config.from_dict(config)
        else:
            update_config = config

        self.config.update([update_config], print_modifications=True)

        if "output" not in self.config:
            self.config["output"] = output_folder

        # set log level
        level_to_set = self.config["general"]["log_level"]
        level_code = logging.getLevelName(level_to_set)
        logger.setLevel(level_code)

        self.load_library()

        torch.set_num_threads(self.config["general"]["thread_count"])

    @property
    def raw_path_list(self) -> list[str]:
        """List of input files locations."""
        return self._raw_path_list

    @raw_path_list.setter
    def raw_path_list(self, raw_path_list: list[str]):
        self._raw_path_list = raw_path_list

    @property
    def config(self) -> dict:
        """Dict with all configuration parameters for the extraction."""
        return self._config

    @config.setter
    def config(self, config: dict) -> None:
        self._config = config

    @property
    def spectral_library(self) -> SpecLibFlat:
        """Flattened Spectral Library."""
        return self._spectral_library

    @spectral_library.setter
    def spectral_library(self, spectral_library: SpecLibFlat) -> None:
        self._spectral_library = spectral_library

    def log_environment(self):
        logger.progress("=================== Environment ===================")
        logger.progress(f"{'alphatims':<15} : {alphatims.__version__:}")
        logger.progress(f"{'alpharaw':<15} : {alpharaw.__version__}")
        logger.progress(f"{'alphabase':<15} : {alphabase.__version__}")
        logger.progress(f"{'alphapeptdeep':<15} : {peptdeep.__version__}")
        logger.progress(f"{'directlfq':<15} : {directlfq.__version__}")
        logger.progress("===================================================")

    def load_library(self):
        """
        Load or build spectral library as configured.

        Steps 1 to 3 are performed depending on the quality and information in the spectral library.
        Step 4 is always performed to prepare the library for search.
        """

        def _parse_modifications(mod_str: str) -> list[str]:
            """Parse modification string."""
            return [] if mod_str == "" else mod_str.split(";")

        # 1. Check if library exists, else perform fasta digest
        dynamic_loader = libtransform.DynamicLoader()
        fasta_digest = libtransform.FastaDigest(
            enzyme=self.config["library_prediction"]["enzyme"],
            fixed_modifications=_parse_modifications(
                self.config["library_prediction"]["fixed_modifications"]
            ),
            variable_modifications=_parse_modifications(
                self.config["library_prediction"]["variable_modifications"]
            ),
            max_var_mod_num=self.config["library_prediction"]["max_var_mod_num"],
            missed_cleavages=self.config["library_prediction"]["missed_cleavages"],
            precursor_len=self.config["library_prediction"]["precursor_len"],
            precursor_charge=self.config["library_prediction"]["precursor_charge"],
            precursor_mz=self.config["library_prediction"]["precursor_mz"],
        )

        if self.library_path is None and self.config["library_prediction"]["predict"]:
            logger.progress("No library provided. Building library from fasta files.")
            spectral_library = fasta_digest(self.fasta_path_list)
        elif (
            self.library_path is None
            and not self.config["library_prediction"]["predict"]
        ):
            logger.error("No library provided and prediction disabled.")
            return
        else:
            spectral_library = dynamic_loader(self.library_path)

        # 2. Check if properties should be predicted

        if self.config["library_prediction"]["predict"]:
            logger.progress("Predicting library properties.")

            pept_deep_prediction = libtransform.PeptDeepPrediction(
                use_gpu=self.config["general"]["use_gpu"],
                fragment_mz=self.config["library_prediction"]["fragment_mz"],
                nce=self.config["library_prediction"]["nce"],
                instrument=self.config["library_prediction"]["instrument"],
                mp_process_num=self.config["general"]["thread_count"],
                checkpoint_folder_path=self.config["library_prediction"][
                    "checkpoint_folder_path"
                ],
                fragment_types=self.config["library_prediction"][
                    "fragment_types"
                ].split(";"),
                max_fragment_charge=self.config["library_prediction"][
                    "max_fragment_charge"
                ],
            )

            spectral_library = pept_deep_prediction(spectral_library)

        # 3. import library and harmoniza
        harmonize_pipeline = libtransform.ProcessingPipeline(
            [
                libtransform.PrecursorInitializer(),
                libtransform.AnnotateFasta(self.fasta_path_list),
                libtransform.IsotopeGenerator(
                    n_isotopes=4, mp_process_num=self.config["general"]["thread_count"]
                ),
                libtransform.RTNormalization(),
            ]
        )
        spectral_library = harmonize_pipeline(spectral_library)

        library_path = os.path.join(self.output_folder, "speclib.hdf")
        logger.info(f"Saving library to {library_path}")
        spectral_library.save_hdf(library_path)

        # 4. prepare library for search
        # This part is always performed, even if a fully compliant library is provided
        prepare_pipeline = libtransform.ProcessingPipeline(
            [
                libtransform.DecoyGenerator(
                    decoy_type="diann",
                    mp_process_num=self.config["general"]["thread_count"],
                ),
                libtransform.FlattenLibrary(
                    self.config["search_advanced"]["top_k_fragments"]
                ),
                libtransform.InitFlatColumns(),
                libtransform.LogFlatLibraryStats(),
            ]
        )

        self.spectral_library = prepare_pipeline(spectral_library)

    def get_run_data(self):
        """Generator for raw data and spectral library."""

        if self.spectral_library is None:
            raise ValueError("no spectral library loaded")

        # iterate over raw files and yield raw data and spectral library
        for i, raw_location in enumerate(self.raw_path_list):
            raw_name = Path(raw_location).stem
            logger.progress(
                f"Loading raw file {i+1}/{len(self.raw_path_list)}: {raw_name}"
            )

            yield raw_name, raw_location, self.spectral_library

    def run(
        self,
        figure_path=None,
        neptune_token=None,
        neptune_tags=None,
        keep_decoys=False,
        fdr=0.01,
    ):
        if neptune_tags is None:
            neptune_tags = []
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
                psm_location = os.path.join(workflow.path, "psm.parquet")
                frag_location = os.path.join(workflow.path, "frag.parquet")

                if self.config["general"]["reuse_quant"]:
                    if os.path.exists(psm_location) and os.path.exists(frag_location):
                        logger.info(f"Found existing quantification for {raw_name}")
                        continue
                    logger.info(f"No existing quantification found for {raw_name}")

                workflow.load(dia_path, speclib)
                workflow.calibration()

                psm_df, frag_df = workflow.extraction()
                psm_df = psm_df[psm_df["qval"] <= self.config["fdr"]["fdr"]]

                if self.config["multiplexing"]["enabled"]:
                    psm_df = workflow.requantify(psm_df)
                    psm_df = psm_df[psm_df["qval"] <= self.config["fdr"]["fdr"]]

                if self.config["transfer_library"]["enabled"]:
                    psm_df, frag_df = workflow.requantify_fragments(psm_df)

                psm_df["run"] = raw_name
                psm_df.to_parquet(psm_location, index=False)
                frag_df.to_parquet(frag_location, index=False)

                workflow.reporter.log_string(f"Finished workflow for {raw_name}")
                workflow.reporter.context.__exit__(None, None, None)
                del workflow

            except peptidecentric.CalibrationError:
                # get full traceback
                logger.error(
                    f"Search for {raw_name} failed as not enough precursors were found for calibration"
                )
                logger.error("This can have the following reasons:")
                logger.error(
                    "   1. The sample was empty and therefore nor precursors were found"
                )
                logger.error("   2. The sample contains only very few precursors.")
                logger.error(
                    "      For small libraries, try to set recalibration_target to a lower value"
                )
                logger.error(
                    "      For large libraries, try to reduce the library size and reduce the calibration MS1 and MS2 tolerance"
                )
                logger.error(
                    "   3. There was a fundamental issue with search parameters"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Search for {raw_name} failed with error {e}", exc_info=True
                )
                raise e

        try:
            base_spec_lib = SpecLibBase()
            base_spec_lib.load_hdf(
                os.path.join(self.output_folder, "speclib.hdf"), load_mod_seq=True
            )

            output = outputtransform.SearchPlanOutput(self.config, self.output_folder)
            output.build(workflow_folder_list, base_spec_lib)

        except Exception as e:
            logger.error(f"Output failed with error {e}", exc_info=True)
            raise e

        logger.progress("=================== Search Finished ===================")

    def clean(self):
        if not self.config["library_loading"]["save_hdf"]:
            os.remove(os.path.join(self.output_folder, "speclib.hdf"))

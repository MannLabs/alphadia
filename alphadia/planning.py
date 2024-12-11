# native imports
import logging
import os
import socket
from collections import defaultdict
from collections.abc import Generator
from datetime import datetime
from importlib import metadata
from pathlib import Path

import alphabase
import alpharaw
import alphatims
import directlfq
import numpy as np
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
from alphadia.exceptions import CustomError
from alphadia.workflow import peptidecentric, reporting
from alphadia.workflow.base import WorkflowBase
from alphadia.workflow.config import Config

logger = logging.getLogger()

SPECLIB_FILE_NAME = "speclib.hdf"


class Plan:  # TODO rename -> SearchStep, planning.py -> search_step.py
    def __init__(
        self,
        output_folder: str,
        raw_path_list: list[str] | None = None,
        library_path: str | None = None,
        fasta_path_list: list[str] | None = None,
        config: dict | Config | None = None,
        config_base_path: str | None = None,
        extra_config: dict | None = None,
        quant_path: str | None = None,
        step_name: str | None = None,
    ) -> None:
        """Highest level class to plan a DIA Search.
        Owns the input file list, speclib and the config.
        Performs required manipulation of the spectral library like transforming RT scales and adding columns.

        Parameters
        ----------

        output_folder : str
            output folder to save the results

        raw_path_list : list
            list of input file locations

        library_path : str, optional
            path to the spectral library file. If not provided, the library is built from fasta files

        fasta_path_list : list, optional
            list of fasta file locations to build the library from

        config_base_path : str, optional
            user-provided yaml file containing the default config.

        config : dict, optional
            user-provided dict to update the default config. Can be used for debugging purposes etc.

        extra_config : dict, optional
            dict to update the final config. Used for multistep searches.

        quant_path : str, optional
            path to directory to save the quantification results (psm & frag parquet files). If not provided, the results are saved in the usual workflow folder

        step_name : str, optional
            name of the step to run. Will be used to distinguish output data between different steps in a multistep search.

        """

        if config is None:
            config = {}
        if fasta_path_list is None:
            fasta_path_list = []
        if raw_path_list is None:
            raw_path_list = []

        self.output_folder = output_folder
        self.raw_path_list = raw_path_list
        self.library_path = library_path
        self.fasta_path_list = fasta_path_list
        self.quant_path = quant_path

        self.spectral_library = None
        self.estimators = None

        # needs to be done before any logging:
        reporting.init_logging(self.output_folder)

        self._print_logo()

        self._print_environment()

        self._config = self._init_config(
            config, extra_config, output_folder, config_base_path
        )

        level_to_set = self._config["general"]["log_level"]
        level_code = logging.getLevelName(level_to_set)
        logger.setLevel(level_code)

        self.init_alphabase()
        self.load_library()

        torch.set_num_threads(self._config["general"]["thread_count"])

    def _print_logo(self) -> None:
        """Print the alphadia logo and version."""
        logger.progress("          _      _         ___ ___   _   ")
        logger.progress(r"     __ _| |_ __| |_  __ _|   \_ _| /_\  ")
        logger.progress("    / _` | | '_ \\ ' \\/ _` | |) | | / _ \\ ")
        logger.progress("    \\__,_|_| .__/_||_\\__,_|___/___/_/ \\_\\")
        logger.progress("           |_|                           ")
        logger.progress("")
        logger.progress(f"version: {alphadia.__version__}")

    def _init_config(
        self,
        user_config: dict | Config,
        extra_config: dict,
        output_folder: str,
        config_base_path: str | None,
    ):
        """Initialize the config with default values and update with user defined values."""

        # default config path is not defined in the function definition to account for different path separators on different OS
        if config_base_path is None:
            # default yaml config location under /misc/config/config.yaml
            config_base_path = os.path.join(
                os.path.dirname(__file__), "constants", "default.yaml"
            )

        logger.info(f"loading default config from {config_base_path}")
        config = Config()
        config.from_yaml(config_base_path)

        # load update config from dict
        if isinstance(user_config, dict):
            update_config = Config("user defined")
            update_config.from_dict(user_config)
        elif isinstance(user_config, Config):
            update_config = user_config
        else:
            raise ValueError("'config' parameter must be of type 'dict' or 'Config'")

        config.update([update_config], print_modifications=True)

        if "output" not in config:
            config["output"] = output_folder

        if extra_config is not None:
            update_config = Config("multistep search")
            update_config.from_dict(extra_config)
            config.update([update_config], print_modifications=True)

        return config

    @property
    def config(self) -> Config:
        """Dict with all configuration parameters for the extraction."""
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        self._config = config

    @property
    def spectral_library(self) -> SpecLibFlat:
        """Flattened Spectral Library."""
        return self._spectral_library

    @spectral_library.setter
    def spectral_library(self, spectral_library: SpecLibFlat) -> None:
        self._spectral_library = spectral_library

    def _print_environment(self) -> None:
        """Log information about the python environment."""

        logger.progress(f"hostname: {socket.gethostname()}")
        now = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        logger.progress(f"date: {now}")

        logger.progress("================ AlphaX Environment ===============")
        logger.progress(f"{'alphatims':<15} : {alphatims.__version__:}")
        logger.progress(f"{'alpharaw':<15} : {alpharaw.__version__}")
        logger.progress(f"{'alphabase':<15} : {alphabase.__version__}")
        logger.progress(f"{'alphapeptdeep':<15} : {peptdeep.__version__}")
        logger.progress(f"{'directlfq':<15} : {directlfq.__version__}")
        logger.progress("===================================================")

        logger.progress("================= Pip Environment =================")
        pip_env = [
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in metadata.distributions()
        ]
        logger.progress(" ".join(pip_env))
        logger.progress("===================================================")

    def init_alphabase(self):
        """Init alphabase by registering custom modifications."""

        # register custom modifications
        if "custom_modifications" in self.config:
            n_modifications = len(self.config["custom_modifications"])
            logging.info(f"Registering {n_modifications} custom modifications")

            modification.add_new_modifications(self.config["custom_modifications"])

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

        prediction_config = self.config["library_prediction"]

        fasta_digest = libtransform.FastaDigest(
            enzyme=prediction_config["enzyme"],
            fixed_modifications=_parse_modifications(
                prediction_config["fixed_modifications"]
            ),
            variable_modifications=_parse_modifications(
                prediction_config["variable_modifications"]
            ),
            max_var_mod_num=prediction_config["max_var_mod_num"],
            missed_cleavages=prediction_config["missed_cleavages"],
            precursor_len=prediction_config["precursor_len"],
            precursor_charge=prediction_config["precursor_charge"],
            precursor_mz=prediction_config["precursor_mz"],
        )

        if self.library_path is None and prediction_config["predict"]:
            logger.progress("No library provided. Building library from fasta files.")
            spectral_library = fasta_digest(self.fasta_path_list)
        elif self.library_path is None and not prediction_config["predict"]:
            logger.error("No library provided and prediction disabled.")
            return
        else:
            spectral_library = dynamic_loader(self.library_path)

        # 2. Check if properties should be predicted

        thread_count = self.config["general"]["thread_count"]

        if prediction_config["predict"]:
            logger.progress("Predicting library properties.")

            pept_deep_prediction = libtransform.PeptDeepPrediction(
                use_gpu=self.config["general"]["use_gpu"],
                fragment_mz=prediction_config["fragment_mz"],
                nce=prediction_config["nce"],
                instrument=prediction_config["instrument"],
                mp_process_num=thread_count,
                peptdeep_model_path=prediction_config["peptdeep_model_path"],
                peptdeep_model_type=prediction_config["peptdeep_model_type"],
                fragment_types=prediction_config["fragment_types"].split(";"),
                max_fragment_charge=prediction_config["max_fragment_charge"],
            )

            spectral_library = pept_deep_prediction(spectral_library)

        # 3. import library and harmonize
        harmonize_pipeline = libtransform.ProcessingPipeline(
            [
                libtransform.PrecursorInitializer(),
                libtransform.AnnotateFasta(self.fasta_path_list),
                libtransform.IsotopeGenerator(
                    n_isotopes=4, mp_process_num=thread_count
                ),
                libtransform.RTNormalization(),
            ]
        )
        spectral_library = harmonize_pipeline(spectral_library)

        multiplexing_config = self.config["library_multiplexing"]
        if multiplexing_config["enabled"]:
            multiplexing = libtransform.MultiplexLibrary(
                multiplex_mapping=multiplexing_config["multiplex_mapping"],
                input_channel=multiplexing_config["input_channel"],
            )
            spectral_library = multiplexing(spectral_library)

        library_path = os.path.join(self.output_folder, SPECLIB_FILE_NAME)
        logger.info(f"Saving library to {library_path}")
        spectral_library.save_hdf(library_path)

        # 4. prepare library for search
        # This part is always performed, even if a fully compliant library is provided
        prepare_pipeline = libtransform.ProcessingPipeline(
            [
                libtransform.DecoyGenerator(
                    decoy_type="diann",
                    mp_process_num=thread_count,
                ),
                libtransform.FlattenLibrary(
                    self.config["search_advanced"]["top_k_fragments"]
                ),
                libtransform.InitFlatColumns(),
                libtransform.LogFlatLibraryStats(),
            ]
        )

        self.spectral_library = prepare_pipeline(spectral_library)

    def get_run_data(self) -> Generator[tuple[str, str, SpecLibFlat]]:
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
    ):
        logger.progress("Starting Search Workflows")

        workflow_folder_list = []
        single_estimators = defaultdict(list)  # needs a better name

        for raw_name, dia_path, speclib in self.get_run_data():
            workflow = None
            try:
                workflow = self._process_raw_file(dia_path, raw_name, speclib)
                workflow_folder_list.append(workflow.path)

                self._update_estimators(single_estimators, workflow)

            except CustomError as e:
                _log_exception_event(e, raw_name, workflow)
                continue

            except Exception as e:
                _log_exception_event(e, raw_name, workflow)
                raise e

            finally:
                if workflow and workflow.reporter:
                    workflow.reporter.log_string(f"Finished workflow for {raw_name}")
                    workflow.reporter.context.__exit__(None, None, None)
                del workflow

        self.estimators = self._aggregate_estimators(single_estimators)

        try:
            base_spec_lib = SpecLibBase()
            base_spec_lib.load_hdf(
                os.path.join(self.output_folder, SPECLIB_FILE_NAME), load_mod_seq=True
            )

            output = outputtransform.SearchPlanOutput(self.config, self.output_folder)
            output.build(workflow_folder_list, base_spec_lib)
        except Exception as e:
            _log_exception_event(e)
            raise e
        finally:
            self._clean()

        logger.progress("=================== Search Finished ===================")

    @staticmethod
    def _update_estimators(
        estimators: dict, workflow: peptidecentric.PeptideCentricWorkflow
    ):
        """Update the estimators with the current workflow."""

        estimators["ms1_accuracy"].append(
            workflow.calibration_manager.get_estimator("precursor", "mz").metrics[
                "median_accuracy"
            ]
        )

    @staticmethod
    def _aggregate_estimators(estimators: dict):
        """Aggregate the estimators over workflows."""

        agg_estimators = {}
        for name, values in estimators.items():
            agg_estimators[name] = np.median(values)

        return agg_estimators

    def _process_raw_file(
        self, dia_path: str, raw_name: str, speclib: SpecLibFlat
    ) -> peptidecentric.PeptideCentricWorkflow:
        """Process a single raw file."""

        workflow = peptidecentric.PeptideCentricWorkflow(
            raw_name,
            self.config,
            quant_path=self.quant_path,
        )

        # check if the raw file is already processed
        psm_location = os.path.join(workflow.path, "psm.parquet")
        frag_location = os.path.join(workflow.path, "frag.parquet")

        if self.config["general"]["reuse_quant"]:
            if os.path.exists(psm_location) and os.path.exists(frag_location):
                logger.info(f"Found existing quantification for {raw_name}")
                return workflow
            logger.info(f"No existing quantification found for {raw_name}")

        workflow.load(dia_path, speclib)

        workflow.timing_manager.set_start_time("optimization")
        workflow.search_parameter_optimization()
        workflow.timing_manager.set_end_time("optimization")

        workflow.timing_manager.set_start_time("extraction")
        psm_df, frag_df = workflow.extraction()
        workflow.timing_manager.set_end_time("extraction")
        workflow.timing_manager.save()

        psm_df = psm_df[psm_df["qval"] <= self.config["fdr"]["fdr"]]

        if self.config["multiplexing"]["enabled"]:
            psm_df = workflow.requantify(psm_df)
            psm_df = psm_df[psm_df["qval"] <= self.config["fdr"]["fdr"]]

        if self.config["transfer_library"]["enabled"]:
            psm_df, frag_df = workflow.requantify_fragments(psm_df)

        psm_df["run"] = raw_name
        psm_df.to_parquet(psm_location, index=False)
        frag_df.to_parquet(frag_location, index=False)

        return workflow

    def _clean(self):
        if not self.config["general"]["save_library"]:
            try:
                os.remove(os.path.join(self.output_folder, SPECLIB_FILE_NAME))
            except Exception as e:
                logger.exception(f"Error deleting library: {e}")


def _log_exception_event(
    e: Exception, raw_name: str | None = None, workflow: WorkflowBase | None = None
) -> None:
    """Log exception and emit event to reporter if available."""

    prefix = (
        "Error:" if raw_name is None else f"Search for {raw_name} failed with error:"
    )

    if isinstance(e, CustomError):
        logger.error(f"{prefix} {e.error_code} {e.msg}")
        logger.error(e.detail_msg)
    else:
        logger.error(f"{prefix} {e}", exc_info=True)

    if workflow is not None and workflow.reporter:
        workflow.reporter.log_string(
            value=str(e),
            verbosity="error",
        )
        workflow.reporter.log_event(name="exception", value=str(e), exception=e)

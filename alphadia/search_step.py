import logging
import os
from collections.abc import Generator
from pathlib import Path

import torch
from alphabase.constants import modification
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia import libtransform, outputtransform
from alphadia.constants.keys import ConfigKeys
from alphadia.exceptions import CustomError
from alphadia.workflow import peptidecentric, reporting
from alphadia.workflow.base import WorkflowBase
from alphadia.workflow.config import MULTISTEP_SEARCH, USER_DEFINED, Config

SPECLIB_FILE_NAME = "speclib.hdf"

logger = logging.getLogger()


class SearchStep:
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
    ) -> None:
        """Highest level class to plan a DIA search step.

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

        """

        if config is None:
            config = {}
        if fasta_path_list is None:
            fasta_path_list = []
        if raw_path_list is None:
            raw_path_list = []

        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        reporting.init_logging(self.output_folder)

        self.raw_path_list = raw_path_list
        self.library_path = library_path
        self.fasta_path_list = fasta_path_list
        self.quant_path = quant_path

        self.spectral_library = None

        self._config = self._init_config(
            config, extra_config, output_folder, config_base_path
        )

        logger.setLevel(logging.getLevelName(self._config["general"]["log_level"]))

        self.init_alphabase()
        self.load_library()

        torch.set_num_threads(self._config["general"]["thread_count"])

        self._log_inputs()

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

        logger.info(f"loading config from {config_base_path}")
        config = Config()
        config.from_yaml(config_base_path)

        config_updates = []
        # load update config from dict
        if isinstance(user_config, dict):
            user_config_update = Config(USER_DEFINED)
            user_config_update.from_dict(user_config)
            config_updates.append(user_config_update)
        elif isinstance(user_config, Config):
            config_updates.append(user_config)
        else:
            raise ValueError("'config' parameter must be of type 'dict' or 'Config'")

        if extra_config is not None:
            extra_config_update = Config(MULTISTEP_SEARCH)
            extra_config_update.from_dict(extra_config)
            # need to overwrite user-defined output folder here
            extra_config[ConfigKeys.OUTPUT_DIRECTORY] = output_folder
            config_updates.append(extra_config_update)

        config.update(config_updates, do_print=True)

        if ConfigKeys.OUTPUT_DIRECTORY not in config:
            config[ConfigKeys.OUTPUT_DIRECTORY] = output_folder
        # TODO need to update also raw_path_list, fasta_list, quant_dir, library in this way before dumping config to disk

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

    def init_alphabase(self):
        """Init alphabase by registering custom modifications."""

        new_modifications = {}
        for mod in self.config["custom_modifications"]:
            new_modifications[mod["name"]] = {"composition": mod["composition"]}

        if new_modifications:
            logging.info(f"Registering {len(new_modifications)} custom modifications")

            modification.add_new_modifications(new_modifications)

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

        if self.library_path is None and not prediction_config["predict"]:
            logger.error("No library provided and prediction disabled.")
            return
        elif self.library_path is None and prediction_config["predict"]:
            logger.progress("No library provided. Building library from fasta files.")

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
            spectral_library = fasta_digest(self.fasta_path_list)
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

        for raw_name, dia_path, speclib in self.get_run_data():
            workflow = None
            try:
                workflow = self._process_raw_file(dia_path, raw_name, speclib)
                workflow_folder_list.append(workflow.path)

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

    def _log_inputs(self):
        """Log all relevant inputs."""

        logger.info(f"Searching {len(self.raw_path_list)} files:")
        for f in self.raw_path_list:
            logger.info(f"  {os.path.basename(f)}")

        logger.info(f"Using {len(self.fasta_path_list)} fasta files:")
        for f in self.fasta_path_list:
            logger.info(f"  {f}")

        logger.info(f"Using library: {self.library_path}")
        logger.info(f"Saving output to: {self.output_folder}")


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

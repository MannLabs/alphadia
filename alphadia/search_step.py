import logging
import os
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from alphabase.constants import modification
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import ConfigKeys, SearchStepFiles
from alphadia.exceptions import ConfigError, CustomError, NoLibraryAvailableError
from alphadia.libtransform.base import ProcessingPipeline
from alphadia.libtransform.decoy import DecoyGenerator
from alphadia.libtransform.fasta_digest import FastaDigest
from alphadia.libtransform.flatten import (
    FlattenLibrary,
    InitFlatColumns,
    LogFlatLibraryStats,
)
from alphadia.libtransform.harmonize import (
    AnnotateFasta,
    IsotopeGenerator,
    PrecursorInitializer,
    RTNormalization,
)
from alphadia.libtransform.loader import DynamicLoader
from alphadia.libtransform.multiplex import MultiplexLibrary
from alphadia.libtransform.prediction import PeptDeepPrediction
from alphadia.outputtransform.search_plan_output import SearchPlanOutput
from alphadia.reporting.reporting import init_logging, move_existing_file
from alphadia.workflow.base import WorkflowBase
from alphadia.workflow.config import (
    MULTISTEP_SEARCH,
    USER_DEFINED,
    USER_DEFINED_CLI_PARAM,
    Config,
)
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow

SPECLIB_FILE_NAME = "speclib.hdf"
SPECLIB_FLAT_FILE_NAME = "speclib_flat.hdf"

logger = logging.getLogger()


class SearchStep:
    def __init__(
        self,
        output_folder: str,
        config: dict | Config | None = None,
        cli_config: dict | None = None,
        extra_config: dict | None = None,
    ) -> None:
        """Highest level class to plan a DIA search step.

        Owns the input file list, speclib and the config.
        Performs required manipulation of the spectral library like transforming RT scales and adding columns.

        Parameters
        ----------

        output_folder : str
            output folder to save the results

        config : dict, optional
            values to update the default config. Overrides values in `default.yaml`.

        cli_config : dict, optional
            additional config values (parameters from the command line). Overrides values in `config`.

        extra_config : dict, optional
            additional config values (parameters to orchestrate multistep searches). Overrides values in `config` and `cli_config`.

        """

        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        init_logging(self.output_folder)

        self._config = self._init_config(
            config, cli_config, extra_config, output_folder
        )
        self._save_config(output_folder)

        logger.setLevel(logging.getLevelName(self._config["general"]["log_level"]))

        self.raw_path_list = self._config[ConfigKeys.RAW_PATHS]
        self.library_path = self._config[ConfigKeys.LIBRARY_PATH]
        self.fasta_path_list = self._config[ConfigKeys.FASTA_PATHS]

        self.spectral_library: SpecLibFlat | None = None

        self._init_alphabase()

        torch.set_num_threads(self._config["general"]["thread_count"])

        self._np_rng = self._get_random_number_generator()

        self._log_inputs()

    def _save_config(self, output_folder: str) -> None:
        """Save the config to a file in the output folder, moving an existing file if necessary."""
        file_path = os.path.join(output_folder, "frozen_config.yaml")
        moved_path = move_existing_file(file_path)
        self._config.to_yaml(file_path)
        if moved_path:
            logging.info(f"Moved existing config file {file_path} to {moved_path}")

    @staticmethod
    def _init_config(
        user_config: dict | Config | None,
        cli_config: dict | None,
        extra_config: dict | None,
        output_folder: str,
    ) -> Config:
        """Initialize the config with default values and update with user defined values."""

        config = SearchStep._load_default_config()

        config_updates = []

        if user_config:
            logger.info("loading additional config provided via CLI")
            # load update config from dict
            if isinstance(user_config, Config):
                config_updates.append(user_config)
            else:
                user_config_update = Config(user_config, name=USER_DEFINED)
                config_updates.append(user_config_update)

        if cli_config:
            logger.info("loading additional config provided via CLI parameters")
            cli_config_update = Config(cli_config, name=USER_DEFINED_CLI_PARAM)
            config_updates.append(cli_config_update)

        if SearchStep._is_ng_activated(config, config_updates):
            ng_default_config = SearchStep._load_default_config(
                file_name="default_ng.yaml"
            )
            config_updates.insert(0, ng_default_config)

        # the update done for multi-step search needs to be last in order to overwrite any user-defined output folder
        if extra_config:
            extra_config_update = Config(extra_config, name=MULTISTEP_SEARCH)
            # need to overwrite user-defined output folder here to have correct value in config dump
            extra_config_update[ConfigKeys.OUTPUT_DIRECTORY] = output_folder
            config_updates.append(extra_config_update)

        if config_updates:
            config.update(config_updates, do_print=True)

        # Note: if not provided by CLI, output_folder is set to the value in config in cli.py
        if (
            current_config_output_folder := config.get(ConfigKeys.OUTPUT_DIRECTORY)
        ) is not None and current_config_output_folder != output_folder:
            logger.warning(
                f"Using output directory '{output_folder}' provided via CLI, the value specified in config ('{current_config_output_folder}') will be ignored."
            )
        config[ConfigKeys.OUTPUT_DIRECTORY] = output_folder

        return config

    @staticmethod
    def _load_default_config(file_name="default.yaml") -> Config:
        """Load default config from file."""
        default_config_path = os.path.join(
            os.path.dirname(__file__), "constants", file_name
        )
        logger.info(f"loading default config from {default_config_path}")
        config = Config()
        config.from_yaml(default_config_path)
        return config

    @staticmethod
    def _is_ng_activated(config: Config, config_updates: list[Config]) -> bool:
        """Decide if the extraction backend is 'ng'.

        If no updates are provided, the decision is based on the default config.
        If updates are provided, they are applied in order to be able to decide based on the final config.
        """
        if config_updates:
            tmp_updated_config = deepcopy(config)
            tmp_updated_config.update(config_updates)

            return tmp_updated_config["search"]["extraction_backend"] == "ng"

        return config["search"]["extraction_backend"] == "ng"

    def _get_random_number_generator(self) -> None | Generator:
        """Getnumpy random number generator if random state is set."""
        if (random_state := self._config["general"]["random_state"]) == -1:
            random_state = np.random.randint(0, 1_000_000)

        if random_state is not None:
            logging.info(f"Setting random state to {random_state}")
            return np.random.default_rng(random_state)

        return None

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

    def _init_alphabase(self):
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
        general_config = self.config["general"]
        prediction_config = self.config["library_prediction"]

        if self.library_path is None:
            if not prediction_config["enabled"]:
                raise NoLibraryAvailableError()
            else:
                logger.progress(
                    "No library provided. Building library from fasta files."
                )

                fasta_digest = FastaDigest(
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
        elif general_config["input_library_type"] == "flat":
            if general_config["save_mbr_library"]:
                # TODO gather such checks in a ConfigValidator class
                raise ConfigError(
                    "general.save_mbr_library",
                    "True",
                    "",
                    "Settings general.save_mbr_library = 'True' and general.input_library_type = 'flat' are incompatible.",
                )

            logger.progress("Loading library (type: flat) from disk..")
            speclib_flat = SpecLibFlat()
            speclib_flat.load_hdf(self.library_path)
            LogFlatLibraryStats()(speclib_flat)
            self.spectral_library = speclib_flat
            return

        else:
            logger.progress("Loading library (type: base) from disk..")
            dynamic_loader = DynamicLoader()
            spectral_library = dynamic_loader(self.library_path)

        # 2. Check if properties should be predicted

        thread_count = general_config["thread_count"]

        if prediction_config["enabled"]:
            logger.progress("Predicting library properties.")

            pept_deep_prediction = PeptDeepPrediction(
                use_gpu=general_config["use_gpu"],
                fragment_mz=prediction_config["fragment_mz"],
                nce=prediction_config["nce"],
                instrument=prediction_config["instrument"],
                mp_process_num=thread_count,
                peptdeep_model_path=prediction_config["peptdeep_model_path"],
                peptdeep_model_type=prediction_config["peptdeep_model_type"],
                fragment_types=prediction_config["fragment_types"],
                max_fragment_charge=prediction_config["max_fragment_charge"],
            )

            spectral_library = pept_deep_prediction(spectral_library)

        # 3. import library and harmonize
        harmonize_pipeline = ProcessingPipeline(
            [
                PrecursorInitializer(),
                AnnotateFasta(self.fasta_path_list),
                IsotopeGenerator(n_isotopes=4, mp_process_num=thread_count),
                RTNormalization(),
            ]
        )
        spectral_library = harmonize_pipeline(spectral_library)

        multiplexing_config = self.config["library_multiplexing"]
        if multiplexing_config["enabled"]:
            multiplexing = MultiplexLibrary(
                multiplex_mapping=multiplexing_config["multiplex_mapping"],
                input_channel=multiplexing_config["input_channel"],
            )
            spectral_library = multiplexing(spectral_library)

        if general_config["save_library"] or general_config["save_mbr_library"]:
            library_path = os.path.join(self.output_folder, SPECLIB_FILE_NAME)
            logger.info(f"Saving library to {library_path}")
            spectral_library.save_hdf(library_path)

        # 4. prepare library for search
        # This part is always performed, even if a fully compliant library is provided
        prepare_pipeline = ProcessingPipeline(
            [
                DecoyGenerator(
                    decoy_type="diann",
                    mp_process_num=thread_count,
                ),
                FlattenLibrary(
                    max(
                        self.config["search"]["top_k_fragments_selection"],
                        self.config["search"]["top_k_fragments_scoring"],
                    ),
                    self.config["search"]["min_fragment_intensity"],
                ),
                InitFlatColumns(),
                LogFlatLibraryStats(),
            ]
        )

        self.spectral_library = prepare_pipeline(spectral_library)

        if general_config["save_flat_library"]:
            library_path = os.path.join(self.output_folder, SPECLIB_FLAT_FILE_NAME)
            logger.info(f"Saving flat library to {library_path}")
            self.spectral_library.save_hdf(library_path)

    def _get_run_data(self) -> Generator[tuple[str, str, SpecLibFlat]]:
        """Generator for raw data and spectral library."""

        # iterate over raw files and yield raw data and spectral library
        for raw_location in self.raw_path_list:
            raw_name = Path(raw_location).stem
            yield raw_name, raw_location, self.spectral_library

    def run(
        self,
    ):
        """Run the search step.

        This has three main parts:
        1. Load or build the spectral library
        2. Iterate over all raw files and perform the search workflow
        3. Collect and summarize the results
        """

        if self.spectral_library is None:
            logger.progress("Loading spectral library")
            self.load_library()

        if not self.raw_path_list:
            logger.warning("No raw files provided, nothing to search.")
            return

        logger.progress("Starting Search Workflows")

        workflow_folder_list = []

        for i, (raw_name, dia_path, speclib) in enumerate(self._get_run_data()):
            workflow = None
            random_state = (
                None if self._np_rng is None else self._np_rng.integers(0, 1_000_000)
            )

            logger.progress(
                f"Loading raw file {i+1}/{len(self.raw_path_list)}: {raw_name}"
                + f" (random_state: {random_state})"
                if random_state is not None
                else ""
            )

            try:
                workflow = PeptideCentricWorkflow(
                    raw_name,
                    self.config,
                    quant_path=self.config["quant_directory"],
                    random_state=random_state,
                )
                workflow_path = Path(workflow.path)

                # check if the raw file is already processed, i.e. if all relevant files exist
                is_already_processed = False
                if self.config["general"]["reuse_quant"]:
                    required_files = [
                        SearchStepFiles.PSM_FILE_NAME,
                        SearchStepFiles.FRAG_FILE_NAME,
                    ]
                    if self.config["transfer_library"]["enabled"]:
                        required_files.append(SearchStepFiles.FRAG_TRANSFER_FILE_NAME)

                    if all(
                        (workflow_path / file_name).exists()
                        for file_name in required_files
                    ):
                        logger.info(
                            f"reuse_quant: found existing quantification for {raw_name}, skipping processing .."
                        )
                        is_already_processed = True
                    logger.info(
                        f"reuse_quant: found no existing quantification for {raw_name}, proceeding with processing .."
                    )

                if not is_already_processed:
                    self._process_raw_file(workflow, dia_path, speclib)

                workflow_folder_list.append(workflow.path)

            except Exception as e:
                _log_exception_event(e, raw_name, workflow)
                if isinstance(e, CustomError):
                    continue
                raise e

            finally:
                if workflow and workflow.reporter:
                    workflow.reporter.log_string(f"Finished workflow for {raw_name}")
                    workflow.reporter.context.__exit__(None, None, None)
                del workflow

        try:
            if self.config["general"]["save_mbr_library"]:
                # TODO: find a way to avoid loading the library again from disk
                base_spec_lib = SpecLibBase()
                base_spec_lib.load_hdf(
                    os.path.join(self.output_folder, SPECLIB_FILE_NAME),
                    load_mod_seq=True,
                )
            else:
                base_spec_lib = None

            output = SearchPlanOutput(self.config, self.output_folder)
            output.build(workflow_folder_list, base_spec_lib)
        except Exception as e:
            _log_exception_event(e)
            raise e
        finally:
            self._clean()

        logger.progress("=================== Search Finished ===================")

    def _process_raw_file(
        self, workflow: PeptideCentricWorkflow, dia_path: str, speclib: SpecLibFlat
    ) -> None:
        """Process a single raw file, storing the results on disk."""
        workflow.timing_manager.set_start_time("total")

        workflow.load(dia_path, speclib)

        workflow.search_parameter_optimization()

        psm_df, frag_df = workflow.extraction()

        if self.config["multiplexing"]["enabled"]:
            psm_df = workflow.requantify(psm_df)

        if self.config["transfer_library"]["enabled"]:
            psm_df, frag_transfer_df = workflow.requantify_fragments(psm_df)
        else:
            frag_transfer_df = None

        workflow_path = Path(workflow.path)
        psm_df["run"] = workflow.instance_name

        for file_name, df in {
            SearchStepFiles.PSM_FILE_NAME: psm_df,
            SearchStepFiles.FRAG_FILE_NAME: frag_df,
            SearchStepFiles.FRAG_TRANSFER_FILE_NAME: frag_transfer_df,
        }.items():
            if df is not None:
                file_path = workflow_path / file_name
                workflow.reporter.log_string(f"Saving results to {file_path}")
                df.to_parquet(file_path, index=False)

        workflow.timing_manager.set_end_time("total")
        workflow.timing_manager.save()

    def _clean(self):
        if not self.config["general"]["save_library"]:
            library_path = Path(self.output_folder) / SPECLIB_FILE_NAME
            try:
                library_path.unlink(missing_ok=True)
            except Exception as e:
                logger.exception(f"Error removing library {library_path}: {e}")

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

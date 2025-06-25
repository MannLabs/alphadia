"""Search plan for single- and multistep search."""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alphadia.constants.keys import ConfigKeys, StatOutputKeys
from alphadia.outputtransform.search_plan_output import (
    SearchPlanOutput,
)
from alphadia.reporting import reporting
from alphadia.reporting.logging import print_environment, print_logo
from alphadia.search_step import (
    SearchStep,
    logger,
)

# TODO the names of the steps need to be adjusted
TRANSFER_STEP_NAME = "transfer"
LIBRARY_STEP_NAME = "library"
MBR_STEP_NAME = "mbr"

# TODO we need to make sure basic users settings are compatible with each step in multistep search
# e.g. by printing warning messages on the biggest mistakes

CONSTANTS_FOLDER_PATH = Path(os.path.dirname(__file__)) / "constants"


class SearchPlan:
    """Search plan for single- and multistep search."""

    def __init__(
        self,
        output_directory: str,
        config: dict | None = None,
        cli_params_config: dict | None = None,
    ):
        """Initialize search plan.

        In case of a single step search, this can be considered as a slim wrapper around the SearchStep class.
        In case of a multistep search, this class orchestrates the different steps, their data paths,
         and passes information from one step to the next.

        Parameters
        ----------
        output_directory:
            Output directory.
        config:
            Configuration provided by user (loaded from file and/or dictionary)
        cli_params_config
            config-like dictionary of parameters directly provided by CLI
        """
        reporting.init_logging(output_directory)

        self._output_dir: Path = Path(output_directory)
        self._user_config: dict = config if config is not None else {}
        self._cli_params_config: dict = (
            cli_params_config if cli_params_config is not None else {}
        )

        # these are the default paths if the library step is the only one
        self._library_step_output_dir: Path = self._output_dir

        # multistep search:
        self._multistep_config: dict | None = None
        self._transfer_step_output_dir: Path | None = None

        # We read the default values for the transfer_step_enabled and mbr_step_enabled directly from the default.yaml,
        # but then forget about them. They will still end up correctly in the frozen_config.yaml as they are read
        # again from the default.yaml later
        user_config_general = self._user_config.get("general", {})
        with (CONSTANTS_FOLDER_PATH / "default.yaml").open() as f:
            default_config_general = yaml.safe_load(f)["general"]
        self._transfer_step_enabled = user_config_general.get(
            "transfer_step_enabled", default_config_general["transfer_step_enabled"]
        )
        self._mbr_step_enabled = user_config_general.get(
            "mbr_step_enabled", default_config_general["mbr_step_enabled"]
        )

        if self._transfer_step_enabled or self._mbr_step_enabled:
            self._update_paths()
            with (CONSTANTS_FOLDER_PATH / "multistep.yaml").open() as f:
                self._multistep_config = yaml.safe_load(f)

    def _update_paths(self) -> None:
        """Set directories for the different steps.

        If the transfer step is enabled, the quant and library paths for the library step are pointed to the
            output of the transfer step.
        If the mbr step is enabled, the quant and library paths for the mbr step are pointed to the output of the
            library step. Also, the output path for the library step is adjusted to be in a subdirectory of the original output path.
        """

        # in case transfer step is enabled, we need to adjust the library step settings
        if self._transfer_step_enabled:
            self._transfer_step_output_dir = self._output_dir / TRANSFER_STEP_NAME

        # in case mbr step is enabled, we need to adjust the library step settings
        if self._mbr_step_enabled:
            self._library_step_output_dir = self._output_dir / LIBRARY_STEP_NAME

    def run_plan(self):
        """Run the search plan.

        Depending on what steps are to be run, the relevant information (e.g. file paths or thresholds) is passed
        from one to the next step via 'extra config'.
        """
        print_logo()
        print_environment()

        extra_config_for_library_step = (
            self._multistep_config[LIBRARY_STEP_NAME]
            if self._transfer_step_enabled or self._mbr_step_enabled
            else {}
        )

        optimized_values_config = {}
        if self._transfer_step_enabled:
            logger.info(f"Running step '{TRANSFER_STEP_NAME}'")
            # predict library (once for all files, file-independent), search all files (emb. parallel), quantify all files together (combine all files) (outer.sh-steps 1, 2, 3)
            # output: DL model
            self.run_step(
                self._transfer_step_output_dir,
                self._multistep_config[TRANSFER_STEP_NAME],
            )

            extra_config_from_transfer_step = {
                "library_prediction": {
                    "peptdeep_model_path": os.path.join(
                        self._transfer_step_output_dir, SearchPlanOutput.TRANSFER_MODEL
                    ),
                    "enabled": True,  # the step following the 'transfer' step needs to have this
                }
            }

            optimized_values_config = self._get_optimized_values_config(
                self._transfer_step_output_dir
            )

            extra_config_for_library_step = (
                extra_config_for_library_step
                | extra_config_from_transfer_step
                | optimized_values_config
            )

        # same as transfer_step
        # output: MBR library
        logger.info(f"Running step '{LIBRARY_STEP_NAME}'")
        self.run_step(
            self._library_step_output_dir,
            extra_config_for_library_step,
        )

        if self._mbr_step_enabled:
            # (outer.sh-steps 4,5)
            logger.info(f"Running step '{MBR_STEP_NAME}'")
            if optimized_values_config == {}:
                optimized_values_config = self._get_optimized_values_config(
                    self._library_step_output_dir
                )

            mbr_step_library_path = str(
                self._library_step_output_dir / f"{SearchPlanOutput.LIBRARY_OUTPUT}.hdf"
            )

            mbr_step_extra_config = (
                self._multistep_config[MBR_STEP_NAME]
                | optimized_values_config
                | {ConfigKeys.LIBRARY_PATH: mbr_step_library_path}
            )
            self.run_step(
                self._output_dir,
                mbr_step_extra_config,
            )

    def run_step(
        self,
        output_directory: Path,
        extra_config: dict,
    ) -> None:
        """Run a single step of the search plan."""
        step = SearchStep(
            output_folder=str(output_directory),
            config=self._user_config,
            cli_config=self._cli_params_config,
            extra_config=extra_config,
        )
        step.run()

    @staticmethod
    def _get_optimized_values_config(output_folder: Path) -> dict:
        """Extract optimized values from a previous step and return an update to the config."""

        df = pd.read_csv(
            output_folder / f"{SearchPlanOutput.STAT_OUTPUT}.tsv", sep="\t"
        )
        target_ms1_tolerance = np.nanmedian(
            df[f"{StatOutputKeys.OPTIMIZATION_PREFIX}{StatOutputKeys.MS1_ERROR}"]
        )
        target_ms2_tolerance = np.nanmedian(
            df[f"{StatOutputKeys.OPTIMIZATION_PREFIX}{StatOutputKeys.MS2_ERROR}"]
        )

        if np.isnan(target_ms1_tolerance) and np.isnan(target_ms2_tolerance):
            logger.warning(
                "Could not extract target_ms1_tolerance and target_ms2_tolerance from previous step."
            )
            return {}

        extra_config = defaultdict(dict)

        if not np.isnan(target_ms1_tolerance):
            extra_config["search"]["target_ms1_tolerance"] = target_ms1_tolerance

        if not np.isnan(target_ms2_tolerance):
            extra_config["search"]["target_ms2_tolerance"] = target_ms2_tolerance

        # Notes:
        # - ms1 & ms2 's calibration is valid across all steps, not dependent on transfer learning
        # - target_mobility_tolerance and target_rt_tolerance should be reoptimized with the lib resulting from transfer learning step

        logger.info(f"Extracted extra_config from previous step: {extra_config}")
        return dict(extra_config)

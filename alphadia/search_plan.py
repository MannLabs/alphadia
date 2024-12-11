"""Search plan for multistep search."""

import os
from typing import Literal

import yaml

from alphadia.outputtransform import SearchPlanOutput
from alphadia.planning import SPECLIB_FILE_NAME, Plan, logger
from alphadia.workflow.base import QUANT_FOLDER_NAME

STEP1_NAME = "transfer"
STEP2_NAME = "library"
STEP3_NAME = "mbr"


class SearchPlan:
    """Search plan for multistep search."""

    def __init__(
        self,
        config: dict,
        output_directory: str,
        library_path: str,
        fasta_path_list: list[str],
        quant_dir: str | None,
        raw_path_list: list[str],
    ):
        """Initialize search plan for multistep search.

        Parameters
        ----------
        config:
            User configuration.
        output_directory:
            Output directory.
        library_path:
            Library path.
        fasta_path_list:
            List of fasta paths.
        quant_dir:
            Quantification directory holding previous results.
        raw_path_list
            List of raw paths.
        """

        self._user_config = config
        self.output_directory = output_directory
        self.library_path = library_path
        self.fasta_path_list = fasta_path_list
        self.quant_dir = quant_dir
        self.raw_path_list = raw_path_list

        multistep_search_config = self._user_config.get("multistep_search", {})
        self.step1_enabled = multistep_search_config.get("transfer_step_enabled", False)
        self.step3_enabled = multistep_search_config.get("mbr_step_enabled", False)

        with open(
            os.path.join(os.path.dirname(__file__), "constants", "multistep.yaml")
        ) as f:
            self.multistep_config = yaml.safe_load(f)

        self.step2_name = None
        self.step2_quant_dir = self.quant_dir
        self.step2_library_path = self.library_path
        self.step2_output_dir = self.output_directory

        self.step1_output_dir = None
        if self.step1_enabled:
            self.step1_output_dir = os.path.join(self.output_directory, STEP1_NAME)

            self.step2_quant_dir = os.path.join(
                self.step1_output_dir, QUANT_FOLDER_NAME
            )
            self.step2_library_path = os.path.join(
                self.step1_output_dir, SPECLIB_FILE_NAME
            )
            self.step2_output_dir = os.path.join(self.output_directory, STEP2_NAME)
            self.step2_name = STEP2_NAME

        self.step3_quant_dir = None
        self.step3_library_path = None
        self.step3_output_dir = None
        if self.step3_enabled:
            self.step3_quant_dir = os.path.join(
                self.step2_output_dir, QUANT_FOLDER_NAME
            )
            self.step3_library_path = os.path.join(
                self.step2_output_dir, SPECLIB_FILE_NAME
            )
            self.step3_output_dir = os.path.join(self.output_directory, STEP3_NAME)

    def run_plan(self):
        """Run the search plans."""

        step2_extra_config = {}
        if self.step1_enabled:
            # predict library (once for all files, file-independent), search all files (emb. parallel), quantify all files together (combine all files)
            # (outer.sh-steps 1, 2, 3)
            # output: DL model
            self.run_step(
                self.step1_output_dir,
                self.library_path,
                self.multistep_config[STEP1_NAME],
                self.quant_dir,
                step=STEP1_NAME,
            )

            add_config = {
                "library_prediction": {
                    "peptdeep_model_path": os.path.join(
                        self.step1_output_dir, SearchPlanOutput.TRANSFER_MODEL
                    )
                }
            }
            step2_extra_config = self.multistep_config[STEP2_NAME] | add_config

        # same as step1
        # output: MBR library
        library_plan = self.run_step(
            self.step2_output_dir,
            self.step2_library_path,
            step2_extra_config,
            self.step2_quant_dir,
            step=self.step2_name,
        )

        if self.step3_enabled and library_plan is not None:
            # (outer.sh-steps 4,5)
            add_config = self._update_config_from_library_plan(library_plan)
            extra_config = self.multistep_config[STEP2_NAME] | add_config
            self.run_step(
                self.step3_output_dir,
                self.step3_library_path,
                extra_config,
                self.step3_quant_dir,
                step=STEP3_NAME,
            )

    def run_step(
        self,
        output_directory: str,
        library_path: str,
        extra_config: dict,
        quant_dir: str | None,
        step: Literal["transfer", "library", "mbr"] | None,
    ) -> Plan:
        plan = Plan(
            output_directory,
            raw_path_list=self.raw_path_list,
            library_path=library_path,
            fasta_path_list=self.fasta_path_list,
            config=self._user_config,
            extra_config=extra_config,
            quant_path=quant_dir,
            step_name=step,
        )
        plan.run()
        return plan

    def _update_config_from_library_plan(self, library_plan):
        # take any required information from library_plan and pass it via config to the next step, e.g.
        new_config = self._user_config | {  # noqa: F841
            "search": {"target_ms1_tolerance": library_plan.estimators["ms1_accuracy"]}
        }
        # think about hardcoding everything here
        logger.info(f"Using ms1_accuracy: {library_plan.estimators['ms1_accuracy']}")
        return new_config

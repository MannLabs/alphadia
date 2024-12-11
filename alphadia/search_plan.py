"""Search plan for multistep search."""

import os
from pathlib import Path

import yaml

from alphadia.outputtransform import SearchPlanOutput
from alphadia.planning import SPECLIB_FILE_NAME, Plan
from alphadia.workflow import reporting
from alphadia.workflow.base import QUANT_FOLDER_NAME

TRANSFER_STEP_NAME = "transfer"
LIBRARY_STEP_NAME = "library"
MBR_STEP_NAME = "mbr"


class SearchPlan:
    """Search plan for multistep search."""

    def __init__(
        self,
        output_directory: str,
        raw_path_list: list[str],
        library_path: str,
        fasta_path_list: list[str],
        config: dict,
        quant_dir: str | None,
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
        self._output_dir = Path(output_directory)
        self._library_path = Path(library_path)
        self._fasta_path_list = fasta_path_list
        self._quant_dir = None if quant_dir is None else Path(quant_dir)
        self._raw_path_list = raw_path_list

        # these are the default settings if the library step is the only one
        self._library_step_quant_dir = self._quant_dir
        self._library_step_library_path = self._library_path
        self._library_step_output_dir = self._output_dir

        # multistep search:
        self._multistep_config = None
        self._transfer_step_output_dir = None
        self._mbr_step_quant_dir = None
        self._mbr_step_library_path = None

        multistep_search_config = self._user_config.get("multistep_search", {})
        self._transfer_step_enabled = multistep_search_config.get(
            "transfer_step_enabled", False
        )
        self._mbr_step_enabled = multistep_search_config.get("mbr_step_enabled", False)

        if self._transfer_step_enabled or self._mbr_step_enabled:
            self._update_paths()
            with (
                Path(os.path.dirname(__file__)) / "constants" / "multistep.yaml"
            ).open() as f:
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
            self._library_step_quant_dir = (
                self._transfer_step_output_dir / QUANT_FOLDER_NAME
            )
            self._library_step_library_path = (
                self._transfer_step_output_dir / SPECLIB_FILE_NAME
            )

        # in case mbr step is enabled, we need to adjust the library step settings
        if self._mbr_step_enabled:
            self._library_step_output_dir = self._output_dir / LIBRARY_STEP_NAME
            self._mbr_step_quant_dir = self._library_step_output_dir / QUANT_FOLDER_NAME
            self._mbr_step_library_path = (
                self._library_step_output_dir / SPECLIB_FILE_NAME
            )

    def run_plan(self):
        """Run the search plan.

        Depending on what steps are to be run, the relevant information (e.g. file paths or thresholds) is passed
        from one to the next step via 'extra config'.
        """

        reporting.init_logging(str(self._output_dir))
        Plan.print_logo()
        Plan.print_environment()

        library_step_extra_config = {}
        if self._transfer_step_enabled:
            # predict library (once for all files, file-independent), search all files (emb. parallel), quantify all files together (combine all files) (outer.sh-steps 1, 2, 3)
            # output: DL model
            self.run_step(
                self._transfer_step_output_dir,
                self._library_path,
                self._multistep_config[TRANSFER_STEP_NAME],
                self._quant_dir,
            )

            add_config = {
                "library_prediction": {
                    "peptdeep_model_path": os.path.join(
                        self._transfer_step_output_dir, SearchPlanOutput.TRANSFER_MODEL
                    )
                }
            }
            library_step_extra_config = (
                self._multistep_config[LIBRARY_STEP_NAME] | add_config
            )

        # same as transfer_step
        # output: MBR library
        library_plan = self.run_step(
            self._library_step_output_dir,
            self._library_step_library_path,
            library_step_extra_config,
            self._library_step_quant_dir,
        )

        if self._mbr_step_enabled:
            # (outer.sh-steps 4,5)
            add_config = self._update_config_from_library_plan(library_plan)
            mbr_step_extra_config = self._multistep_config[MBR_STEP_NAME] | add_config
            self.run_step(
                self._output_dir,
                self._mbr_step_library_path,
                mbr_step_extra_config,
                self._mbr_step_quant_dir,
            )

    def run_step(
        self,
        output_directory: Path,
        library_path: Path,
        extra_config: dict,
        quant_dir: Path | None,
    ) -> Plan:
        """Run a single step of the search plan."""
        plan = Plan(
            str(output_directory),
            raw_path_list=self._raw_path_list,
            library_path=str(library_path),
            fasta_path_list=self._fasta_path_list,
            config=self._user_config,
            extra_config=extra_config,
            quant_path=None if quant_dir is None else str(quant_dir),
        )
        plan.run()
        return plan

    def _update_config_from_library_plan(self, library_plan: Plan) -> dict:
        """Update the config based on the library plan."""
        new_config = {}
        # take any required information from library_plan and pass it via config to the next step, e.g.
        # new_config = self._user_config | {  # noqa: F841
        #     "search": {"target_ms1_tolerance": library_plan.estimators["ms1_accuracy"]}
        # }

        # logger.info(f"Using ms1_accuracy: {library_plan.estimators['ms1_accuracy']}")
        return new_config

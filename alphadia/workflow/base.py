# native imports
import logging
import os

# alpha family imports
from alphabase.spectral_library.base import SpecLibBase

# alphadia imports
from alphadia.data import alpharaw, bruker
from alphadia.workflow import manager, reporting

# third party imports

logger = logging.getLogger()

TEMP_FOLDER = ".progress"


class WorkflowBase:
    """Base class for all workflows. This class is responsible for creating the workflow folder.
    It also initializes the calibration_manager and fdr_manager for the workflow.
    """

    CALIBRATION_MANAGER_PATH = "calibration_manager.pkl"
    OPTIMIZATION_MANAGER_PATH = "optimization_manager.pkl"
    FDR_MANAGER_PATH = "fdr_manager.pkl"
    FIGURE_PATH = "figures"

    def __init__(
        self,
        instance_name: str,
        config: dict,
    ) -> None:
        """
        Parameters
        ----------

        instance_name: str
            Name for the particular workflow instance. this will usually be the name of the raw file

        parent_path: str
            Path where the workflow folder will be created

        config: dict
            Configuration for the workflow. This will be used to initialize the calibration manager and fdr manager

        """
        self._instance_name = instance_name
        self._parent_path = os.path.join(config["output"], TEMP_FOLDER)
        self._config = config

        if not os.path.exists(self.parent_path):
            logger.info(f"Creating parent folder for workflows at {self.parent_path}")
            os.mkdir(self.parent_path)

        if not os.path.exists(self.path):
            logger.info(
                f"Creating workflow folder for {self.instance_name} at {self.path}"
            )
            os.mkdir(self.path)

    def load(
        self,
        dia_data_path: str,
        spectral_library: SpecLibBase,
    ) -> None:
        self.reporter = reporting.Pipeline(
            backends=[
                reporting.LogBackend(),
                reporting.JSONLBackend(path=self.path),
                reporting.FigureBackend(path=self.path),
            ]
        )
        self.reporter.context.__enter__()
        self.reporter.log_event("section_start", {"name": "Initialize Workflow"})

        self.reporter.log_event("loading_data", {"progress": 0})
        # load the raw data
        self._dia_data = self._get_dia_data_object(dia_data_path)
        self.reporter.log_event("loading_data", {"progress": 1})

        # load the spectral library
        self._spectral_library = spectral_library.copy()

        # initialize the calibration manager
        self._calibration_manager = manager.CalibrationManager(
            self.config["calibration_manager"],
            path=os.path.join(self.path, self.CALIBRATION_MANAGER_PATH),
            load_from_file=self.config["general"]["reuse_calibration"],
            reporter=self.reporter,
        )

        if not self._dia_data.has_mobility:
            logging.info("Disabling ion mobility calibration")
            self._calibration_manager.disable_mobility_calibration()

        # initialize the optimization manager
        self._optimization_manager = manager.OptimizationManager(
            self.config["optimization_manager"],
            path=os.path.join(self.path, self.OPTIMIZATION_MANAGER_PATH),
            load_from_file=self.config["general"]["reuse_calibration"],
            figure_path=os.path.join(self.path, self.FIGURE_PATH),
            reporter=self.reporter,
        )

        self.reporter.log_event("section_stop", {})

    @property
    def instance_name(self) -> str:
        """Name for the particular workflow instance. this will usually be the name of the raw file"""
        return self._instance_name

    @property
    def parent_path(self) -> str:
        """Path where the workflow folder will be created"""
        return self._parent_path

    @property
    def path(self) -> str:
        """Path to the workflow folder"""
        return os.path.join(self.parent_path, self.instance_name)

    @property
    def config(self) -> dict:
        """Configuration for the workflow."""
        return self._config

    @property
    def calibration_manager(self) -> str:
        """Calibration manager for the workflow. Owns the RT, IM, MZ calibration and the calibration data"""
        return self._calibration_manager

    @property
    def optimization_manager(self) -> str:
        """Optimization manager for the workflow. Owns the optimization data"""
        return self._optimization_manager

    @property
    def spectral_library(self) -> SpecLibBase:
        """Spectral library for the workflow. Owns the spectral library data"""
        return self._spectral_library

    @property
    def dia_data(
        self,
    ) -> bruker.TimsTOFTransposeJIT | alpharaw.AlphaRawJIT:
        """DIA data for the workflow. Owns the DIA data"""
        return self._dia_data

    def _get_dia_data_object(
        self, dia_data_path: str
    ) -> bruker.TimsTOFTranspose | alpharaw.AlphaRaw:
        """Get the correct data class depending on the file extension of the DIA data file.

        Parameters
        ----------

        dia_data_path: str
            Path to the DIA data file

        Returns
        -------
        typing.Union[bruker.TimsTOFTranspose, thermo.Thermo],
            TimsTOFTranspose object containing the DIA data

        """
        file_extension = os.path.splitext(dia_data_path)[1]

        if self.config["general"]["wsl"]:
            # copy file to /tmp
            import shutil

            tmp_path = "/tmp"
            tmp_dia_data_path = os.path.join(tmp_path, os.path.basename(dia_data_path))
            shutil.copyfile(dia_data_path, tmp_dia_data_path)
            dia_data_path = tmp_dia_data_path

        if file_extension.lower() == ".d" or file_extension.lower() == ".hdf":
            self.reporter.log_metric("raw_data_type", "bruker")
            dia_data = bruker.TimsTOFTranspose(
                dia_data_path,
                mmap_detector_events=self.config["general"]["mmap_detector_events"],
            )

        elif file_extension.lower() == ".raw":
            self.reporter.log_metric("raw_data_type", "thermo")
            # check if cv selection exists
            cv = None
            if (
                "raw_data_loading" in self.config
                and "cv" in self.config["raw_data_loading"]
            ):
                cv = self.config["raw_data_loading"]["cv"]

            dia_data = alpharaw.Thermo(
                dia_data_path,
                process_count=self.config["general"]["thread_count"],
                astral_ms1=self.config["general"]["astral_ms1"],
                cv=cv,
            )

        elif file_extension.lower() == ".mzml":
            self.reporter.log_metric("raw_data_type", "mzml")

            dia_data = alpharaw.MzML(
                dia_data_path,
                process_count=self.config["general"]["thread_count"],
            )

        elif file_extension.lower() == ".wiff":
            self.reporter.log_metric("raw_data_type", "sciex")

            dia_data = alpharaw.Sciex(
                dia_data_path,
                process_count=self.config["general"]["thread_count"],
            )

        else:
            raise ValueError(
                f"Unknown file extension {file_extension} for file at {dia_data_path}"
            )

        # remove tmp file if wsl
        if self.config["general"]["wsl"]:
            os.remove(tmp_dia_data_path)
        return dia_data

# native imports
import logging
import os

# alpha family imports
from alphabase.spectral_library.base import SpecLibBase

from alphadia.constants.keys import ConfigKeys

# alphadia imports
from alphadia.data import alpharaw_wrapper, bruker
from alphadia.workflow import manager, reporting
from alphadia.workflow.config import Config
from alphadia.workflow.managers.raw_file_manager import RawFileManager

# third party imports

logger = logging.getLogger()

QUANT_FOLDER_NAME = "quant"


class WorkflowBase:
    """Base class for all workflows. This class is responsible for creating the workflow folder.
    It also initializes the calibration_manager and fdr_manager for the workflow.
    """

    RAW_FILE_MANAGER_PKL_NAME = "raw_file_manager.pkl"
    CALIBRATION_MANAGER_PKL_NAME = "calibration_manager.pkl"
    OPTIMIZATION_MANAGER_PKL_NAME = "optimization_manager.pkl"
    TIMING_MANAGER_PKL_NAME = "timing_manager.pkl"
    FDR_MANAGER_PKL_NAME = "fdr_manager.pkl"
    FIGURES_FOLDER_NAME = "figures"

    def __init__(
        self,
        instance_name: str,
        config: Config,
        quant_path: str = None,
    ) -> None:
        """
        Parameters
        ----------

        instance_name: str
            Name for the particular workflow instance. this will usually be the name of the raw file

        config: dict
            Configuration for the workflow. This will be used to initialize the calibration manager and fdr manager

        quant_path: str
            path to directory holding quant folders, relevant for distributed searches

        """
        self._instance_name: str = instance_name
        self._quant_path: str = quant_path or os.path.join(
            config[ConfigKeys.OUTPUT_DIRECTORY], QUANT_FOLDER_NAME
        )
        logger.info(f"Quantification results path: {self._quant_path}")

        self._config: Config = config
        self.reporter: reporting.Pipeline | None = None
        self._dia_data: bruker.TimsTOFTranspose | alpharaw_wrapper.AlphaRaw | None = (
            None
        )
        self._spectral_library: SpecLibBase | None = None
        self._calibration_manager: manager.CalibrationManager | None = None
        self._optimization_manager: manager.OptimizationManager | None = None
        self._timing_manager: manager.TimingManager | None = None

        if not os.path.exists(self._quant_path):
            logger.info(f"Creating parent folder for workflows at {self._quant_path}")
            os.makedirs(self._quant_path)

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

        # load the raw data
        self.reporter.log_event("loading_data", {"progress": 0})
        raw_file_manager = RawFileManager(
            self.config,
            path=os.path.join(self.path, self.RAW_FILE_MANAGER_PKL_NAME),
            reporter=self.reporter,
        )

        self._dia_data = raw_file_manager.get_dia_data_object(dia_data_path)
        raw_file_manager.save()

        self.reporter.log_event("loading_data", {"progress": 1})

        # load the spectral library
        self._spectral_library = spectral_library.copy()

        # initialize the calibration manager
        self._calibration_manager = manager.CalibrationManager(
            path=os.path.join(self.path, self.CALIBRATION_MANAGER_PKL_NAME),
            load_from_file=self.config["general"]["reuse_calibration"],
            reporter=self.reporter,
        )

        if not self._dia_data.has_mobility:
            logging.info("Disabling ion mobility calibration")
            self._calibration_manager.disable_mobility_calibration()

        # initialize the optimization manager
        self._optimization_manager = manager.OptimizationManager(
            self.config,
            gradient_length=self.dia_data.rt_values.max(),
            path=os.path.join(self.path, self.OPTIMIZATION_MANAGER_PKL_NAME),
            load_from_file=self.config["general"]["reuse_calibration"],
            figure_path=os.path.join(self.path, self.FIGURES_FOLDER_NAME),
            reporter=self.reporter,
        )

        self._timing_manager = manager.TimingManager(
            path=os.path.join(self.path, self.TIMING_MANAGER_PKL_NAME),
            load_from_file=self.config["general"]["reuse_calibration"],
        )

        self.reporter.log_event("section_stop", {})

    @property
    def instance_name(self) -> str:
        """Name for the particular workflow instance. this will usually be the name of the raw file"""
        return self._instance_name

    @property
    def quant_path(self) -> str:
        """Path where the workflow folder will be created"""
        return self._quant_path

    @property
    def path(self) -> str:
        """Path to the workflow folder"""
        return os.path.join(self._quant_path, self.instance_name)

    @property
    def config(self) -> Config:
        """Configuration for the workflow."""
        return self._config

    @property
    def calibration_manager(self) -> manager.CalibrationManager:
        """Calibration manager for the workflow. Owns the RT, IM, MZ calibration and the calibration data"""
        return self._calibration_manager

    @property
    def optimization_manager(self) -> manager.OptimizationManager:
        """Optimization manager for the workflow. Owns the optimization data"""
        return self._optimization_manager

    @property
    def timing_manager(self) -> manager.TimingManager:
        """Optimization manager for the workflow. Owns the timing data"""
        return self._timing_manager

    @property
    def spectral_library(self) -> SpecLibBase | None:
        """Spectral library for the workflow. Owns the spectral library data"""
        return self._spectral_library

    @property
    def dia_data(
        self,
    ) -> bruker.TimsTOFTranspose | alpharaw_wrapper.AlphaRawJIT:
        """DIA data for the workflow. Owns the DIA data"""
        return self._dia_data

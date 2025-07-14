import logging
import os

from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import ConfigKeys
from alphadia.constants.settings import FIGURES_FOLDER_NAME
from alphadia.raw_data import DiaData
from alphadia.reporting import reporting
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.managers.raw_file_manager import RawFileManager
from alphadia.workflow.managers.timing_manager import TimingManager

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

        quant_path_ = quant_path or os.path.join(
            config[ConfigKeys.OUTPUT_DIRECTORY], QUANT_FOLDER_NAME
        )

        logger.info(f"Quantification results path: {quant_path_}")

        self._path = os.path.join(quant_path_, self._instance_name)

        self._figure_path: str = (
            os.path.join(self.path, FIGURES_FOLDER_NAME)
            if config[ConfigKeys.GENERAL][ConfigKeys.SAVE_FIGURES]
            else None
        )

        self._config: Config = config
        self.reporter: reporting.Pipeline | None = None
        self._dia_data: DiaData | None = None
        self._spectral_library: SpecLibFlat | None = None
        self._calibration_manager: CalibrationManager | None = None
        self._optimization_manager: OptimizationManager | None = None
        self._timing_manager: TimingManager | None = None

        for path in [self._figure_path, self.path]:
            if path and not os.path.exists(path):
                logger.info(f"Creating folder {path}")
                os.makedirs(
                    path,
                    exist_ok=True,
                )

    def load(
        self,
        dia_data_path: str,
        spectral_library: SpecLibFlat,
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
        self._spectral_library: SpecLibFlat = spectral_library.copy()

        # initialize the calibration manager
        self._calibration_manager = CalibrationManager(
            path=os.path.join(self.path, self.CALIBRATION_MANAGER_PKL_NAME),
            load_from_file=self.config["general"]["reuse_calibration"],
            has_mobility=self._dia_data.has_mobility,
            reporter=self.reporter,
        )

        # initialize the optimization manager
        self._optimization_manager = OptimizationManager(
            self.config,
            gradient_length=self.dia_data.rt_values.max(),
            path=os.path.join(self.path, self.OPTIMIZATION_MANAGER_PKL_NAME),
            load_from_file=self.config["general"]["reuse_calibration"],
            figure_path=self._figure_path,
            reporter=self.reporter,
        )

        self._timing_manager = TimingManager(
            path=os.path.join(self.path, self.TIMING_MANAGER_PKL_NAME),
            load_from_file=self.config["general"]["reuse_calibration"],
        )

        self.reporter.log_event("section_stop", {})

    @property
    def path(self) -> str:
        """Path to the workflow folder, e.g. `first_search/quant/raw_file_xyz.raw`"""
        return self._path

    @property
    def config(self) -> Config:
        """Configuration for the workflow."""
        return self._config

    @property
    def calibration_manager(self) -> CalibrationManager:
        """Calibration manager for the workflow. Owns the RT, IM, MZ calibration and the calibration data"""
        return self._calibration_manager

    @property
    def optimization_manager(self) -> OptimizationManager:
        """Optimization manager for the workflow. Owns the optimization data"""
        return self._optimization_manager

    @property
    def timing_manager(self) -> TimingManager:
        """Optimization manager for the workflow. Owns the timing data"""
        return self._timing_manager

    @property
    def spectral_library(self) -> SpecLibFlat | None:
        """Spectral library for the workflow. Owns the spectral library data"""
        return self._spectral_library

    @property
    def dia_data(
        self,
    ) -> DiaData:
        """DIA data for the workflow. Owns the DIA data"""
        return self._dia_data

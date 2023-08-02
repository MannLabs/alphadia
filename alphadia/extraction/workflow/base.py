import os
import logging
logger = logging.getLogger()
import tempfile
import numpy as np
import pandas as pd

from alphadia.extraction import data
from alphadia.extraction.workflow import manager

from alphabase.spectral_library.base import SpecLibBase

class WorkflowBase():
    """Base class for all workflows. This class is responsible for creating the workflow folder.
    It also initializes the calibration_manager and fdr_manager for the workflow.
    """
    TEMP_FOLDER = ".progress"
    CALIBRATION_MANAGER_PATH = "calibration_manager.pkl"
    OPTIMIZATION_MANAGER_PATH = "optimization_manager.pkl"
    FDR_MANAGER_PATH = "fdr_manager.pkl"
    FIGURE_PATH = "figures"

    def __init__(self,
        instance_name: str,
        config: dict,
        dia_data_path: str,
        spectral_library: SpecLibBase,
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
        self.run = None
        self._instance_name = instance_name
        self._parent_path = os.path.join(config['output'],self.TEMP_FOLDER)
        self._config = config
        
        if not os.path.exists(self.parent_path):
            logger.info(f"Creating parent folder for workflows at {self.parent_path}")
            os.mkdir(self.parent_path)
        
        if not os.path.exists(self.path):
            logger.info(f"Creating workflow folder for {self.instance_name} at {self.path}")
            os.mkdir(self.path)

        if not os.path.exists(os.path.join(self.path, self.FIGURE_PATH)):
            os.mkdir(os.path.join(self.path, self.FIGURE_PATH))

        # load the raw data
        self._dia_data = self._get_dia_data_object(dia_data_path)

        # load the spectral library
        self._spectral_library = spectral_library.copy()

        # initialize the calibration manager
        self._calibration_manager = manager.CalibrationManager(
            config['calibration_manager'],
            path = os.path.join(self.path, self.CALIBRATION_MANAGER_PATH),
            load_from_file = config['general']['reuse_calibration']
        )
        # initialize the optimization manager
        self._optimization_manager = manager.OptimizationManager(
            config['optimization_manager'],
            path = os.path.join(self.path, self.OPTIMIZATION_MANAGER_PATH),
            load_from_file = config['general']['reuse_calibration']
        )

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
    def dia_data(self) -> data.TimsTOFTransposeJIT:
        """DIA data for the workflow. Owns the DIA data"""
        return self._dia_data
    
    def _get_dia_data_object(
            self, 
            dia_data_path: str
        ) -> data.TimsTOFTranspose:
        """ Get the correct data class depending on the file extension of the DIA data file.

        Parameters
        ----------

        dia_data_path: str
            Path to the DIA data file

        Returns
        -------
        data.TimsTOFTranspose
            TimsTOFTranspose object containing the DIA data
        
        """
        file_extension = os.path.splitext(dia_data_path)[1]

        if file_extension == '.d':
            return data.TimsTOFTranspose(dia_data_path)
        elif file_extension == '.hdf':
            return data.TimsTOFTranspose(dia_data_path)
        else:
            raise ValueError(f'Unknown file extension {file_extension} for file at {dia_data_path}')
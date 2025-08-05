"""Manager handling the raw data file and its statistics."""

import logging
import os

import numpy as np

from alphadia.raw_data import DiaData
from alphadia.raw_data.alpharaw_wrapper import AlphaRawBase, MzML, Sciex, Thermo
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.workflow.config import Config
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


class RawFileManager(BaseManager):
    def __init__(
        self,
        config: None | Config = None,
        path: None | str = None,
        load_from_file: bool = False,
        **kwargs,
    ):
        """Handles raw file loading and contains information on the raw file."""
        self.stats = {}  # needs to be before super().__init__ to avoid overwriting loaded values

        super().__init__(path=path, load_from_file=load_from_file, **kwargs)

        self._config: Config = config

        # deliberately not storing the dia_data object as an instance variable to avoid the saved manager file being too large

        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

    def get_dia_data_object(self, dia_data_path: str) -> DiaData:
        """Get the correct data class depending on the file extension of the DIA data file.

        Parameters
        ----------

        dia_data_path: str
            Path to the DIA data file

        Returns
        -------
        DiaData
            object containing the DIA data

        """
        file_extension = os.path.splitext(dia_data_path)[1]

        if file_extension.lower() == ".d":
            raw_data_type = "bruker"
            dia_data = TimsTOFTranspose(
                dia_data_path,
                mmap_detector_events=self._config["general"]["mmap_detector_events"],
            )

        elif file_extension.lower() == ".hdf":
            raw_data_type = "alpharaw"
            dia_data = AlphaRawBase(dia_data_path)

        elif file_extension.lower() == ".raw":
            raw_data_type = "thermo"

            dia_data = Thermo(
                dia_data_path,
                process_count=self._config["general"]["thread_count"],
                astral_ms1=self._config["general"]["astral_ms1"],
            )

        elif file_extension.lower() == ".mzml":
            raw_data_type = "mzml"

            dia_data = MzML(dia_data_path)

        elif file_extension.lower() == ".wiff":
            raw_data_type = "sciex"

            dia_data = Sciex(dia_data_path)

        else:
            raise ValueError(
                f"Unknown file extension {file_extension} for file at {dia_data_path}"
            )

        self.reporter.log_metric("raw_data_type", raw_data_type)

        self._calc_stats(dia_data)

        self._log_stats()

        return dia_data

    def _calc_stats(self, dia_data: DiaData):
        """Calculate statistics from the DIA data."""
        rt_values = dia_data.rt_values
        cycle = dia_data.cycle

        stats = {}
        stats["rt_limit_min"] = rt_values.min()
        stats["rt_limit_max"] = rt_values.max()

        cycle_length = cycle.shape[1]
        stats["cycle_length"] = cycle_length
        stats["cycle_duration"] = np.diff(rt_values[::cycle_length]).mean()
        stats["cycle_number"] = len(rt_values) // cycle_length

        flat_cycle = cycle.flatten()
        flat_cycle = flat_cycle[flat_cycle > 0]

        stats["msms_range_min"] = flat_cycle.min()
        stats["msms_range_max"] = flat_cycle.max()

        self.stats = stats

    def _log_stats(self):
        """Log the statistics calculated from the DIA data."""
        rt_duration = self.stats["rt_limit_max"] - self.stats["rt_limit_min"]

        logger.info(
            f"{'RT (min)':<20}: {self.stats['rt_limit_min']/60:.1f} - {self.stats['rt_limit_max']/60:.1f}"
        )
        logger.info(f"{'RT duration (sec)':<20}: {rt_duration:.1f}")
        logger.info(f"{'RT duration (min)':<20}: {rt_duration/60:.1f}")

        logger.info(f"{'Cycle len (scans)':<20}: {self.stats['cycle_length']:.0f}")
        logger.info(f"{'Cycle len (sec)':<20}: {self.stats['cycle_duration']:.2f}")
        logger.info(f"{'Number of cycles':<20}: {self.stats['cycle_number']:.0f}")

        logger.info(
            f"{'MS2 range (m/z)':<20}: {self.stats['msms_range_min']:.1f} - {self.stats['msms_range_max']:.1f}"
        )

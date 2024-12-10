"""Manager handling the raw data file and its statistics."""

import logging
import os

import numpy as np

from alphadia.data import alpharaw_wrapper, bruker
from alphadia.workflow.config import Config
from alphadia.workflow.manager import BaseManager

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

        # deliberately not saving the actual raw data here to avoid the saved manager file being too large

        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

    def get_dia_data_object(
        self, dia_data_path: str
    ) -> bruker.TimsTOFTranspose | alpharaw_wrapper.AlphaRaw:
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

        if file_extension.lower() == ".d":
            raw_data_type = "bruker"
            dia_data = bruker.TimsTOFTranspose(
                dia_data_path,
                mmap_detector_events=self._config["general"]["mmap_detector_events"],
            )

        elif file_extension.lower() == ".hdf":
            raw_data_type = "alpharaw"
            dia_data = alpharaw_wrapper.AlphaRawBase(
                dia_data_path,
                process_count=self._config["general"]["thread_count"],
            )

        elif file_extension.lower() == ".raw":
            raw_data_type = "thermo"

            cv = self._config.get("raw_data_loading", {}).get("cv")

            dia_data = alpharaw_wrapper.Thermo(
                dia_data_path,
                process_count=self._config["general"]["thread_count"],
                astral_ms1=self._config["general"]["astral_ms1"],
                cv=cv,
            )

        elif file_extension.lower() == ".mzml":
            raw_data_type = "mzml"

            dia_data = alpharaw_wrapper.MzML(
                dia_data_path,
                process_count=self._config["general"]["thread_count"],
            )

        elif file_extension.lower() == ".wiff":
            raw_data_type = "sciex"

            dia_data = alpharaw_wrapper.Sciex(
                dia_data_path,
                process_count=self._config["general"]["thread_count"],
            )

        else:
            raise ValueError(
                f"Unknown file extension {file_extension} for file at {dia_data_path}"
            )

        self.reporter.log_metric("raw_data_type", raw_data_type)

        self._calc_stats(dia_data)

        self._log_stats()

        return dia_data

    def _calc_stats(
        self, dia_data: bruker.TimsTOFTranspose | alpharaw_wrapper.AlphaRaw
    ):
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

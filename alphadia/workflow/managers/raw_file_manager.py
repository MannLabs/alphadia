"""Manager handling the raw data file and its statistics."""

import logging
import os

import numpy as np

from alphadia.data import alpharaw_wrapper, bruker
from alphadia.workflow.manager import BaseManager

logger = logging.getLogger()


class RawFileManager(BaseManager):
    def __init__(
        self,
        config: None | dict = None,
        path: None | str = None,
        **kwargs,
    ):
        """Contains and updates timing information for the portions of the workflow."""
        super().__init__(path=path, load_from_file=False, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        self._config = config

        self._stats = {}

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

        is_wsl = self._config["general"]["wsl"]
        if is_wsl:
            # copy file to /tmp # TODO why is that?
            import shutil

            tmp_path = "/tmp"
            tmp_dia_data_path = os.path.join(tmp_path, os.path.basename(dia_data_path))
            shutil.copyfile(dia_data_path, tmp_dia_data_path)
            dia_data_path = tmp_dia_data_path

        if file_extension.lower() == ".d" or file_extension.lower() == ".hdf":
            raw_data_type = "bruker"
            dia_data = bruker.TimsTOFTranspose(
                dia_data_path,
                mmap_detector_events=self._config["general"]["mmap_detector_events"],
            )

        elif file_extension.lower() == ".raw":
            raw_data_type = "thermo"

            cv = self._config.get(["raw_data_loading"], {}).get("cv")

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

        # remove tmp file if wsl
        if is_wsl:
            os.remove(tmp_dia_data_path)

        return dia_data

    def calc_stats(self, dia_data: bruker.TimsTOFTranspose | alpharaw_wrapper.AlphaRaw):
        """Calculate statistics from the DIA data."""
        rt_values = dia_data.rt_values
        cycle = dia_data.cycle

        self._stats["rt_limits"] = rt_values.min() / 60, rt_values.max() / 60
        self._stats["rt_duration_sec"] = rt_values.max() - rt_values.min()

        cycle_length = cycle.shape[1]
        self._stats["cycle_length"] = cycle_length
        self._stats["cycle_duration"] = np.diff(rt_values[::cycle_length]).mean()
        self._stats["cycle_number"] = len(rt_values) // cycle_length

        flat_cycle = cycle.flatten()
        flat_cycle = flat_cycle[flat_cycle > 0]

        self._stats["msms_range_min"] = flat_cycle.min()
        self._stats["msms_range_max"] = flat_cycle.max()

        self._log_stats()

    def _log_stats(self):
        """Log the statistics calculated from the DIA data."""
        rt_duration_min = self._stats["rt_duration_sec"] / 60

        logger.info(
            f"{'RT (min)':<20}: {self._stats['rt_limits'][0]:.1f} - {self._stats['rt_limits'][1]:.1f}"
        )
        logger.info(f"{'RT duration (sec)':<20}: {self._stats['rt_duration_sec']:.1f}")
        logger.info(f"{'RT duration (min)':<20}: {rt_duration_min:.1f}")

        logger.info(f"{'Cycle len (scans)':<20}: {self._stats['cycle_length']:.0f}")
        logger.info(f"{'Cycle len (sec)':<20}: {self._stats['cycle_duration']:.2f}")
        logger.info(f"{'Number of cycles':<20}: {self._stats['cycle_number']:.0f}")

        logger.info(
            f"{'MS2 range (m/z)':<20}: {self._stats['msms_range_min']:.1f} - {self._stats['msms_range_max']:.1f}"
        )

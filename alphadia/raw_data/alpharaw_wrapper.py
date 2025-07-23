"""Module providing methods to read and process raw data in the following formats: Thermo, Sciex, MzML, AlphaRawBase."""

import logging

import numpy as np
import pandas as pd
from alpharaw.ms_data_base import MSData_Base
from alpharaw.mzml import MzMLReader
from alpharaw.sciex import SciexWiffData
from alpharaw.thermo import ThermoRawData

from alphadia.raw_data.dia_cycle import determine_dia_cycle
from alphadia.raw_data.jitclasses.alpharaw_jit import AlphaRawJIT

logger = logging.getLogger()


def _is_ms1_dia(spectrum_df: pd.DataFrame) -> bool:
    """Check if the MS1 spectra follow a DIA cycle. This check is stricter than just relying on failing to determine a cycle.

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        The spectrum dataframe.

    """
    ms1_df = spectrum_df[spectrum_df["ms_level"] == 1]
    return ms1_df["spec_idx"].diff().value_counts().shape[0] == 1


class AlphaRaw(MSData_Base):
    def __init__(self, centroided: bool = True, save_as_hdf: bool = False):
        """
        Parameters
        ----------
        centroided : bool, optional
            If peaks will be centroided after loading, by default True
        save_as_hdf : bool, optional
            If automatically save the data into HDF5 format, by default False
        """
        super().__init__(centroided, save_as_hdf)

        self.has_mobility = False
        self.has_ms1 = True

    def _process_alpharaw(self, astral_ms1: bool = False):
        self._filter_spectra(astral_ms1)

        self.rt_values = self.spectrum_df.rt.values.astype(np.float32) * 60
        self.zeroth_frame = 0

        if _is_ms1_dia(self.spectrum_df):
            self.cycle, self.cycle_start, self.cycle_length = determine_dia_cycle(
                self.spectrum_df
            )
        else:
            logger.warning(
                "The MS1 spectra in the raw file do not follow a DIA cycle.\n"
                "AlphaDIA will therefore not be able to use the MS1 information.\n"
                "While acquiring data, please make sure to use an integer loop count of 1 or 2 over time based loop count in seconds."
            )

            self.spectrum_df = self.spectrum_df[self.spectrum_df.ms_level > 1]
            self.cycle, self.cycle_start, self.cycle_length = determine_dia_cycle(
                self.spectrum_df
            )
            self.has_ms1 = False

        self.spectrum_df = self.spectrum_df.iloc[self.cycle_start :]
        self.rt_values = self.spectrum_df.rt.values.astype(np.float32) * 60

        self.precursor_cycle_max_index = len(self.rt_values) // self.cycle.shape[1]
        self.mobility_values = np.array([1e-6, 0], dtype=np.float32)

        self.max_mz_value = self.spectrum_df.precursor_mz.max().astype(np.float32)
        self.min_mz_value = self.spectrum_df.precursor_mz.min().astype(np.float32)

        self.quad_max_mz_value = (
            self.spectrum_df[self.spectrum_df["ms_level"] == 2]
            .isolation_upper_mz.max()
            .astype(np.float32)
        )
        self.quad_min_mz_value = (
            self.spectrum_df[self.spectrum_df["ms_level"] == 2]
            .isolation_lower_mz.min()
            .astype(np.float32)
        )

        self.peak_start_idx_list = self.spectrum_df.peak_start_idx.values.astype(
            np.int64
        )
        self.peak_stop_idx_list = self.spectrum_df.peak_stop_idx.values.astype(np.int64)
        self.mz_values = self.peak_df.mz.values.astype(np.float32)
        self.intensity_values = self.peak_df.intensity.values.astype(np.float32)

        self.scan_max_index = 1
        self.frame_max_index = len(self.rt_values) - 1

    def _filter_spectra(self, astral_ms1: bool = False) -> None:
        """Filter the spectra."""

    def to_jitclass(self) -> AlphaRawJIT:
        """Create a AlphaRawJIT with the current state of this class."""
        return AlphaRawJIT(
            self.cycle,
            self.rt_values,
            self.mobility_values,
            self.zeroth_frame,
            self.max_mz_value,
            self.min_mz_value,
            self.quad_max_mz_value,
            self.quad_min_mz_value,
            self.precursor_cycle_max_index,
            self.peak_start_idx_list,
            self.peak_stop_idx_list,
            self.mz_values,
            self.intensity_values,
            self.scan_max_index,
            self.frame_max_index,
        )


class AlphaRawBase(AlphaRaw, MSData_Base):
    def __init__(self, raw_file_path: str):
        super().__init__()
        self.load_hdf(raw_file_path)
        self._process_alpharaw()


class MzML(AlphaRaw, MzMLReader):
    def __init__(self, raw_file_path: str):
        super().__init__()
        self.load_raw(raw_file_path)
        self._process_alpharaw()


class Sciex(AlphaRaw, SciexWiffData):
    def __init__(self, raw_file_path: str):
        super().__init__()
        self.load_raw(raw_file_path)
        self._process_alpharaw()


class Thermo(AlphaRaw, ThermoRawData):
    def __init__(
        self, raw_file_path: str, process_count: int = 10, astral_ms1: bool = False
    ):
        super().__init__(process_count=process_count)
        self.load_raw(raw_file_path)
        self._process_alpharaw(astral_ms1)

    def _filter_spectra(self, astral_ms1: bool = False):
        """Filter the spectra for MS1 or MS2 spectra."""
        # filter for Astral or Orbitrap MS1 spectra
        if astral_ms1:
            self.spectrum_df = self.spectrum_df[self.spectrum_df["nce"] > 0.1]
            self.spectrum_df.loc[self.spectrum_df["nce"] < 1.1, "ms_level"] = 1
            self.spectrum_df.loc[self.spectrum_df["nce"] < 1.1, "precursor_mz"] = -1.0
            self.spectrum_df.loc[
                self.spectrum_df["nce"] < 1.1, "isolation_lower_mz"
            ] = -1.0
            self.spectrum_df.loc[
                self.spectrum_df["nce"] < 1.1, "isolation_upper_mz"
            ] = -1.0
        else:
            self.spectrum_df = self.spectrum_df[
                (self.spectrum_df["nce"] < 0.1) | (self.spectrum_df["nce"] > 1.1)
            ]

        self.spectrum_df["spec_idx"] = np.arange(len(self.spectrum_df))

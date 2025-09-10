"""Module providing methods to read and process raw data in the following formats: Thermo, Sciex, MzML, AlphaRawBase."""

import logging
from abc import ABC

import numpy as np
from alpharaw.ms_data_base import MSData_Base
from alpharaw.mzml import MzMLReader
from alpharaw.sciex import SciexWiffData
from alpharaw.thermo import ThermoRawData

from alphadia.raw_data.dia_cycle import determine_dia_cycle
from alphadia.search.jitclasses.alpharaw_jit import AlphaRawJIT

logger = logging.getLogger()


class AlphaRaw(MSData_Base, ABC):
    def __init__(self, centroided: bool = True, save_as_hdf: bool = False):
        """Abstract class providing data structures and methods for reading and pre-processing raw data.

        Parameters
        ----------
        centroided : bool, optional
            If peaks will be centroided after loading, by default True
        save_as_hdf : bool, optional
            If automatically save the data into HDF5 format, by default False
        """
        super().__init__(centroided, save_as_hdf)

        self.has_mobility: bool = False
        self.has_ms1: bool = True
        self._zeroth_frame: int = 0
        self._scan_max_index: int = 1
        self.mobility_values: np.ndarray[tuple[int], np.dtype[np.float32]] = np.array(
            [1e-6, 0], dtype=np.float32
        )

        self._mz_values: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None
        self.rt_values: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None
        self._intensity_values: np.ndarray[tuple[int], np.dtype[np.float32]] | None = (
            None
        )

        self.cycle: (
            np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]] | None
        ) = None
        self._cycle_start: int | None = None
        self._cycle_length: int | None = None
        self._precursor_cycle_max_index: int | None = None

        self._max_mz_value: np.float32 | None = None
        self._min_mz_value: np.float32 | None = None

        self._quad_max_mz_value: np.ndarray[tuple[int], np.dtype[np.float32]] | None = (
            None
        )
        self._quad_min_mz_value: np.ndarray[tuple[int], np.dtype[np.float32]] | None = (
            None
        )

        self._peak_start_idx_list: np.ndarray[tuple[int], np.dtype[np.int64]] | None = (
            None
        )
        self._peak_stop_idx_list: np.ndarray[tuple[int], np.dtype[np.int64]] | None = (
            None
        )
        self.frame_max_index: int | None = None

    def _preprocess_raw_data(self, astral_ms1: bool = False):
        """Process the raw data to extract relevant information."""
        self._filter_spectra(astral_ms1)

        if not self._is_ms1_dia():
            logger.warning(
                "The MS1 spectra in the raw file do not follow a DIA cycle.\n"
                "AlphaDIA will therefore not be able to use the MS1 information.\n"
                "While acquiring data, please make sure to use an integer loop count of 1 or 2 over time based loop count in seconds."
            )

            self.spectrum_df = self.spectrum_df[self.spectrum_df.ms_level > 1]
            self.has_ms1 = False

        self.cycle, self._cycle_start, self._cycle_length = determine_dia_cycle(
            self.spectrum_df
        )

        self.spectrum_df = self.spectrum_df.iloc[self._cycle_start :]
        self.rt_values = self.spectrum_df.rt.values.astype(np.float32) * 60

        self._precursor_cycle_max_index = len(self.rt_values) // self.cycle.shape[1]

        self._max_mz_value = self.spectrum_df.precursor_mz.max().astype(np.float32)
        self._min_mz_value = self.spectrum_df.precursor_mz.min().astype(np.float32)

        self._quad_max_mz_value = (
            self.spectrum_df[self.spectrum_df["ms_level"] == 2]
            .isolation_upper_mz.max()
            .astype(np.float32)
        )
        self._quad_min_mz_value = (
            self.spectrum_df[self.spectrum_df["ms_level"] == 2]
            .isolation_lower_mz.min()
            .astype(np.float32)
        )

        self._peak_start_idx_list = self.spectrum_df.peak_start_idx.values.astype(
            np.int64
        )
        self._peak_stop_idx_list = self.spectrum_df.peak_stop_idx.values.astype(
            np.int64
        )
        self._mz_values = self.peak_df.mz.values.astype(np.float32)
        self._intensity_values = self.peak_df.intensity.values.astype(np.float32)

        self.frame_max_index = len(self.rt_values) - 1

    def _is_ms1_dia(self) -> bool:
        """Return whether the MS1 spectra follow a DIA cycle."""
        ms1_df = self.spectrum_df[self.spectrum_df["ms_level"] == 1]
        return ms1_df["spec_idx"].diff().value_counts().shape[0] == 1

    def _filter_spectra(self, astral_ms1: bool = False) -> None:
        """Filter the spectra."""

    def to_jitclass(self) -> AlphaRawJIT:
        """Create a AlphaRawJIT with the current state of this class."""
        return AlphaRawJIT(
            self.cycle,
            self.rt_values,
            self.mobility_values,
            self._zeroth_frame,
            self._max_mz_value,
            self._min_mz_value,
            self._quad_max_mz_value,
            self._quad_min_mz_value,
            self._precursor_cycle_max_index,
            self._peak_start_idx_list,
            self._peak_stop_idx_list,
            self._mz_values,
            self._intensity_values,
            self._scan_max_index,
            self.frame_max_index,
        )


class AlphaRawBase(AlphaRaw, MSData_Base):
    """Class holding pre-processed raw data in AlphaBase format."""

    def __init__(self, raw_file_path: str):
        super().__init__()
        self.load_hdf(raw_file_path)
        self._preprocess_raw_data()


class MzML(AlphaRaw, MzMLReader):
    """Class holding pre-processed MzML raw data."""

    def __init__(self, raw_file_path: str):
        super().__init__()
        self.load_raw(raw_file_path)
        self._preprocess_raw_data()


class Sciex(AlphaRaw, SciexWiffData):
    """Class holding pre-processed Sciex raw data."""

    def __init__(self, raw_file_path: str):
        super().__init__()
        self.load_raw(raw_file_path)
        self._preprocess_raw_data()


class Thermo(AlphaRaw, ThermoRawData):
    """Class holding pre-processed Thermo raw data."""

    def __init__(
        self, raw_file_path: str, process_count: int = 10, astral_ms1: bool = False
    ):
        AlphaRaw.__init__(self)
        ThermoRawData.__init__(self, process_count=process_count)
        self.load_raw(raw_file_path)
        self._preprocess_raw_data(astral_ms1)

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

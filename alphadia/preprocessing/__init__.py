"""Preprocess dia data."""

import logging

import alphabase.io

from . import connecting
from . import smoothing
from . import peakfinding
from . import deisotoping
from . import peakstats
from . import msmsgeneration
from . import calibration


class Workflow:

    def run_default(
        self,
    ):
        self.set_connector()
        self.set_smoother()
        self.set_peak_collection()
        self.set_peak_stats_calculator()
        self.set_deisotoper()
        self.set_msms_generator()

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_connector(self):
        connector = connecting.PushConnector(
            self.dia_data,
            # subcycle_tolerance=3,
            # scan_tolerance=6,
        )
        self.connector = connector

    def set_smoother(self):
        self.smoother = smoothing.Smoother()
        self.smoother.set_dia_data(self.dia_data)
        self.smoother.set_connector(self.connector)
        self.smoother.smooth()

    def set_peak_collection(self):
        self.peakfinder = peakfinding.PeakFinder()
        self.peakfinder.set_dia_data(self.dia_data)
        self.peakfinder.set_connector(self.connector)
        self.peakfinder.set_smoother(self.smoother)
        self.peakfinder.find_peaks()

    def set_deisotoper(self):
        self.deisotoper = deisotoping.Deisotoper()
        self.deisotoper.set_dia_data(self.dia_data)
        self.deisotoper.set_connector(self.connector)
        self.deisotoper.set_peak_collection(self.peakfinder.peak_collection)
        self.deisotoper.set_peak_stats_calculator(
            self.peak_stats_calculator
        )
        self.deisotoper.deisotope()

    def set_peak_stats_calculator(self):
        self.peak_stats_calculator = peakstats.PeakStatsCalculator()
        self.peak_stats_calculator.set_dia_data(self.dia_data)
        self.peak_stats_calculator.set_peakfinder(self.peakfinder)
        self.peak_stats_calculator.calculate_stats()

    def set_msms_generator(self):
        self.msms_generator = msmsgeneration.MSMSGenerator()
        self.msms_generator.set_dia_data(self.dia_data)
        self.msms_generator.set_connector(self.connector)
        self.msms_generator.set_peak_collection(
            self.peakfinder.peak_collection
        )
        self.msms_generator.set_deisotoper(self.deisotoper)
        self.msms_generator.set_peak_stats_calculator(
            self.peak_stats_calculator
        )
        self.msms_generator.create_msms_spectra()

    def save_to_hdf(self, file_name=None):
        if file_name is None:
            file_name = f"{self.dia_data.bruker_d_folder_name[:-2]}_preprocess_workflow.hdf"
        logging.info(f"Saving preprocessing workflow results to {file_name}.")
        hdf = alphabase.io.hdf.HDF_File(
            file_name,
            read_only=False,
            truncate=True,
        )
        hdf.connector = self._get_step_as_dict(
            self.connector
        )
        hdf.smoother = self._get_step_as_dict(
            self.smoother
        )
        hdf.peakfinder = self._get_step_as_dict(
            self.peakfinder
        )
        hdf.peakfinder.peak_collection = self._get_step_as_dict(
            self.peakfinder.peak_collection
        )
        hdf.deisotoper = self._get_step_as_dict(
            self.deisotoper
        )
        hdf.peak_stats_calculator = self._get_step_as_dict(
            self.peak_stats_calculator
        )
        hdf.msms_generator = self._get_step_as_dict(
            self.msms_generator
        )

    def _get_step_as_dict(self, step):
        skip_vals = [
            "dia_data",
            "connector",
            "smoother",
            "peak_collection",
            "peakfinder",
            "deisotoper",
            "peak_stats_calculator",
        ]
        return {
            key: val for (
                key,
                val
            ) in step.__dict__.items() if key not in skip_vals
        }

    def load_from_hdf(self):
        hdf = alphabase.io.hdf.HDF_File(
            f"{self.dia_data.bruker_hdf_file_name[:-4]}_preprocess_workflow.hdf",
            read_only=False,
        )
        self.connector = connecting.PushConnector(self.dia_data)
        self.smoother = smoothing.Smoother()
        self.peakfinder = peakfinding.PeakFinder()
        self.deisotoper = deisotoping.Deisotoper()
        self.peak_stats_calculator = peakstats.PeakStatsCalculator()
        self.msms_generator = msmsgeneration.MSMSGenerator()
        self.connector.__dict__ = self._load_from_hdf_dict(
            hdf.connector
        )
        self.smoother.__dict__ = self._load_from_hdf_dict(
            hdf.smoother
        )
        self.peakfinder.__dict__ = self._load_from_hdf_dict(
            hdf.peakfinder
        )
        self.deisotoper.__dict__ = self._load_from_hdf_dict(
            hdf.deisotoper
        )
        self.peak_stats_calculator.__dict__ = self._load_from_hdf_dict(
            hdf.peak_stats_calculator
        )
        self.msms_generator.__dict__ = self._load_from_hdf_dict(
            hdf.msms_generator
        )
        # self.connector.set_dia_data(self.dia_data)
        self.smoother.set_dia_data(self.dia_data)
        self.smoother.set_connector(self.connector)
        self.peakfinder.set_dia_data(self.dia_data)
        self.peakfinder.set_connector(self.connector)
        self.peakfinder.set_smoother(self.smoother)
        self.peakfinder.peak_collection = peakfinding.PeakCollection()
        self.peakfinder.peak_collection.__dict__ = self._load_from_hdf_dict(
            hdf.peakfinder.peak_collection
        )
        self.deisotoper.set_dia_data(self.dia_data)
        self.deisotoper.set_connector(self.connector)
        self.deisotoper.set_peak_collection(self.peakfinder.peak_collection)
        self.peak_stats_calculator.set_dia_data(self.dia_data)
        self.peak_stats_calculator.set_peakfinder(self.peakfinder)
        self.msms_generator.set_dia_data(self.dia_data)
        self.msms_generator.set_peak_collection(self.peakfinder.peak_collection)
        self.msms_generator.set_deisotoper(self.deisotoper)
        self.msms_generator.set_peak_stats_calculator(
            self.peak_stats_calculator
        )

    def _load_from_hdf_dict(self, element):
        select_dict = {}
        for key, val in element.__dict__.items():
            if isinstance(val, alphabase.io.hdf.HDF_Dataset):
                val = val.mmap
            select_dict[key] = val
        return select_dict

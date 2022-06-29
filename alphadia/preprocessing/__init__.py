"""Preprocess dia data."""

from . import connecting
from . import smoothing
from . import peakfinding
from . import deisotoping


class Workflow:

    def run_default(
        self,
    ):
        self.set_connector()
        self.set_smoother()
        self.set_peak_collection()
        self.set_deisotoper()
        # self.set_peak_stats()

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_connector(self):
        self.connector = connecting.Connector()
        self.connector.set_dia_data(self.dia_data)
        self.connector.connect()

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
        self.deisotoper.deisotope()

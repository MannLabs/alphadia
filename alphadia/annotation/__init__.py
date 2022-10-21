"""Identify pseudo MSMS data data."""

from . import identification
from . import psm_stats
from . import library
from . import percolation


class Annotator:

    def set_ions(self, precursor_df, fragment_df):
        # self.preprocessing_workflow = preprocessing_workflow
        self.precursor_df = precursor_df
        self.fragment_df = fragment_df

    def set_library(self, library):
        self.library = library

    def set_msms_identifier(self):
        self.msms_identifier = identification.MSMSIdentifier()
        # self.msms_identifier.set_preprocessor(self.preprocessing_workflow)
        self.msms_identifier.set_ions(
            self.precursor_df,
            self.fragment_df,
        )
        self.msms_identifier.set_library(self.library)
        self.msms_identifier.identify()

    def set_psm_stats_calculator(self):
        self.psm_stats_calculator = psm_stats.PSMStatsCalculator()
        # self.psm_stats_calculator.set_preprocessor(self.preprocessing_workflow)
        self.psm_stats_calculator.set_ions(self.precursor_df, self.fragment_df)
        self.psm_stats_calculator.set_library(self.library)
        self.psm_stats_calculator.set_annotation(
            self.msms_identifier.annotation
        )
        self.psm_stats_calculator.estimate_mz_tolerance()

    def set_percolator(self):
        self.percolator = percolation.Percolator()
        self.percolator.set_annotation(
            self.psm_stats_calculator.annotation
        )
        self.percolator.percolate()

    def run_default(self):
        self.set_msms_identifier()
        self.set_psm_stats_calculator()
        self.msms_identifier.update_ppm_values_from_stats_calculator(
            self.psm_stats_calculator
        )
        self.msms_identifier.identify()
        self.psm_stats_calculator.set_annotation(
            self.msms_identifier.annotation
        )
        self.psm_stats_calculator.update_annotation_stats()
        self.set_percolator()

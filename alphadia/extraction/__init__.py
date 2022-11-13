"""Preprocess dia data."""

import logging

import alphabase.io.hdf
from . import elution_group_assembly

from . import calibration
from . import planning

class Workflow:

    def run_default(
        self
    ):
        self.set_elution_group_assembler()
        
    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_library(self, library):
        self.library = library

    def set_elution_group_assembler(self, **kwargs):
        self.elution_group_assembler = elution_group_assembly.ElutionGroupAssembler( **kwargs)
        self.elution_group_assembler.set_dia_data(self.dia_data)
        self.elution_group_assembler.set_library(self.library)
        self.elution_group_assembler.run()

    def save_to_hdf(self, file_name=None):
        if file_name is None:
            file_name = f"{self.dia_data.bruker_hdf_file_name[:-4]}_extraction_workflow.hdf"
        logging.info(f"Saving preprocessing workflow results to {file_name}.")
        hdf = alphabase.io.hdf.HDF_File(
            file_name,
            read_only=False,
            truncate=True,
        )
        hdf.elution_group_assembler = self._get_step_as_dict(
            self.elution_group_assembler
        )

    def _get_step_as_dict(self, step):
        skip_vals = [
            "dia_data",
            "library"
        ]
        return {
            key: val for (
                key,
                val
            ) in step.__dict__.items() if key not in skip_vals
        }

    def load_from_hdf(self, file_name=None):
        if file_name is None:
            file_name = f"{self.dia_data.bruker_hdf_file_name[:-4]}_extraction_workflow.hdf",
        logging.info(f"Loading preprocessing workflow from {file_name}.")
        hdf = alphabase.io.hdf.HDF_File(
            file_name,  # type: ignore
            read_only=False,
        )

    def _load_from_hdf_dict(self, element):
        select_dict = {}
        for key, val in element.__dict__.items():
            if isinstance(val, alphabase.io.hdf.HDF_Dataset):
                val = val.mmap
            select_dict[key] = val
        return select_dict

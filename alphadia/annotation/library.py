"""Import library."""

import logging

import numpy as np

import alphabase.io.hdf


class Library:

    def import_from_file(self, library_file_name):
        logging.info("Loading library")
        self.library_file_name = library_file_name
        self.lib = alphabase.io.hdf.HDF_File(
            self.library_file_name
        #     read_only=False
        )

        predicted_library_df = self.lib.library.precursor_df[...]
        # predicted_library_df.sort_values(by=["rt_pred", "mobility_pred"], inplace=True)
        predicted_library_df.sort_values(by="precursor_mz", inplace=True)
        predicted_library_df.reset_index(level=0, inplace=True)
        predicted_library_df.rename(columns={"index": "original_index"}, inplace=True)
        predicted_library_df.decoy = predicted_library_df.decoy.astype(np.bool_)

        self.y_mzs = self.lib.library.fragment_mz_df.y_z1.mmap
        self.b_mzs = self.lib.library.fragment_mz_df.b_z1.mmap
        self.y_ions_intensities = self.lib.library.fragment_intensity_df.y_z1.mmap
        self.b_ions_intensities = self.lib.library.fragment_intensity_df.b_z1.mmap

        self.predicted_library_df = predicted_library_df

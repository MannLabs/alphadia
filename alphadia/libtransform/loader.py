import logging
import os
from pathlib import Path

from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.reader import LibraryReaderBase

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()


class DynamicLoader(ProcessingStep):
    def __init__(self, modification_mapping: dict | None = None) -> None:
        """Load a spectral library from a file. The file type is dynamically inferred from the file ending.
        Expects a `str` as input and will return a `SpecLibBase` object.

        Supported file types are:

        **Alphabase hdf5 files**
        The library is loaded into a `SpecLibBase` object and immediately returned.

        **Long format csv files**
        The classical spectral library format as returned by MSFragger.
        It will be imported and converted to a `SpecLibBase` format. This might require additional parsing information.
        """
        if modification_mapping is None:
            modification_mapping = {}
        self.modification_mapping = modification_mapping

    def validate(self, input: str) -> bool:
        """Validate the input object. It is expected that the input is a path to a file which exists."""
        valid = True
        valid &= isinstance(input, str | Path)

        if not os.path.exists(input):
            logger.error(f"Input path {input} does not exist")
            valid = False

        return valid

    def forward(self, input_path: str) -> SpecLibBase:
        """Load the spectral library from the input path. The file type is dynamically inferred from the file ending."""
        # get ending of file
        file_type = Path(input_path).suffix

        if file_type in [".hdf5", ".h5", ".hdf"]:
            logger.info(f"Loading {file_type} library from {input_path}")
            library = SpecLibBase()
            library.load_hdf(input_path, load_mod_seq=True)

        elif file_type in [".csv", ".tsv"]:
            logger.info(f"Loading {file_type} library from {input_path}")
            library = LibraryReaderBase()
            library.add_modification_mapping(self.modification_mapping)
            library.import_file(input_path)

        else:
            raise ValueError(f"File type {file_type} not supported")

        # TODO: this is a hack to get the charged_frag_types from the fragment_mz_df
        # this should be fixed ASAP in alphabase
        library.charged_frag_types = library.fragment_mz_df.columns.tolist()

        return library

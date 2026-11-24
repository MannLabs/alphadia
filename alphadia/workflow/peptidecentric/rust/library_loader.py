"""Load a flat spectral library from HDF with alphadia-search-rs, for the Rust extraction backend.

The search reads only the numeric columns. The string columns (sequence, modifications, proteins) are read
at the end of the search of a raw file, only for the precursors of its PSMs.
"""

import logging
import time

import h5py
import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import SpecLibFlat as SpecLibFlatRust
from alphadia_search_rs import read_hdf_columns

logger = logging.getLogger()

PRECURSOR_GROUP = "library/precursor_df"
FRAGMENT_GROUP = "library/fragment_df"

# String columns that are added to the PSMs.
STRING_PRECURSOR_COLUMNS = ["sequence", "mods", "mod_sites", "proteins", "genes"]
# Columns that are added to the PSMs from the library file: the string columns and the precursor key.
PSM_LIBRARY_COLUMNS = [*STRING_PRECURSOR_COLUMNS, "mod_seq_charge_hash"]
# String columns that are not read.
SKIPPED_PRECURSOR_COLUMNS = ["instrument", "protein_idxes"]


class LibraryStringColumns:
    """Read the string columns of the precursors of a library file, only for given precursors."""

    def __init__(self, library_path: str, precursor_idx: np.ndarray) -> None:
        self._library_path = library_path
        # the precursor_idx of each row of the file
        self._precursor_idx = precursor_idx
        self._is_row_number = np.array_equal(
            precursor_idx, np.arange(len(precursor_idx))
        )
        if not self._is_row_number and np.any(np.diff(precursor_idx) <= 0):
            raise ValueError("The precursor_idx of the library file is not sorted")

    def add_to(
        self, df: pd.DataFrame, columns: list[str] = PSM_LIBRARY_COLUMNS
    ) -> pd.DataFrame:
        """Add `columns` of the precursors of `df` to `df`, matched by precursor_idx."""
        missing = [c for c in columns if c not in df.columns]
        if not missing:
            return df
        precursor_idx = df["precursor_idx"].to_numpy(dtype=np.int64)
        unique_idx, inverse = np.unique(precursor_idx, return_inverse=True)
        if self._is_row_number:
            rows = unique_idx
        else:
            rows = np.searchsorted(self._precursor_idx, unique_idx)
        if len(rows) and (
            rows[-1] >= len(self._precursor_idx)
            or not np.array_equal(self._precursor_idx[rows], unique_idx)
        ):
            raise ValueError("Precursors of the PSMs are not in the library file")
        values = read_hdf_columns(
            self._library_path, PRECURSOR_GROUP, missing, rows.astype(np.uint64)
        )
        df = df.copy(deep=False)
        for column in missing:
            df[column] = values[column][inverse]
        return df


def load_flat_library(
    library_path: str,
) -> tuple[SpecLibFlat, SpecLibFlatRust, LibraryStringColumns]:
    """Load the flat library in `library_path`.

    Returns the alphabase library without the string columns, the alphadia-search-rs library
    and the reader of the string columns.
    """
    time_start = time.time()
    with h5py.File(library_path, "r") as hdf:
        column_names = list(hdf[PRECURSOR_GROUP].keys())
    not_read = set(STRING_PRECURSOR_COLUMNS + SKIPPED_PRECURSOR_COLUMNS)
    precursor_df = pd.DataFrame(
        read_hdf_columns(
            library_path,
            PRECURSOR_GROUP,
            [c for c in column_names if c not in not_read],
        ),
        copy=False,
    )
    fragment_df = pd.DataFrame(
        read_hdf_columns(library_path, FRAGMENT_GROUP), copy=False
    )
    logger.info(
        f"Reading the numeric library columns took {time.time() - time_start:.1f} s"
    )

    time_start = time.time()
    spectral_library_rust = SpecLibFlatRust.from_hdf(library_path)
    logger.info(
        f"Reading the alphadia-search-rs library took {time.time() - time_start:.1f} s"
    )

    string_columns = LibraryStringColumns(
        library_path, precursor_df["precursor_idx"].to_numpy()
    )

    spectral_library = SpecLibFlat()
    spectral_library._precursor_df = precursor_df
    spectral_library._fragment_df = fragment_df
    spectral_library._protein_df = pd.DataFrame()
    return spectral_library, spectral_library_rust, string_columns

"""Load a flat spectral library from HDF with alphadia-search-rs, for the Rust extraction backend.

The search reads only the numeric columns. The PSMs of each raw file keep the numeric key mod_seq_charge_hash.
The string columns (sequence, modifications, proteins) are read once for the output, only for its precursors.
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

# String columns that are added to the precursor table of the search output, by mod_seq_charge_hash.
STRING_PRECURSOR_COLUMNS = ["sequence", "mods", "mod_sites", "proteins", "genes"]
# String columns that are not read.
SKIPPED_PRECURSOR_COLUMNS = ["instrument", "protein_idxes"]
# The key of a precursor in the PSMs of a raw file, together with `decoy`: a decoy has the hash of its target.
# The string columns are added to the output by this key.
PRECURSOR_KEY = "mod_seq_charge_hash"


class LibraryStrings:
    """The string columns of the precursors of a library file, read only for the precursors of the output.

    The search and the PSMs of each raw file keep only the numeric key `mod_seq_charge_hash`.
    A decoy has the hash of its target, so a precursor is identified by the hash and `decoy`.
    The precursors with the same hash and decoy have the same sequence, modifications, charge and proteins.
    """

    def __init__(
        self,
        library_path: str,
        precursor_idx: np.ndarray,
        precursor_key: np.ndarray,
        decoy: np.ndarray,
    ) -> None:
        """`precursor_idx`, `precursor_key` and `decoy` are the columns of the library file, one value per row."""
        self._library_path = library_path
        self._precursor_idx = precursor_idx
        self._precursor_key = precursor_key
        self._decoy = decoy
        self._is_row_number = np.array_equal(
            precursor_idx, np.arange(len(precursor_idx))
        )
        if not self._is_row_number and np.any(np.diff(precursor_idx) <= 0):
            raise ValueError("The precursor_idx of the library file is not sorted")
        # for each decoy value: the rows of the library file with this value, sorted by key
        self._rows_by_key: dict[int, np.ndarray] = {}

    def add_key(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """Add `mod_seq_charge_hash` to the PSMs of a raw file, matched by precursor_idx."""
        if PRECURSOR_KEY in psm_df.columns:
            return psm_df
        precursor_idx = psm_df["precursor_idx"].to_numpy(dtype=np.int64)
        rows = (
            precursor_idx
            if self._is_row_number
            else np.searchsorted(self._precursor_idx, precursor_idx)
        )
        if len(rows) and (
            rows.max() >= len(self._precursor_idx)
            or not np.array_equal(self._precursor_idx[rows], precursor_idx)
        ):
            raise ValueError("Precursors of the PSMs are not in the library file")
        psm_df = psm_df.copy(deep=False)
        psm_df[PRECURSOR_KEY] = self._precursor_key[rows]
        return psm_df

    def add_strings(
        self, df: pd.DataFrame, columns: list[str] = STRING_PRECURSOR_COLUMNS
    ) -> pd.DataFrame:
        """Add the string `columns` to `df`, matched by mod_seq_charge_hash."""
        missing = [c for c in columns if c not in df.columns]
        if not missing:
            return df
        time_start = time.time()
        keys = df[PRECURSOR_KEY].to_numpy(dtype=np.uint64)
        decoys = df["decoy"].to_numpy()
        # the row of the library file of each row of df
        df_rows = np.empty(len(df), dtype=np.int64)
        for decoy in np.unique(decoys):
            if decoy not in self._rows_by_key:
                rows = np.flatnonzero(self._decoy == decoy)
                self._rows_by_key[decoy] = rows[
                    np.argsort(self._precursor_key[rows], kind="stable")
                ]
            library_rows = self._rows_by_key[decoy]
            sorted_keys = self._precursor_key[library_rows]
            is_decoy = decoys == decoy
            positions = np.searchsorted(sorted_keys, keys[is_decoy])
            if len(positions) and (
                positions.max() >= len(sorted_keys)
                or not np.array_equal(sorted_keys[positions], keys[is_decoy])
            ):
                raise ValueError(
                    "Precursor keys of the output are not in the library file"
                )
            df_rows[is_decoy] = library_rows[positions]
        # rows in file order make the HDF read faster
        rows, inverse = np.unique(df_rows, return_inverse=True)
        values = read_hdf_columns(
            self._library_path, PRECURSOR_GROUP, missing, rows.astype(np.uint64)
        )
        df = df.copy(deep=False)
        for column in missing:
            df[column] = values[column][inverse]
        logger.info(
            f"Reading the library string columns of {len(rows):,} precursors took {time.time() - time_start:.1f} s"
        )
        return df


def load_flat_library(
    library_path: str,
) -> tuple[SpecLibFlat, SpecLibFlatRust, LibraryStrings]:
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

    library_strings = LibraryStrings(
        library_path,
        precursor_df["precursor_idx"].to_numpy(),
        precursor_df[PRECURSOR_KEY].to_numpy(),
        precursor_df["decoy"].to_numpy(),
    )

    spectral_library = SpecLibFlat()
    spectral_library._precursor_df = precursor_df
    spectral_library._fragment_df = fragment_df
    spectral_library._protein_df = pd.DataFrame()
    return spectral_library, spectral_library_rust, library_strings

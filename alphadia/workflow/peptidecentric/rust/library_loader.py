"""Load a flat spectral library from HDF with alphadia-search-rs, for the Rust extraction backend.

The numeric columns and the Rust library are read at once. The string columns (sequence, modifications, proteins)
are read in a background thread, because the search needs them only for the final extraction.
"""

import logging
import threading
import time

import h5py
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import SpecLibFlat as SpecLibFlatRust
from alphadia_search_rs import read_hdf_columns

logger = logging.getLogger()

PRECURSOR_GROUP = "library/precursor_df"
FRAGMENT_GROUP = "library/fragment_df"

# String columns that the search reads only for the final extraction and the output.
DEFERRED_PRECURSOR_COLUMNS = ["sequence", "mods", "mod_sites", "proteins", "genes"]
# String columns that the search does not use.
SKIPPED_PRECURSOR_COLUMNS = ["instrument", "protein_idxes"]


class DeferredPrecursorColumns:
    """Read precursor columns of a library file in a background thread.

    The HDF read in alphadia-search-rs releases the GIL, so the search continues while the thread reads.
    """

    def __init__(self, library_path: str, columns: list[str]) -> None:
        self._library_path = library_path
        self._columns = columns
        self._result: pd.DataFrame | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._read, name="deferred-precursor-columns", daemon=True
        )
        self._thread.start()

    def _read(self) -> None:
        try:
            time_start = time.time()
            columns = read_hdf_columns(
                self._library_path, PRECURSOR_GROUP, ["precursor_idx", *self._columns]
            )
            precursor_idx = columns.pop("precursor_idx")
            self._result = pd.DataFrame(
                columns, index=pd.Index(precursor_idx, name="precursor_idx"), copy=False
            )
            logger.info(
                f"Reading the precursor columns {self._columns} took {time.time() - time_start:.1f} s"
            )
        except BaseException as e:  # noqa: BLE001 the error is raised again in `get`
            self._error = e

    def get(self) -> pd.DataFrame:
        """Wait for the read and return the columns, indexed by precursor_idx."""
        time_start = time.time()
        self._thread.join()
        waited = time.time() - time_start
        if waited > 0.1:
            logger.info(f"Waited {waited:.1f} s for the deferred precursor columns")
        if self._error is not None:
            raise self._error
        return self._result


def add_deferred_columns(
    precursor_df: pd.DataFrame, deferred_df: pd.DataFrame
) -> pd.DataFrame:
    """Add the columns of `deferred_df` that `precursor_df` does not have, matched by precursor_idx."""
    missing = [c for c in deferred_df.columns if c not in precursor_df.columns]
    if not missing:
        return precursor_df
    values = deferred_df[missing].reindex(precursor_df["precursor_idx"].to_numpy())
    precursor_df = precursor_df.copy(deep=False)
    for column in missing:
        precursor_df[column] = values[column].to_numpy()
    return precursor_df


def load_flat_library(
    library_path: str,
) -> tuple[SpecLibFlat, SpecLibFlatRust, DeferredPrecursorColumns]:
    """Load the flat library in `library_path`.

    Returns the alphabase library without the deferred string columns, the alphadia-search-rs library
    and the reader of the deferred columns.
    """
    time_start = time.time()
    with h5py.File(library_path, "r") as hdf:
        column_names = list(hdf[PRECURSOR_GROUP].keys())
    not_read = set(DEFERRED_PRECURSOR_COLUMNS + SKIPPED_PRECURSOR_COLUMNS)
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

    # start after the other reads: the HDF5 library serialises reads of all threads
    deferred = DeferredPrecursorColumns(library_path, DEFERRED_PRECURSOR_COLUMNS)

    spectral_library = SpecLibFlat()
    spectral_library._precursor_df = precursor_df
    spectral_library._fragment_df = fragment_df
    spectral_library._protein_df = pd.DataFrame()
    return spectral_library, spectral_library_rust, deferred

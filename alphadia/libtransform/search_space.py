"""Derive the search space of a spectral library from its target precursors.

If a spectral library is provided, the in-silico digest settings in the config (`library_prediction.precursor_len`, ...)
are not used. To still document the actual search space (e.g. for ProteoBench, which parses the log),
it is derived from the target precursors of the library and logged.

All quantities are computed column-wise with vectorized pandas/numpy operations (no per-precursor Python code),
as libraries can have tens of millions of precursors.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()

# precursor m/z column candidates: `precursor_mz` in alphabase libraries, renamed to `mz_library` by `InitFlatColumns`
PRECURSOR_MZ_COLUMNS = ("precursor_mz", CalibCols.MZ_LIBRARY, "mz")

# columns that define a modified peptide. `mod_sites` is included so that positional isomers
# (same modifications on different residues) count as different modified peptides.
MODIFIED_PEPTIDE_COLUMNS = ("sequence", "mods", "mod_sites")

# The `proteins` entry of a precursor is the ';'-joined list of proteins its peptide maps to. The search space counts
# the distinct individual proteins over all entries (no protein grouping or parsimony; those happen after the search).
PROTEIN_COLUMN = "proteins"
PROTEIN_DELIMITER = ";"

MODIFICATIONS_SEPARATOR = ";"


@dataclass
class LibrarySearchSpace:
    """Search space of a spectral library, derived from its target precursors.

    Ranges are None if the respective column is missing from the library.
    """

    peptide_len: tuple[int, int] | None
    precursor_charge: tuple[int, int] | None
    precursor_mz: tuple[int, int] | None
    modifications: list[str] = field(default_factory=list)
    n_modified_peptides: int = 0
    n_precursors: int = 0
    n_proteins: int | None = None


def _min_max(values: np.ndarray) -> tuple[float, float] | None:
    """Get (min, max) of an array, None if it is empty."""
    if len(values) == 0:
        return None
    return values.min(), values.max()


def _count_proteins(protein_entries: np.ndarray | pd.Series) -> int:
    """Number of distinct proteins in ';'-joined protein entries; only the distinct entries are split."""
    entries = pd.unique(pd.Series(protein_entries).dropna())
    proteins = (
        pd.Series(entries, dtype=object)
        .str.split(PROTEIN_DELIMITER)
        .explode()
        .str.strip()
    )
    return int(proteins[proteins.astype(bool)].nunique())


def _int_min_max(values: np.ndarray) -> tuple[int, int] | None:
    min_max = _min_max(values)
    return None if min_max is None else (int(min_max[0]), int(min_max[1]))


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get an integer code per value and the distinct values, with missing values treated as ''.

    Missing values occur e.g. as `mods` and `mod_sites` of unmodified peptides, which are '' otherwise.
    """
    codes, uniques = pd.factorize(values)
    missing = codes < 0
    if missing.any():
        empty_idx = np.flatnonzero(uniques == "")
        if len(empty_idx):
            codes[missing] = empty_idx[0]
        else:
            codes[missing] = len(uniques)
            uniques = np.append(uniques, "")
    return codes.astype(np.int64, copy=False), uniques


def _combine_codes(
    codes: np.ndarray, n_codes: int, other_codes: np.ndarray, n_other: int
) -> tuple[np.ndarray, int]:
    """Combine codes in [0, n_codes) and [0, n_other) into one code per row that is unique per pair of codes.

    Returns the combined codes and their upper bound. Integer arithmetic only, so no strings are concatenated.
    """
    if n_codes * n_other >= 2**62:
        # make the codes dense first to avoid an int64 overflow
        codes, uniques = pd.factorize(codes)
        n_codes = len(uniques)
    return codes * n_other + other_codes, n_codes * n_other


def _get_target_values(
    precursor_df: pd.DataFrame, column: str, target_mask: np.ndarray | None
) -> np.ndarray:
    """Get the values of `column` for the target precursors, copying only this column."""
    values = precursor_df[column].to_numpy()
    return values if target_mask is None else values[target_mask]


def _get_peptide_len(
    precursor_df: pd.DataFrame, target_mask: np.ndarray | None
) -> tuple[int, int] | None:
    """Get the peptide length range, from `nAA` if present, else from the sequences."""
    if "nAA" in precursor_df.columns:
        lengths = _get_target_values(precursor_df, "nAA", target_mask)
    else:
        sequences = _get_target_values(precursor_df, "sequence", target_mask)
        lengths = pd.Series(sequences, dtype=object).str.len().to_numpy()
    return _int_min_max(lengths)


def _get_precursor_mz(
    precursor_df: pd.DataFrame, target_mask: np.ndarray | None
) -> tuple[int, int] | None:
    """Get the precursor m/z range, rounded outwards to integers."""
    mz_column = next(
        (c for c in PRECURSOR_MZ_COLUMNS if c in precursor_df.columns), None
    )
    if mz_column is None:
        return None
    mz_min_max = _min_max(_get_target_values(precursor_df, mz_column, target_mask))
    if mz_min_max is None:
        return None
    return math.floor(mz_min_max[0]), math.ceil(mz_min_max[1])


def _get_modified_peptides(
    precursor_df: pd.DataFrame, target_mask: np.ndarray | None
) -> tuple[np.ndarray, int, list[str]]:
    """Get an integer code per target precursor that identifies its modified peptide, and the distinct modifications.

    The columns of `MODIFIED_PEPTIDE_COLUMNS` are factorized once each and their codes combined. The distinct
    modifications are taken from the factorized `mods` column, so only its distinct ';'-joined strings are split.

    Returns
    -------
    tuple[np.ndarray, int, list[str]]
        Codes of the modified peptides, upper bound of the codes (exclusive), sorted distinct modifications.
    """
    n_targets = len(precursor_df) if target_mask is None else int(target_mask.sum())
    peptide_codes, n_peptide_codes = np.zeros(n_targets, dtype=np.int64), 1
    modifications = []
    for column in MODIFIED_PEPTIDE_COLUMNS:
        if column not in precursor_df.columns:
            continue
        column_codes, uniques = _factorize(
            _get_target_values(precursor_df, column, target_mask)
        )
        peptide_codes, n_peptide_codes = _combine_codes(
            peptide_codes, n_peptide_codes, column_codes, len(uniques)
        )
        if column == "mods":
            mods = MODIFICATIONS_SEPARATOR.join(uniques).split(MODIFICATIONS_SEPARATOR)
            modifications = sorted(set(mods) - {""})
    return peptide_codes, n_peptide_codes, modifications


def _count_precursors(
    peptide_codes: np.ndarray, n_peptide_codes: int, charges: np.ndarray | None
) -> int:
    """Count the distinct (modified peptide, charge) combinations."""
    if charges is None or len(charges) == 0:
        return len(np.unique(peptide_codes))
    precursor_codes, _ = _combine_codes(
        peptide_codes, n_peptide_codes, charges, int(charges.max()) + 1
    )
    return len(np.unique(precursor_codes))


def get_library_search_space(precursor_df: pd.DataFrame) -> LibrarySearchSpace:
    """Derive the search space from the target precursors of a spectral library.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe with at least a `sequence` column. Optional columns: `decoy` (decoys are excluded),
        `nAA` (used for the length if present), `mods`, `mod_sites`, `charge`, precursor m/z (`precursor_mz` or
        `mz_library`), `proteins`.

    Returns
    -------
    LibrarySearchSpace
        The derived search space.
    """
    columns = precursor_df.columns
    target_mask = precursor_df["decoy"].to_numpy() == 0 if "decoy" in columns else None

    charges = (
        _get_target_values(precursor_df, "charge", target_mask).astype(np.int64)
        if "charge" in columns
        else None
    )
    peptide_codes, n_peptide_codes, modifications = _get_modified_peptides(
        precursor_df, target_mask
    )

    return LibrarySearchSpace(
        peptide_len=_get_peptide_len(precursor_df, target_mask),
        precursor_charge=None if charges is None else _int_min_max(charges),
        precursor_mz=_get_precursor_mz(precursor_df, target_mask),
        modifications=modifications,
        n_modified_peptides=len(np.unique(peptide_codes)),
        n_precursors=_count_precursors(peptide_codes, n_peptide_codes, charges),
        n_proteins=(
            _count_proteins(
                _get_target_values(precursor_df, PROTEIN_COLUMN, target_mask)
            )
            if PROTEIN_COLUMN in columns
            else None
        ),
    )


def log_library_search_space(search_space: LibrarySearchSpace) -> None:
    """Log the search space; ranges use the keys and format of the config tree print (`library_prediction` section).

    The log is parsed by external tools (e.g. ProteoBench), so keys and format should be kept stable.
    Free-text lines must not contain a colon or any of the keys.
    """
    logger.info("============ Library search space ============")
    logger.info("Derived from the target precursors of the spectral library")

    for key, min_max in [
        ("peptide_len", search_space.peptide_len),
        ("precursor_charge", search_space.precursor_charge),
        ("precursor_mz", search_space.precursor_mz),
    ]:
        if min_max is None:
            continue
        logger.info(f"{key}:")
        for value in min_max:
            logger.info(f"    - {value}")

    logger.info(
        f"modifications: {MODIFICATIONS_SEPARATOR.join(search_space.modifications)}"
    )
    logger.info(f"n_modified_peptides: {search_space.n_modified_peptides}")
    logger.info(f"n_precursors: {search_space.n_precursors}")
    if search_space.n_proteins is not None:
        logger.info(f"n_proteins: {search_space.n_proteins}")


class LogLibrarySearchSpace(ProcessingStep):
    def __init__(self) -> None:
        """Log the search space (ranges, modifications and counts) derived from the target precursors of a library."""
        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` (or `SpecLibFlat`) object."""
        return isinstance(input, SpecLibBase) and "sequence" in input.precursor_df

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Derive and log the search space, return the input unchanged."""
        log_library_search_space(get_library_search_space(input.precursor_df))
        return input

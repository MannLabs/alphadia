"""Derive the search space (digestion and modification settings) of a spectral library from its target precursors.

If a spectral library is provided, the in-silico digest settings in the config (`library_prediction.enzyme`, ...)
are not used. To still document the actual search space (e.g. for ProteoBench, which parses the log),
it is inferred from the peptide sequences of the library and logged using the same keys as the config.
"""

import logging
import math
import re
from dataclasses import dataclass, field

import pandas as pd
from alphabase.protein.fasta import protease_dict
from alphabase.spectral_library.base import SpecLibBase

from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()

# Candidate enzymes and the residues they cleave after (C-terminal), ordered from most to least specific.
# Keys are names in alphabase's `protease_dict` (and valid values for `library_prediction.enzyme`).
ENZYME_C_TERM_RESIDUES = {
    "lys-c": "K",
    "arg-c": "R",
    "glu-c": "E",
    "trypsin": "KR",
    "chymotrypsin": "FLYWMH",
}

# minimum share of target peptides that need to end with a cleavage residue of an enzyme to infer this enzyme
MIN_ENZYME_SPECIFICITY = 0.9

UNKNOWN_ENZYME = "unknown"

# precursor m/z column candidates: `precursor_mz` in alphabase libraries, renamed to `mz_library` by `InitFlatColumns`
PRECURSOR_MZ_COLUMNS = ("precursor_mz", CalibCols.MZ_LIBRARY, "mz")

N_TERM = "N-term"
C_TERM = "C-term"


@dataclass
class LibrarySearchSpace:
    """Search space of a spectral library, derived from its target precursors."""

    n_precursors: int
    n_peptides: int
    precursor_len: tuple[int, int]
    precursor_charge: tuple[int, int] | None
    precursor_mz: tuple[int, int] | None
    enzyme: str
    enzyme_specificity: float
    missed_cleavages: int | None
    missed_cleavages_distribution: dict[int, int] = field(default_factory=dict)
    fixed_modifications: list[str] = field(default_factory=list)
    variable_modifications: list[str] = field(default_factory=list)
    max_var_mod_num: int = 0


def _infer_enzyme(sequences: pd.Series) -> tuple[str, float]:
    """Infer the enzyme from the C-terminal residues of the peptides.

    Returns the most specific enzyme for which at least `MIN_ENZYME_SPECIFICITY` of the peptides end with one of its
    cleavage residues, and the share of peptides that do so. Note that protein C-terminal peptides do not need to end
    with a cleavage residue, so the share is slightly below 1 also for a perfectly specific digest.
    """
    c_term_residues = sequences.str[-1]
    for enzyme, residues in ENZYME_C_TERM_RESIDUES.items():
        specificity = float(c_term_residues.isin(list(residues)).mean())
        if specificity >= MIN_ENZYME_SPECIFICITY:
            return enzyme, specificity

    trypsin_specificity = float(
        c_term_residues.isin(list(ENZYME_C_TERM_RESIDUES["trypsin"])).mean()
    )
    return UNKNOWN_ENZYME, trypsin_specificity


def _count_missed_cleavages(sequences: pd.Series, enzyme: str) -> pd.Series:
    """Count the internal cleavage sites of each peptide, using the cleavage rule of alphabase's `protease_dict`."""
    regex = re.compile(protease_dict[enzyme])

    def _count(sequence: str) -> int:
        return sum(
            1 for match in regex.finditer(sequence) if match.start() + 1 < len(sequence)
        )

    return sequences.map(_count)


def _get_eligible_count(site: str, sequences: pd.Series) -> int:
    """Get the number of positions in `sequences` that could carry a modification on `site`.

    `site` is the part of the alphabase modification name after the '@', e.g. 'C', 'Any_N-term', 'Q^Any_N-term'.
    Terminal modifications are not distinguished by 'Any' and 'Protein', i.e. a 'Protein_N-term' modification
    is eligible on all peptide N-termini.
    """
    residue, _, terminus = site.rpartition("^") if "^" in site else ("", "", site)
    if terminus.endswith(N_TERM):
        if residue:
            return int(sequences.str.startswith(residue).sum())
        return len(sequences)
    if terminus.endswith(C_TERM):
        if residue:
            return int(sequences.str.endswith(residue).sum())
        return len(sequences)
    # residue-specific modification
    return int(sequences.str.count(re.escape(terminus)).sum())


def _get_modifications(peptide_df: pd.DataFrame) -> tuple[list[str], list[str], int]:
    """Classify the modifications of the peptides as fixed or variable and get the max. number of variable modifications.

    A modification is considered fixed if every eligible residue (or terminus) carries it, else variable.
    The max. number of variable modifications counts the variable modifications on one peptide, terminal ones included
    and fixed ones never. Note that alphabase's `max_var_mod_num` leaves out terminal modifications, so this can be one
    higher than the setting the library was built with (e.g. Acetyl@Protein_N-term on top of one Oxidation@M).
    """
    if "mods" not in peptide_df.columns:
        return [], [], 0

    mods_per_peptide = peptide_df["mods"].fillna("").str.split(";")
    all_mods = mods_per_peptide.explode()
    mod_counts = all_mods[all_mods != ""].value_counts()

    fixed_mods, variable_mods = [], []
    for mod, n_occupied in mod_counts.items():
        site = mod.rsplit("@", 1)[-1]
        n_eligible = _get_eligible_count(site, peptide_df["sequence"])
        if n_occupied >= n_eligible > 0:
            fixed_mods.append(mod)
        else:
            variable_mods.append(mod)

    variable_mods_set = set(variable_mods)
    n_var_mods_per_peptide = mods_per_peptide.map(
        lambda mods: sum(1 for mod in mods if mod in variable_mods_set)
    )
    max_var_mod_num = (
        int(n_var_mods_per_peptide.max()) if len(n_var_mods_per_peptide) else 0
    )

    return sorted(fixed_mods), sorted(variable_mods), max_var_mod_num


def get_library_search_space(precursor_df: pd.DataFrame) -> LibrarySearchSpace:
    """Derive the search space from the target precursors of a spectral library.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe with at least a `sequence` column. Optional columns: `decoy` (decoys are excluded),
        `mods`, `mod_sites`, `charge`, precursor m/z (`precursor_mz` or `mz_library`).

    Returns
    -------
    LibrarySearchSpace
        The derived search space.
    """
    if "decoy" in precursor_df.columns:
        precursor_df = precursor_df[precursor_df["decoy"] == 0]

    peptide_columns = [
        c for c in ["sequence", "mods", "mod_sites"] if c in precursor_df.columns
    ]
    peptide_df = precursor_df[peptide_columns].drop_duplicates()
    sequences = precursor_df["sequence"].drop_duplicates()

    sequence_lengths = sequences.str.len()

    precursor_charge = (
        (int(precursor_df["charge"].min()), int(precursor_df["charge"].max()))
        if "charge" in precursor_df.columns
        else None
    )
    mz_column = next(
        (c for c in PRECURSOR_MZ_COLUMNS if c in precursor_df.columns), None
    )
    precursor_mz = (
        (
            math.floor(precursor_df[mz_column].min()),
            math.ceil(precursor_df[mz_column].max()),
        )
        if mz_column is not None
        else None
    )

    enzyme, enzyme_specificity = _infer_enzyme(sequences)
    if enzyme == UNKNOWN_ENZYME:
        missed_cleavages, missed_cleavages_distribution = None, {}
    else:
        n_missed_cleavages = _count_missed_cleavages(sequences, enzyme)
        missed_cleavages = int(n_missed_cleavages.max())
        missed_cleavages_distribution = {
            int(k): int(v)
            for k, v in n_missed_cleavages.value_counts().sort_index().items()
        }

    fixed_mods, variable_mods, max_var_mod_num = _get_modifications(peptide_df)

    return LibrarySearchSpace(
        n_precursors=len(precursor_df),
        n_peptides=len(peptide_df),
        precursor_len=(int(sequence_lengths.min()), int(sequence_lengths.max())),
        precursor_charge=precursor_charge,
        precursor_mz=precursor_mz,
        enzyme=enzyme,
        enzyme_specificity=enzyme_specificity,
        missed_cleavages=missed_cleavages,
        missed_cleavages_distribution=missed_cleavages_distribution,
        fixed_modifications=fixed_mods,
        variable_modifications=variable_mods,
        max_var_mod_num=max_var_mod_num,
    )


def log_library_search_space(search_space: LibrarySearchSpace) -> None:
    """Log the search space using the keys and format of the config tree print (`library_prediction` section).

    The log is parsed by external tools (e.g. ProteoBench), so keys and format should be kept stable.
    Free-text lines must not contain a colon or any of the keys.
    """
    logger.info("============ Library search space ============")
    logger.info(
        f"Derived from {search_space.n_precursors:,} target precursors "
        f"({search_space.n_peptides:,} modified peptides) of the spectral library"
    )
    logger.info(
        f"Share of peptides with a C-terminal cleavage residue of the inferred enzyme ({search_space.enzyme}) "
        f"is {search_space.enzyme_specificity:.1%}"
    )
    logger.info(
        "The enzyme is an estimate from the C-terminal residues of the peptides and not always exact "
        "(e.g. trypsin and trypsin/P, or N-terminal cutters, cannot be told apart)"
    )
    logger.info(f"enzyme: {search_space.enzyme}")
    if search_space.missed_cleavages is not None:
        logger.info(f"missed_cleavages: {search_space.missed_cleavages}")
        distribution = ", ".join(
            f"{k}={v:,}" for k, v in search_space.missed_cleavages_distribution.items()
        )
        logger.info(f"Peptides per number of missed cleavages {distribution}")

    for key, min_max in [
        ("precursor_len", search_space.precursor_len),
        ("precursor_charge", search_space.precursor_charge),
        ("precursor_mz", search_space.precursor_mz),
    ]:
        if min_max is None:
            continue
        logger.info(f"{key}:")
        for value in min_max:
            logger.info(f"    - {value}")

    logger.info(f"fixed_modifications: {';'.join(search_space.fixed_modifications)}")
    logger.info(
        f"variable_modifications: {';'.join(search_space.variable_modifications)}"
    )
    logger.info(f"max_var_mod_num: {search_space.max_var_mod_num}")


class LogLibrarySearchSpace(ProcessingStep):
    def __init__(self) -> None:
        """Log the search space (digestion and modification settings) derived from the target precursors of a library."""
        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` (or `SpecLibFlat`) object."""
        return isinstance(input, SpecLibBase) and "sequence" in input.precursor_df

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Derive and log the search space, return the input unchanged."""
        log_library_search_space(get_library_search_space(input.precursor_df))
        return input

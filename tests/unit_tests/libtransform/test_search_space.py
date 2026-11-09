import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from alphabase.spectral_library.base import SpecLibBase

import alphadia
from alphadia.libtransform.search_space import (
    LibrarySearchSpace,
    LogLibrarySearchSpace,
    get_library_search_space,
)


def _precursor_df() -> pd.DataFrame:
    """Small library: charge states, a positional isomer, a repeated precursor and decoys that must be ignored."""
    return pd.DataFrame(
        {
            "sequence": [
                "PEPTIDEK",
                "PEPTIDEK",  # other charge state -> same modified peptide, other precursor
                "ACDEFKGHR",
                "MCPEPKPTR",
                "MAAAAMR",
                "MAAAAMR",  # positional isomer -> other modified peptide
                "MAAAAMR",  # repeated precursor (e.g. another channel) -> not counted again
                "AAAAAAAAAAAAAAAAAAAAAAAAAAK",  # 27 aa
                # decoys: different lengths, charges, mods and proteins, which must not show up in the results
                "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                "KKKKKRRRRA",
            ],
            "mods": [
                "",
                "",
                "Carbamidomethyl@C",
                "Acetyl@Protein_N-term;Oxidation@M;Carbamidomethyl@C",
                "Oxidation@M",
                "Oxidation@M",
                "Oxidation@M",
                "",
                "Phospho@S",
                "Phospho@S;Deamidated@N",
            ],
            "mod_sites": ["", "", "2", "0;1;2", "1", "6", "6", "", "1", "1;2"],
            "charge": [2, 3, 2, 3, 2, 2, 2, 4, 1, 6],
            "precursor_mz": [
                450.2,
                300.8,
                512.3,
                640.1,
                380.4,
                380.4,
                380.4,
                701.9,
                100.0,
                2000.0,
            ],
            "proteins": ["P1", "P1", "P2", "P2;P3", "P3", "P3", "P3", "P4", "D1", "D2"],
            "decoy": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        }
    )


def test_get_library_search_space():
    """Test that ranges, modifications and counts are derived from the targets only."""
    # when
    search_space = get_library_search_space(_precursor_df())

    # then: n_proteins counts P1-P4 ('P2;P3' adds no new protein, the decoy proteins are ignored)
    assert search_space == LibrarySearchSpace(
        peptide_len=(7, 27),
        precursor_charge=(2, 4),
        precursor_mz=(300, 702),
        modifications=["Acetyl@Protein_N-term", "Carbamidomethyl@C", "Oxidation@M"],
        n_modified_peptides=6,
        n_precursors=7,
        n_proteins=4,
    )


def test_get_library_search_space_uses_naa_and_flat_library_mz_column():
    """Test that `nAA` is used for the length and `mz_library` (renamed by `InitFlatColumns`) for the m/z."""
    precursor_df = _precursor_df().rename(columns={"precursor_mz": "mz_library"})
    precursor_df["nAA"] = np.int32(10)

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space.peptide_len == (10, 10)
    assert search_space.precursor_mz == (300, 702)


def test_get_library_search_space_handles_missing_mods():
    """Test that missing (NaN) modifications are treated like unmodified peptides."""
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDEK", "PEPTIDEK", "PEPTIDEK"],
            "mods": [np.nan, "", "Oxidation@M"],
            "mod_sites": [np.nan, "", "1"],
            "charge": [2, 2, 2],
        }
    )

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space.modifications == ["Oxidation@M"]
    assert search_space.n_modified_peptides == 2
    assert search_space.n_precursors == 2


def test_get_library_search_space_without_optional_columns():
    """Test that only the sequence column is required."""
    precursor_df = pd.DataFrame({"sequence": ["PEPTIDEK", "PEPKTIDER", "PEPTIDEK"]})

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space == LibrarySearchSpace(
        peptide_len=(8, 9),
        precursor_charge=None,
        precursor_mz=None,
        modifications=[],
        n_modified_peptides=2,
        n_precursors=2,
        n_proteins=None,
    )


def test_get_library_search_space_without_targets():
    """Test that a library without targets gives empty results instead of failing."""
    precursor_df = _precursor_df()
    precursor_df["decoy"] = 1

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space == LibrarySearchSpace(
        peptide_len=None,
        precursor_charge=None,
        precursor_mz=None,
        modifications=[],
        n_modified_peptides=0,
        n_precursors=0,
        n_proteins=0,
    )


def test_log_library_search_space_format(caplog):
    """Test the logged block: ranges in the format of the config tree print, then modifications and counts."""
    speclib = SpecLibBase()
    speclib._precursor_df = _precursor_df()

    # when
    with caplog.at_level(logging.INFO):
        result = LogLibrarySearchSpace()(speclib)

    # then
    assert result is speclib
    messages = [r.getMessage() for r in caplog.records]
    start = messages.index("============ Library search space ============")
    assert messages[start:] == [
        "============ Library search space ============",
        "Derived from the target precursors of the spectral library",
        "peptide_len:",
        "    - 7",
        "    - 27",
        "precursor_charge:",
        "    - 2",
        "    - 4",
        "precursor_mz:",
        "    - 300",
        "    - 702",
        "modifications: Acetyl@Protein_N-term;Carbamidomethyl@C;Oxidation@M",
        "n_modified_peptides: 6",
        "n_precursors: 7",
        "n_proteins: 4",
    ]


def test_log_library_search_space_is_enabled_by_default():
    """Test that the flag steering the search-space log exists in the default config and is enabled."""
    default_config_path = Path(alphadia.__file__).parent / "constants" / "default.yaml"
    with default_config_path.open() as f:
        default_config = yaml.safe_load(f)

    assert default_config["general"]["log_library_search_space"] is True

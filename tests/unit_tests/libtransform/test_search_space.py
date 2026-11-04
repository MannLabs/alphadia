import logging

import pandas as pd
import pytest
from alphabase.spectral_library.base import SpecLibBase

from alphadia.libtransform.search_space import (
    UNKNOWN_ENZYME,
    LogLibrarySearchSpace,
    get_library_search_space,
)


def _tryptic_precursor_df() -> pd.DataFrame:
    """Small tryptic library: two charge states of some peptides, plus decoys that must be ignored."""
    return pd.DataFrame(
        {
            "sequence": [
                "PEPTIDEK",  # 0 missed cleavages
                "PEPTIDEK",
                "ACDEFKGHR",  # 1 missed cleavage, C
                "MCPEPKPTR",  # 1 missed cleavage (KP, trypsin/P rule of alphabase), M and C
                "MAAAAAR",  # unmodified M
                "AAAAAAAAAAAAAAAAAAAAAAAAAAK",  # 27 aa
                # decoys: different lengths, charges, enzyme and mods, which must not show up in the results
                "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                "KKKKKRRRRA",
            ],
            "mods": [
                "",
                "",
                "Carbamidomethyl@C",
                "Acetyl@Protein_N-term;Oxidation@M;Carbamidomethyl@C",
                "",
                "",
                "Phospho@S",
                "Phospho@S",
            ],
            "mod_sites": ["", "", "2", "0;1;2", "", "", "1", "1"],
            "charge": [2, 3, 2, 3, 2, 4, 1, 6],
            "precursor_mz": [450.2, 300.8, 512.3, 640.1, 380.4, 701.9, 100.0, 2000.0],
            "decoy": [0, 0, 0, 0, 0, 0, 1, 1],
        }
    )


def test_get_library_search_space_tryptic():
    """Test that lengths, charges, m/z, enzyme, missed cleavages and modifications are derived from the targets."""
    # when
    search_space = get_library_search_space(_tryptic_precursor_df())

    # then
    assert search_space.n_precursors == 6
    assert search_space.n_peptides == 5
    assert search_space.precursor_len == (7, 27)
    assert search_space.precursor_charge == (2, 4)
    assert search_space.precursor_mz == (300, 702)
    assert search_space.enzyme == "trypsin"
    assert search_space.enzyme_specificity == 1.0
    assert search_space.missed_cleavages == 1
    assert search_space.missed_cleavages_distribution == {0: 3, 1: 2}
    # every C carries Carbamidomethyl -> fixed; one M is unmodified -> variable;
    # protein N-term acetylation is only on a subset of peptides -> variable
    assert search_space.fixed_modifications == ["Carbamidomethyl@C"]
    assert search_space.variable_modifications == [
        "Acetyl@Protein_N-term",
        "Oxidation@M",
    ]
    assert search_space.max_var_mod_num == 2


@pytest.mark.parametrize(
    ("c_terms", "expected_enzyme"),
    [
        ("KKKKKKKKKK", "lys-c"),
        ("RRRRRRRRRR", "arg-c"),
        ("KRKRKRKRKA", "trypsin"),  # 90% specific
        ("KRKRKRKRAA", UNKNOWN_ENZYME),  # 80% specific
    ],
)
def test_get_library_search_space_infers_enzyme(c_terms, expected_enzyme):
    """Test that the most specific enzyme explaining the C-termini is inferred."""
    precursor_df = pd.DataFrame(
        {"sequence": [f"PEPTIDE{'A' * i}{c}" for i, c in enumerate(c_terms)]}
    )

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space.enzyme == expected_enzyme
    if expected_enzyme == UNKNOWN_ENZYME:
        assert search_space.missed_cleavages is None
    else:
        assert search_space.missed_cleavages == 0


def test_get_library_search_space_uses_flat_library_mz_column():
    """Test that the precursor m/z is read from `mz_library` (renamed by `InitFlatColumns`) if `precursor_mz` is absent."""
    precursor_df = _tryptic_precursor_df().rename(
        columns={"precursor_mz": "mz_library"}
    )

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space.precursor_mz == (300, 702)


def test_get_library_search_space_without_optional_columns():
    """Test that only the sequence column is required."""
    precursor_df = pd.DataFrame({"sequence": ["PEPTIDEK", "PEPKTIDER"]})

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space.precursor_len == (8, 9)
    assert search_space.precursor_charge is None
    assert search_space.precursor_mz is None
    assert search_space.missed_cleavages == 1
    assert search_space.fixed_modifications == []
    assert search_space.variable_modifications == []
    assert search_space.max_var_mod_num == 0


def test_get_library_search_space_terminal_modifications():
    """Test fixed/variable classification of terminal modifications, with and without residue specificity."""
    precursor_df = pd.DataFrame(
        {
            "sequence": ["QPEPTIDEK", "QAAAAAAK", "EAAAAAAR"],
            "mods": [
                "Gln->pyro-Glu@Q^Any_N-term;Dimethyl@Any_N-term",
                "Gln->pyro-Glu@Q^Any_N-term;Dimethyl@Any_N-term",
                "Dimethyl@Any_N-term",
            ],
            "mod_sites": ["0;0", "0;0", "0"],
        }
    )

    # when
    search_space = get_library_search_space(precursor_df)

    # then
    assert search_space.fixed_modifications == [
        "Dimethyl@Any_N-term",
        "Gln->pyro-Glu@Q^Any_N-term",
    ]
    assert search_space.variable_modifications == []
    assert search_space.max_var_mod_num == 0


def test_log_library_search_space_format(caplog):
    """Test that the search space is logged with the config keys and in the format of the config tree print."""
    speclib = SpecLibBase()
    speclib._precursor_df = _tryptic_precursor_df()

    # when
    with caplog.at_level(logging.INFO):
        result = LogLibrarySearchSpace()(speclib)

    # then
    assert result is speclib
    messages = [r.getMessage() for r in caplog.records]
    start = messages.index("enzyme: trypsin")
    assert messages[start:] == [
        "enzyme: trypsin",
        "missed_cleavages: 1",
        "Peptides per number of missed cleavages 0=3, 1=2",
        "precursor_len:",
        "    - 7",
        "    - 27",
        "precursor_charge:",
        "    - 2",
        "    - 4",
        "precursor_mz:",
        "    - 300",
        "    - 702",
        "fixed_modifications: Carbamidomethyl@C",
        "variable_modifications: Acetyl@Protein_N-term;Oxidation@M",
        "max_var_mod_num: 2",
    ]
    # free-text lines must not be picked up as key-value pairs by log parsers
    assert all(":" not in m for m in messages[:start] if "Running" not in m)

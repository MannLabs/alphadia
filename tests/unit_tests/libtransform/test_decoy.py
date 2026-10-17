import pandas as pd
from alphabase.spectral_library.decoy import DIANNDecoyGenerator, decoy_lib_provider

from alphadia.libtransform.decoy import (
    DIANN_INNER,
    DIANN_KEEP_PROLINE,
    DIANNInnerDecoyGenerator,
    DIANNKeepProlineDecoyGenerator,
)
from alphadia.libtransform.diann27_decoy import (
    DIANN27DecoyGenerator,
    diann27_decoy_precursors,
)


def test_keep_proline_decoy_mutates_the_residue_before_the_proline():
    assert DIANNKeepProlineDecoyGenerator()._decoy("AGLLDPK") == "ALLLEPK"


def test_keep_proline_decoy_equals_diann_decoy_without_proline_before_the_c_terminus():
    for sequence in ("AGLLDEK", "PEPTIDER", "APPLEPAR"):
        assert DIANNKeepProlineDecoyGenerator()._decoy(
            sequence
        ) == DIANNDecoyGenerator()._decoy(sequence)


def test_keep_proline_decoy_is_registered():
    assert isinstance(
        decoy_lib_provider.decoy_dict[DIANN_KEEP_PROLINE](),
        DIANNKeepProlineDecoyGenerator,
    )


def test_inner_decoy_mutates_the_third_and_the_fourth_to_last_residue():
    assert DIANNInnerDecoyGenerator()._decoy("AGLLDEVK") == "AGVLDDVK"


def test_inner_decoy_keeps_b2_and_y2():
    decoy = DIANNInnerDecoyGenerator()._decoy("LGEHNIDVLEGNEQFINAAK")
    assert decoy[:2] == "LG"
    assert decoy[-2:] == "AK"


def test_inner_decoy_moves_a_mutation_off_a_proline():
    assert DIANNInnerDecoyGenerator()._decoy("PEPTIDEK") == "PEPSIEEK"
    assert DIANNInnerDecoyGenerator()._decoy("AAAAPGK") == "AALLPGK"


def test_inner_decoy_is_registered():
    assert isinstance(
        decoy_lib_provider.decoy_dict[DIANN_INNER](), DIANNInnerDecoyGenerator
    )


DIANN27_WEIGHTS = {aa: 1.0 for aa in "ADEFGHIKLMNPQRSTVWY"}


def test_diann27_decoy_swaps_the_second_and_the_second_to_last_residue():
    assert DIANN27DecoyGenerator(DIANN27_WEIGHTS).decoy("AFLLDEVK", "", "") == (
        "AVLLDEFK",
        "",
        "",
    )


def test_diann27_decoy_moves_modifications_with_their_residue():
    assert DIANN27DecoyGenerator(DIANN27_WEIGHTS).decoy(
        "ACLLDEVK", "Acetyl@Protein_N-term;Carbamidomethyl@C", "0;2"
    ) == ("AVLLDECK", "Acetyl@Protein_N-term;Carbamidomethyl@C", "0;7")


def test_diann27_decoy_does_not_swap_residues_in_a_shift_window():
    sequence = "AELLDGFLK"
    decoy, _, _ = DIANN27DecoyGenerator(DIANN27_WEIGHTS).decoy(sequence, "", "")
    assert decoy != "ALLLDGFEK"
    assert decoy[0] == sequence[0] and decoy[-1] == sequence[-1]
    assert decoy[1] != sequence[1] and decoy[-2] != sequence[-2]


def test_diann27_decoy_is_deterministic():
    generator = DIANN27DecoyGenerator(DIANN27_WEIGHTS)
    assert generator.decoy("ALGDEFIKLLK", "", "") == generator.decoy(
        "ALGDEFIKLLK", "", ""
    )


def test_diann27_decoy_precursors_give_every_charge_the_same_decoy():
    precursor_df = pd.DataFrame(
        {
            "sequence": ["AFLLDEVK", "AFLLDEVK", "PEPTIDER"],
            "mods": ["", "", ""],
            "mod_sites": ["", "", ""],
            "charge": [2, 3, 2],
        }
    )
    decoy_df = diann27_decoy_precursors(precursor_df, mp_process_num=1)
    assert decoy_df["sequence"].tolist()[:2] == ["AVLLDEFK", "AVLLDEFK"]
    assert decoy_df["charge"].tolist() == [2, 3, 2]


def test_diann27_inner_decoy_keeps_the_outer_two_residues_and_swaps_the_third():
    assert DIANN27DecoyGenerator(DIANN27_WEIGHTS, keep_outer_ions=True).decoy(
        "AGFLLDEVTK", "", ""
    ) == ("AGVLLDEFTK", "", "")


def test_diann27_inner_decoy_moves_modifications_with_their_residue():
    assert DIANN27DecoyGenerator(DIANN27_WEIGHTS, keep_outer_ions=True).decoy(
        "MGCLLDEVTK", "Acetyl@Protein_N-term;Oxidation@M;Carbamidomethyl@C", "0;1;3"
    ) == (
        "MGVLLDECTK",
        "Acetyl@Protein_N-term;Oxidation@M;Carbamidomethyl@C",
        "0;1;8",
    )

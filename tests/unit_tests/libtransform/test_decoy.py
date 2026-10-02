from alphabase.spectral_library.decoy import DIANNDecoyGenerator, decoy_lib_provider

from alphadia.libtransform.decoy import (
    DIANN_INNER,
    DIANN_KEEP_PROLINE,
    DIANNInnerDecoyGenerator,
    DIANNKeepProlineDecoyGenerator,
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

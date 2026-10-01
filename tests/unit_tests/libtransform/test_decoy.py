from alphabase.spectral_library.decoy import DIANNDecoyGenerator, decoy_lib_provider

from alphadia.libtransform.decoy import (
    DIANN_KEEP_PROLINE,
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

import logging

import numpy as np
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import DIANNDecoyGenerator, decoy_lib_provider

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()

DIANN_KEEP_PROLINE = "diann_keep_proline"
DIANN_INNER = "diann_inner"


class DIANNKeepProlineDecoyGenerator(DIANNDecoyGenerator):
    """DIA-NN decoy that keeps a proline before the C-terminal residue and mutates the residue before it instead.

    The DIA-NN map turns that proline into L, so no decoy ends in xP[KR]. Those peptides fragment into an intense
    y2 (PK / PR) ion of the same mass for all of them, which only targets can match; where it is abundant (plasma),
    false targets ending in xP[KR] pass without any decoy to account for them.
    """

    def _decoy(self, sequence: str) -> str:
        if sequence[-2] != "P":
            return super()._decoy(sequence)
        return (
            sequence[0]
            + self._mutate(sequence[1])
            + sequence[2:-3]
            + self._mutate(sequence[-3])
            + sequence[-2:]
        )

    def _mutate(self, amino_acid: str) -> str:
        return self.mutated_AAs[self.raw_AAs.index(amino_acid)]


decoy_lib_provider.register(DIANN_KEEP_PROLINE, DIANNKeepProlineDecoyGenerator)


class DIANNInnerDecoyGenerator(DIANNKeepProlineDecoyGenerator):
    """DIA-NN map applied to the third and the fourth-to-last residue instead of the second and the second-to-last.

    Every decoy keeps its target's b1, b2, y1 and y2. These short ions are the ones a false target most often shares
    with co-eluting peptides by chance; with the outermost residues mutated, decoys carry rarer short-ion masses and
    match fewer of them, so they underestimate false targets. A proline at either mutated position moves that
    mutation one residue inwards, as in `DIANNKeepProlineDecoyGenerator`.
    """

    def _decoy(self, sequence: str) -> str:
        first, second = 2, len(sequence) - 3
        if sequence[second] == "P":
            second -= 1
        if sequence[first] == "P" and first + 1 < second:
            first += 1
        residues = list(sequence)
        residues[first] = self._mutate(sequence[first])
        residues[second] = self._mutate(sequence[second])
        return "".join(residues)


decoy_lib_provider.register(DIANN_INNER, DIANNInnerDecoyGenerator)

DIANN_INNER_ALT1 = "diann_inner_alt1"
DIANN_INNER_ALT2 = "diann_inner_alt2"
# Residue maps of the extra MBR decoys: the same mutated positions as `DIANNInnerDecoyGenerator`, other residues. Each
# map gives every residue a mass shift different from the other maps (and none near-isobaric to the original, e.g. no
# K <-> Q), so the decoys of one target share only its unmutated ions and are separate draws of the same null.
_RAW_AAS = "GAVLIFMPWSCTYHKRQENDBJOUXZsty"
_ALT1_AAS = "AGFFFMFAYAAGFNRKHASTVVVVVVtss"
_ALT2_AAS = "STYYYWYSFGGAWFHHTSGAVVVVVVtss"


class DIANNInnerAlt1DecoyGenerator(DIANNInnerDecoyGenerator):
    """`DIANNInnerDecoyGenerator` with the first alternative residue map."""

    def __init__(self, raw_AAs: str = _RAW_AAS, mutated_AAs: str = _ALT1_AAS):  # noqa: N803 # alphabase's names
        super().__init__(raw_AAs, mutated_AAs)


class DIANNInnerAlt2DecoyGenerator(DIANNInnerDecoyGenerator):
    """`DIANNInnerDecoyGenerator` with the second alternative residue map."""

    def __init__(self, raw_AAs: str = _RAW_AAS, mutated_AAs: str = _ALT2_AAS):  # noqa: N803 # alphabase's names
        super().__init__(raw_AAs, mutated_AAs)


decoy_lib_provider.register(DIANN_INNER_ALT1, DIANNInnerAlt1DecoyGenerator)
decoy_lib_provider.register(DIANN_INNER_ALT2, DIANNInnerAlt2DecoyGenerator)
EXTRA_DECOY_TYPES = (DIANN_INNER_ALT1, DIANN_INNER_ALT2)


def generate_decoy_library(
    target_lib: SpecLibBase, decoy_type: str, mp_process_num: int
) -> SpecLibBase:
    """A library of one decoy of `decoy_type` per target of `target_lib` (a copy, `target_lib` is not changed)."""
    decoy_lib = decoy_lib_provider.get_decoy_lib(decoy_type, target_lib.copy())
    decoy_lib.charged_frag_types = target_lib.charged_frag_types
    decoy_lib.decoy_sequence(mp_process_num=mp_process_num)
    decoy_lib.calc_precursor_mz()
    decoy_lib.remove_unused_fragments()
    decoy_lib.calc_fragment_mz_df()
    decoy_lib._precursor_df["decoy"] = 1
    return decoy_lib


class DecoyGenerator(ProcessingStep):
    def __init__(self, decoy_type: str = "diann", mp_process_num: int = 8) -> None:
        """Generate decoys for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------
        decoy_type : str, optional
            Type of decoys to generate: `pseudo_reverse`, `diann`, `diann_keep_proline` or `diann_inner`. Default is `diann`.

        """
        super().__init__()
        self.decoy_type = decoy_type
        self.mp_process_num = mp_process_num

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Generate decoys for the spectral library."""
        if "decoy" not in input.precursor_df.columns:
            input.precursor_df["decoy"] = 0

        decoy_values = input.precursor_df["decoy"].unique()
        if len(decoy_values) > 1:
            logger.info("Decoys already present, skipping decoy generation")
            return input

        decoy_lib = generate_decoy_library(input, self.decoy_type, self.mp_process_num)

        # keep original precursor_idx and only create new ones for decoys
        start_precursor_idx = input.precursor_df["precursor_idx"].max() + 1
        decoy_lib._precursor_df["precursor_idx"] = np.arange(
            start_precursor_idx, start_precursor_idx + len(decoy_lib.precursor_df)
        )

        input.append(decoy_lib)
        input._precursor_df.sort_values("elution_group_idx", inplace=True)
        input._precursor_df.reset_index(drop=True, inplace=True)
        input.precursor_df["precursor_idx"] = np.arange(len(input.precursor_df))
        input.remove_unused_fragments()

        return input

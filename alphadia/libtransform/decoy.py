import logging
import zlib

import numpy as np
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import DIANNDecoyGenerator, decoy_lib_provider

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()

DIANN_KEEP_PROLINE = "diann_keep_proline"
DIANN_INNER = "diann_inner"
DIANN_INNER_FREQUENCY = "diann_inner_frequency"
# Interior residue frequencies of the human tryptic library (I counted as L); C and P are never drawn.
INTERIOR_RESIDUE_FREQUENCY = {
    "A": 0.0766,
    "D": 0.0540,
    "E": 0.0814,
    "F": 0.0410,
    "G": 0.0712,
    "H": 0.0294,
    "K": 0.0216,
    "L": 0.1611,
    "M": 0.0217,
    "N": 0.0406,
    "Q": 0.0537,
    "R": 0.0212,
    "S": 0.0914,
    "T": 0.0581,
    "V": 0.0667,
    "W": 0.0136,
    "Y": 0.0297,
}


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


class DIANNInnerFrequencyDecoyGenerator(DIANNInnerDecoyGenerator):
    """Inner decoy whose replacement residues are drawn at library frequency instead of taken from the fixed map.

    The fixed map sends most residues to L, V or S, so inner decoys' b3+ / y3+ compositions are ~1.2x as common among
    library peptides as a false target's. Decoys then share fragments with co-eluting candidates more often than false
    targets do, which features describing that competition learn to recognise. A residue drawn at the library's own
    frequency (never the original, C or P) gives decoys a false target's ion commonness. The draw is seeded by the
    sequence, so every charge state of a peptide gets the same decoy.
    """

    def _decoy(self, sequence: str) -> str:
        rng = np.random.default_rng(zlib.crc32(sequence.encode()))
        first, second = 2, len(sequence) - 3
        if sequence[second] == "P":
            second -= 1
        if sequence[first] == "P" and first + 1 < second:
            first += 1
        residues = list(sequence)
        for position in (first, second):
            choices = [
                aa for aa in INTERIOR_RESIDUE_FREQUENCY if aa != sequence[position]
            ]
            weights = np.array([INTERIOR_RESIDUE_FREQUENCY[aa] for aa in choices])
            residues[position] = choices[
                rng.choice(len(choices), p=weights / weights.sum())
            ]
        return "".join(residues)


decoy_lib_provider.register(DIANN_INNER_FREQUENCY, DIANNInnerFrequencyDecoyGenerator)


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

        decoy_lib = decoy_lib_provider.get_decoy_lib(self.decoy_type, input.copy())

        decoy_lib.charged_frag_types = input.charged_frag_types
        decoy_lib.decoy_sequence(mp_process_num=self.mp_process_num)
        decoy_lib.calc_precursor_mz()
        decoy_lib.remove_unused_fragments()
        decoy_lib.calc_fragment_mz_df()
        decoy_lib._precursor_df["decoy"] = 1

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

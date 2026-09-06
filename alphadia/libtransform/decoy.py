import logging
import random
import zlib

import numpy as np
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import BaseDecoyGenerator, decoy_lib_provider

from alphadia.constants.keys import DecoyType
from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()

# A shuffle that reproduces the target is retried this often before it is given up on;
# alphabase drops decoys identical to a target afterwards anyway.
_MAX_SHUFFLE_ATTEMPTS = 10


class ShuffleDecoyGenerator(BaseDecoyGenerator):
    """Shuffle the residues between the fixed first and last one.

    Reversal and DIA-NN's two-residue mutation keep most of the target's local sequence
    context, so the decoy's fragment series still resemble a real peptide's. A shuffle
    destroys that context. The permutation is seeded by the sequence, so a decoy is the
    same in every process and every run.
    """

    def _decoy(self, sequence: str) -> str:
        if len(sequence) < 4:  # noqa: PLR2004
            return sequence
        rng = random.Random(zlib.crc32(sequence.encode()))
        inner = list(sequence[1:-1])
        for _ in range(_MAX_SHUFFLE_ATTEMPTS):
            rng.shuffle(inner)
            decoy = sequence[0] + "".join(inner) + sequence[-1]
            if decoy != sequence:
                break
        return decoy


decoy_lib_provider.register(DecoyType.SHUFFLE, ShuffleDecoyGenerator)  # ty: ignore[invalid-argument-type] # alphabase's annotation names the instance, the registry holds classes


class DecoyGenerator(ProcessingStep):
    def __init__(
        self, decoy_type: str = DecoyType.DIANN, mp_process_num: int = 8
    ) -> None:
        """Generate decoys for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------
        decoy_type : str, optional
            Type of decoys to generate: `diann` (default), `pseudo_reverse` or `shuffle`.

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

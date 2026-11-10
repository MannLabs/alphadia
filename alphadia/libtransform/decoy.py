import logging

import numpy as np
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import DIANNDecoyGenerator, decoy_lib_provider

from alphadia.constants.keys import DecoyType
from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()

DIANN_ALT1 = "diann_alt1"
DIANN_ALT2 = "diann_alt2"

# Residue maps of the extra MBR decoys: the positions of the DIA-NN decoy, other residues. Each map gives every residue
# a mass shift different from the DIA-NN map and from the other map (and none near-isobaric to the original, e.g. no
# K <-> Q), so the decoys of one target share only its unmutated ions and are separate draws of the same null.
_RAW_AAS = "GAVLIFMPWSCTYHKRQENDBJOUXZsty"
_ALT1_AAS = "AGFFFMFAYAAGFNRKHASTVVVVVVtss"
_ALT2_AAS = "STYYYWYSFGGAWFHHTSGAVVVVVVtss"


class DIANNAlt1DecoyGenerator(DIANNDecoyGenerator):
    """DIA-NN decoy with the first alternative residue map."""

    def __init__(self, raw_AAs: str = _RAW_AAS, mutated_AAs: str = _ALT1_AAS):  # noqa: N803 # alphabase's names
        super().__init__(raw_AAs, mutated_AAs)


class DIANNAlt2DecoyGenerator(DIANNDecoyGenerator):
    """DIA-NN decoy with the second alternative residue map."""

    def __init__(self, raw_AAs: str = _RAW_AAS, mutated_AAs: str = _ALT2_AAS):  # noqa: N803 # alphabase's names
        super().__init__(raw_AAs, mutated_AAs)


decoy_lib_provider.register(DIANN_ALT1, DIANNAlt1DecoyGenerator)
decoy_lib_provider.register(DIANN_ALT2, DIANNAlt2DecoyGenerator)

# the extra decoys of `append_extra_decoys`, by the decoy type of the library
EXTRA_DECOY_TYPES = {DecoyType.DIANN: (DIANN_ALT1, DIANN_ALT2)}


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


def append_extra_decoys(
    speclib: SpecLibBase, decoy_type: str, n_extra: int, mp_process_num: int = 8
) -> SpecLibBase:
    """`speclib` (targets and one decoy of `decoy_type` each) with `n_extra` more decoys per target."""
    extra_types = EXTRA_DECOY_TYPES.get(decoy_type, ())
    if n_extra > len(extra_types):
        raise ValueError(
            f"{decoy_type} decoys support at most {len(extra_types)} extra decoys per target, not {n_extra}"
        )
    targets = speclib.copy()
    targets._precursor_df = targets._precursor_df[
        targets._precursor_df["decoy"] == 0
    ].copy()
    targets.remove_unused_fragments()
    for extra_type in extra_types[:n_extra]:
        speclib.append(generate_decoy_library(targets, extra_type, mp_process_num))
    speclib._precursor_df.sort_values("elution_group_idx", inplace=True)
    speclib._precursor_df.reset_index(drop=True, inplace=True)
    speclib._precursor_df["precursor_idx"] = np.arange(len(speclib._precursor_df))
    speclib.remove_unused_fragments()
    return speclib


class DecoyGenerator(ProcessingStep):
    def __init__(
        self, decoy_type: str = DecoyType.DIANN, mp_process_num: int = 8
    ) -> None:
        """Generate decoys for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------
        decoy_type : str, optional
            Type of decoys to generate. Currently only `pseudo_reverse` and `diann` are supported. Default is `diann`.

        """
        super().__init__()
        if decoy_type not in DecoyType.get_values():
            raise ValueError(
                f"Unknown decoy_type: {decoy_type}. Valid options are {DecoyType.get_values()}"
            )
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

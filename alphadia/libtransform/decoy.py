import logging

import numpy as np
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import decoy_lib_provider

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()


class DecoyGenerator(ProcessingStep):
    def __init__(self, decoy_type: str = "diann", mp_process_num: int = 8) -> None:
        """Generate decoys for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------
        decoy_type : str, optional
            Type of decoys to generate. Currently only `pseudo_reverse` and `diann` are supported. Default is `diann`.

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
            logger.warning(
                "Input library already contains decoys. Skipping decoy generation. \n Please note that decoys generated outside of alphabase are not supported."
            )
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

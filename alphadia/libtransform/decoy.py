import logging

import numpy as np
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import decoy_lib_provider

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()


HIDDEN_DECOY_SEED = 42
HIDDEN_DECOY_VALUE = 2


class DecoyGenerator(ProcessingStep):
    def __init__(
        self,
        decoy_type: str = "diann",
        mp_process_num: int = 8,
        hidden_decoy_fraction: float = 0.0,
    ) -> None:
        """Generate decoys for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------
        decoy_type : str, optional
            Type of decoys to generate. Currently only `pseudo_reverse` and `diann` are supported. Default is `diann`.

        hidden_decoy_fraction : float, optional
            Fraction of decoys reserved as hidden (decoy=2) for final FDR estimation.
            Set to 0.0 to disable (default, current behavior).

        """
        super().__init__()
        self.decoy_type = decoy_type
        self.mp_process_num = mp_process_num
        self.hidden_decoy_fraction = hidden_decoy_fraction

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

        if self.hidden_decoy_fraction > 0.0:
            n_decoys = len(decoy_lib._precursor_df)
            n_hidden = int(round(n_decoys * self.hidden_decoy_fraction))
            rng = np.random.default_rng(seed=HIDDEN_DECOY_SEED)
            hidden_indices = rng.choice(n_decoys, size=n_hidden, replace=False)
            decoy_lib._precursor_df.iloc[
                hidden_indices,
                decoy_lib._precursor_df.columns.get_loc("decoy"),
            ] = HIDDEN_DECOY_VALUE
            logger.info(
                f"Hidden decoy split: {n_decoys - n_hidden} training, {n_hidden} hidden"
            )

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

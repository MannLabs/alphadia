import logging

import numpy as np
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import decoy_lib_provider

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()

DUAL_DECOY_SENTINEL = 99
DUAL_DECOY_HIDDEN_FRACTION = 0.25


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
            Fraction of decoys to mark as hidden entrapment targets (0.0 = disabled). Default is 0.0.

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

        input._precursor_df["is_hidden_decoy"] = False
        decoy_lib._precursor_df["is_hidden_decoy"] = False

        if self.hidden_decoy_fraction == DUAL_DECOY_SENTINEL:
            decoy_lib2 = decoy_lib_provider.get_decoy_lib(
                "pseudo_reverse", input.copy()
            )
            decoy_lib2.charged_frag_types = input.charged_frag_types
            decoy_lib2.decoy_sequence(mp_process_num=self.mp_process_num)
            decoy_lib2.calc_precursor_mz()
            decoy_lib2.remove_unused_fragments()
            decoy_lib2.calc_fragment_mz_df()
            decoy_lib2._precursor_df["decoy"] = 1
            decoy_lib2._precursor_df["is_hidden_decoy"] = False

            rng = np.random.default_rng(seed=42)

            n_decoys_1 = len(decoy_lib._precursor_df)
            n_hidden_1 = int(n_decoys_1 * DUAL_DECOY_HIDDEN_FRACTION)
            hidden_idx_1 = rng.choice(n_decoys_1, size=n_hidden_1, replace=False)
            decoy_lib._precursor_df.iloc[
                hidden_idx_1, decoy_lib._precursor_df.columns.get_loc("decoy")
            ] = 0
            decoy_lib._precursor_df.iloc[
                hidden_idx_1,
                decoy_lib._precursor_df.columns.get_loc("is_hidden_decoy"),
            ] = True

            n_decoys_2 = len(decoy_lib2._precursor_df)
            n_hidden_2 = int(n_decoys_2 * DUAL_DECOY_HIDDEN_FRACTION)
            hidden_idx_2 = rng.choice(n_decoys_2, size=n_hidden_2, replace=False)
            decoy_lib2._precursor_df.iloc[
                hidden_idx_2, decoy_lib2._precursor_df.columns.get_loc("decoy")
            ] = 0
            decoy_lib2._precursor_df.iloc[
                hidden_idx_2,
                decoy_lib2._precursor_df.columns.get_loc("is_hidden_decoy"),
            ] = True

            logger.info(
                f"Created dual decoy libraries: {n_hidden_1} hidden ({n_hidden_1/n_decoys_1:.1%}) + {n_decoys_1 - n_hidden_1} visible ({(n_decoys_1 - n_hidden_1)/n_decoys_1:.1%}) ({self.decoy_type}), "
                f"{n_hidden_2} hidden ({n_hidden_2/n_decoys_2:.1%}) + {n_decoys_2 - n_hidden_2} visible ({(n_decoys_2 - n_hidden_2)/n_decoys_2:.1%}) (pseudo_reverse)"
            )

            decoy_lib.append(decoy_lib2)
        elif self.hidden_decoy_fraction > 0:
            n_decoys = len(decoy_lib._precursor_df)
            n_hidden = int(n_decoys * self.hidden_decoy_fraction)
            if n_hidden == 0:
                logger.warning(
                    f"hidden_decoy_fraction={self.hidden_decoy_fraction} results in 0 hidden decoys (n_decoys={n_decoys}), skipping"
                )
            else:
                rng = np.random.default_rng(seed=42)
                hidden_idx = rng.choice(n_decoys, size=n_hidden, replace=False)
                decoy_lib._precursor_df.iloc[
                    hidden_idx, decoy_lib._precursor_df.columns.get_loc("decoy")
                ] = 0
                decoy_lib._precursor_df.iloc[
                    hidden_idx,
                    decoy_lib._precursor_df.columns.get_loc("is_hidden_decoy"),
                ] = True
                logger.info(
                    f"Created {n_hidden} hidden decoys out of {n_decoys} total decoys ({self.hidden_decoy_fraction:.1%})"
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

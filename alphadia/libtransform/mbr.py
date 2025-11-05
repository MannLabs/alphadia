import logging

import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()


class MbrLibraryBuilder(ProcessingStep):
    def __init__(self, fdr=0.01, keep_decoy_in_mbr_library=True) -> None:
        super().__init__()
        self.fdr = fdr
        self.keep_decoy_in_mbr_library = keep_decoy_in_mbr_library

    def validate(self, psm_df, base_library) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return True

    def forward(self, psm_df: pd.DataFrame, base_library: SpecLibBase) -> SpecLibBase:
        # Filter by FDR threshold
        psm_df = psm_df[psm_df["qval"] <= self.fdr]

        if self.keep_decoy_in_mbr_library:
            # Keep both targets and decoys that passed FDR
            # Aggregate RT for both targets and decoys separately
            rt_df = psm_df.groupby("elution_group_idx", as_index=False).agg(
                rt=pd.NamedAgg(column=CalibCols.RT_OBSERVED, aggfunc="median"),
                pg=pd.NamedAgg(column="pg", aggfunc="first"),
            )

            # Get elution groups with targets that passed FDR
            target_elution_groups = psm_df[psm_df["decoy"] == 0][
                "elution_group_idx"
            ].unique()

            # Filter to include only:
            # 1. Elution groups with targets that passed FDR
            # 2. Both targets and decoys from those groups
            rt_df = rt_df[rt_df["elution_group_idx"].isin(target_elution_groups)]

        else:
            # Original behavior: only targets
            psm_df = psm_df[psm_df["decoy"] == 0]

            rt_df = psm_df.groupby("elution_group_idx", as_index=False).agg(
                rt=pd.NamedAgg(column=CalibCols.RT_OBSERVED, aggfunc="median"),
                pg=pd.NamedAgg(column="pg", aggfunc="first"),
            )

        mbr_spec_lib = base_library.copy()
        if "rt" in mbr_spec_lib._precursor_df.columns:
            mbr_spec_lib._precursor_df.drop(columns=["rt"], inplace=True)

        mbr_spec_lib._precursor_df = mbr_spec_lib._precursor_df.merge(
            rt_df, on="elution_group_idx", how="right"
        )
        mbr_spec_lib._precursor_df["genes"] = mbr_spec_lib._precursor_df["pg"]
        mbr_spec_lib._precursor_df["proteins"] = mbr_spec_lib._precursor_df["pg"]

        mbr_spec_lib._precursor_df.drop(columns=["pg"], inplace=True)

        mbr_spec_lib.remove_unused_fragments()

        return mbr_spec_lib

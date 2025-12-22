import logging

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df

from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep
from alphadia.libtransform.decoy import DecoyGenerator

logger = logging.getLogger()


def _compute_lookup_indices(
    target_keys: np.ndarray,
    target_fallback_keys: np.ndarray,
    fallback_lookup_keys: np.ndarray,
    specific_lookup_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute indices for value lookup with fallback and specific matching.

    This function computes indices that map each target to a value in a lookup table.
    It uses a two-level lookup strategy:
    1. Fallback: Every target gets an index based on its fallback key (elution_group_idx)
    2. Specific: Targets whose primary key (mod_seq_charge_hash) exists in the specific
       lookup table get marked for override

    Both lookups use pandas hash-based indexing for O(n) average case complexity.

    Parameters
    ----------
    target_keys : np.ndarray
        Primary keys for specific lookup (e.g., lib mod_seq_charge_hash).
    target_fallback_keys : np.ndarray
        Fallback keys for each target (e.g., lib elution_group_idx).
    fallback_lookup_keys : np.ndarray
        Keys in fallback lookup table (e.g., PSM elution_group_idx).
    specific_lookup_keys : np.ndarray
        Keys in specific lookup table (e.g., PSM mod_seq_charge_hash).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - fallback_indices: Index into fallback lookup for each target
        - specific_indices: Index into specific lookup for each target (valid only where specific_index_mask=True)
        - specific_index_mask: Boolean mask indicating which targets have a specific match

    """
    n_targets = len(target_keys)

    fallback_key_to_idx = pd.Series(
        np.arange(len(fallback_lookup_keys)), index=fallback_lookup_keys
    )
    fallback_indices = fallback_key_to_idx.reindex(target_fallback_keys).values.astype(
        np.int64
    )

    if len(specific_lookup_keys) > 0:
        specific_key_to_idx = pd.Series(
            np.arange(len(specific_lookup_keys)), index=specific_lookup_keys
        )
        specific_lookup = specific_key_to_idx.reindex(target_keys)
        specific_index_mask = ~pd.isna(specific_lookup.values)
        specific_indices = np.where(
            specific_index_mask, specific_lookup.values, 0
        ).astype(np.int64)
    else:
        specific_index_mask = np.zeros(n_targets, dtype=bool)
        specific_indices = np.zeros(n_targets, dtype=np.int64)

    return fallback_indices, specific_indices, specific_index_mask


def _apply_lookup_indices(
    fallback_values: np.ndarray,
    specific_values: np.ndarray,
    fallback_indices: np.ndarray,
    specific_indices: np.ndarray,
    specific_index_mask: np.ndarray,
) -> np.ndarray:
    """Apply precomputed indices to get values."""
    result = fallback_values[fallback_indices]
    if specific_index_mask.any():
        result[specific_index_mask] = specific_values[
            specific_indices[specific_index_mask]
        ]
    return result


class MbrLibraryBuilder(ProcessingStep):
    def __init__(self, fdr=0.01, keep_decoys=True) -> None:
        super().__init__()
        self.fdr = fdr
        self.keep_decoys = keep_decoys

    def validate(self, psm_df, base_library) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return True

    def _assign_rt_and_protein_groups(
        self,
        mbr_spec_lib: SpecLibBase,
        agg_by_eg: pd.DataFrame,
        agg_by_hash: pd.DataFrame,
    ) -> None:
        """Assign RT and protein groups to MBR library precursors.

        Parameters
        ----------
        mbr_spec_lib : SpecLibBase
            MBR library to update in-place.
        agg_by_eg : pd.DataFrame
            Aggregated PSM data by elution_group_idx with columns: elution_group_idx, rt, pg.
        agg_by_hash : pd.DataFrame
            Aggregated PSM data by mod_seq_charge_hash with columns: mod_seq_charge_hash, rt, pg.

        """
        lib_hashes = mbr_spec_lib._precursor_df["mod_seq_charge_hash"].values
        lib_eg_indices = mbr_spec_lib._precursor_df["elution_group_idx"].values

        fallback_indices, specific_indices, specific_index_mask = (
            _compute_lookup_indices(
                target_keys=lib_hashes,
                target_fallback_keys=lib_eg_indices,
                fallback_lookup_keys=agg_by_eg["elution_group_idx"].values,
                specific_lookup_keys=agg_by_hash["mod_seq_charge_hash"].values,
            )
        )

        mbr_spec_lib._precursor_df["rt"] = _apply_lookup_indices(
            fallback_values=agg_by_eg["rt"].values,
            specific_values=agg_by_hash["rt"].values,
            fallback_indices=fallback_indices,
            specific_indices=specific_indices,
            specific_index_mask=specific_index_mask,
        )

        pg_values = _apply_lookup_indices(
            fallback_values=agg_by_eg["pg"].values,
            specific_values=agg_by_hash["pg"].values,
            fallback_indices=fallback_indices,
            specific_indices=specific_indices,
            specific_index_mask=specific_index_mask,
        )
        mbr_spec_lib._precursor_df["genes"] = pg_values
        mbr_spec_lib._precursor_df["proteins"] = pg_values

    def forward(self, psm_df: pd.DataFrame, base_library: SpecLibBase) -> SpecLibBase:
        """Build MBR library from PSM results and base library.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM results with columns: elution_group_idx, decoy, qval, rt_observed,
            pg, mod_seq_charge_hash.
        base_library : SpecLibBase
            Base spectral library containing target precursors.

        Returns
        -------
        SpecLibBase
            MBR library with RT and protein group assignments.

        Notes
        -----
        MBR library generation procedure:
        1. Filter PSMs by FDR threshold
        2. Get elution groups that passed FDR (targets only, or targets+decoys)
        3. Filter base library to those elution groups
        4. Generate decoys if keep_decoys=True, then rehash so each
           precursor has a unique mod_seq_charge_hash
        5. Assign RT and protein groups to each precursor

        RT and protein group assignment strategy:
        - If a precursor's mod_seq_charge_hash was identified in PSM, use its specific RT/pg
        - Otherwise, fall back to the median RT / first pg of its elution group

        This ensures identified precursors get their observed values while unidentified
        precursors (e.g., decoys after rehashing) inherit sensible defaults from their
        elution group.

        """
        psm_df = psm_df[psm_df["qval"] <= self.fdr]

        if self.keep_decoys:
            elution_groups = psm_df["elution_group_idx"].unique()
        else:
            elution_groups = psm_df[psm_df["decoy"] == 0]["elution_group_idx"].unique()

        mbr_spec_lib = base_library.copy()
        mbr_spec_lib._precursor_df = mbr_spec_lib._precursor_df[
            mbr_spec_lib._precursor_df["elution_group_idx"].isin(elution_groups)
        ].copy()
        mbr_spec_lib.remove_unused_fragments()

        if self.keep_decoys:
            decoy_generator = DecoyGenerator(decoy_type="diann")
            mbr_spec_lib = decoy_generator(mbr_spec_lib)
            # Decoys inherit target hashes from DecoyGenerator, rehash to get unique hashes
            mbr_spec_lib._precursor_df = hash_precursor_df(mbr_spec_lib._precursor_df)

        agg_by_eg = psm_df.groupby("elution_group_idx", as_index=False).agg(
            rt=pd.NamedAgg(column=CalibCols.RT_OBSERVED, aggfunc="median"),
            pg=pd.NamedAgg(column="pg", aggfunc="first"),
        )
        agg_by_hash = psm_df.groupby("mod_seq_charge_hash", as_index=False).agg(
            rt=pd.NamedAgg(column=CalibCols.RT_OBSERVED, aggfunc="median"),
            pg=pd.NamedAgg(column="pg", aggfunc="first"),
        )

        self._assign_rt_and_protein_groups(mbr_spec_lib, agg_by_eg, agg_by_hash)

        return mbr_spec_lib

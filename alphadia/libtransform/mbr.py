import logging

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df

from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep
from alphadia.libtransform.decoy import DecoyGenerator

logger = logging.getLogger()


def compute_lookup_indices(
    target_keys: np.ndarray,
    target_fallback_keys: np.ndarray,
    fallback_lookup_keys: np.ndarray,
    specific_lookup_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute indices for value lookup. Sort once, reuse for multiple value columns.

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
        (fallback_indices, specific_indices, has_specific_mask)

    """
    n_targets = len(target_keys)

    fallback_sort_idx = np.argsort(fallback_lookup_keys)
    sorted_fallback_keys = fallback_lookup_keys[fallback_sort_idx]
    insert_idx = np.searchsorted(sorted_fallback_keys, target_fallback_keys)
    insert_idx = np.clip(insert_idx, 0, len(sorted_fallback_keys) - 1)
    fallback_indices = fallback_sort_idx[insert_idx]

    if len(specific_lookup_keys) > 0:
        specific_key_to_idx = pd.Series(
            np.arange(len(specific_lookup_keys)), index=specific_lookup_keys
        )
        specific_lookup = specific_key_to_idx.reindex(target_keys)
        has_specific = ~pd.isna(specific_lookup.values)
        specific_indices = np.where(has_specific, specific_lookup.values, 0).astype(
            np.int64
        )
    else:
        has_specific = np.zeros(n_targets, dtype=bool)
        specific_indices = np.zeros(n_targets, dtype=np.int64)

    return fallback_indices, specific_indices, has_specific


def apply_lookup_indices(
    fallback_values: np.ndarray,
    specific_values: np.ndarray,
    fallback_indices: np.ndarray,
    specific_indices: np.ndarray,
    has_specific: np.ndarray,
) -> np.ndarray:
    """Apply precomputed indices to get values."""
    result = fallback_values[fallback_indices]
    if has_specific.any():
        result[has_specific] = specific_values[specific_indices[has_specific]]
    return result


class MbrLibraryBuilder(ProcessingStep):
    def __init__(self, fdr=0.01, keep_decoys_in_mbr_library=True) -> None:
        super().__init__()
        self.fdr = fdr
        self.keep_decoys_in_mbr_library = keep_decoys_in_mbr_library

    def validate(self, psm_df, base_library) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return True

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
        4. Generate decoys if keep_decoys_in_mbr_library=True, then rehash so each
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

        if self.keep_decoys_in_mbr_library:
            elution_groups = psm_df["elution_group_idx"].unique()
        else:
            elution_groups = psm_df[psm_df["decoy"] == 0]["elution_group_idx"].unique()

        mbr_spec_lib = base_library.copy()
        mbr_spec_lib._precursor_df = mbr_spec_lib._precursor_df[
            mbr_spec_lib._precursor_df["elution_group_idx"].isin(elution_groups)
        ].copy()
        mbr_spec_lib.remove_unused_fragments()

        if self.keep_decoys_in_mbr_library:
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

        lib_hashes = mbr_spec_lib._precursor_df["mod_seq_charge_hash"].values
        lib_eg_idx = mbr_spec_lib._precursor_df["elution_group_idx"].values

        fallback_idx, specific_idx, has_specific = compute_lookup_indices(
            target_keys=lib_hashes,
            target_fallback_keys=lib_eg_idx,
            fallback_lookup_keys=agg_by_eg["elution_group_idx"].values,
            specific_lookup_keys=agg_by_hash["mod_seq_charge_hash"].values,
        )

        mbr_spec_lib._precursor_df["rt"] = apply_lookup_indices(
            fallback_values=agg_by_eg["rt"].values,
            specific_values=agg_by_hash["rt"].values,
            fallback_indices=fallback_idx,
            specific_indices=specific_idx,
            has_specific=has_specific,
        )

        pg_values = apply_lookup_indices(
            fallback_values=agg_by_eg["pg"].values,
            specific_values=agg_by_hash["pg"].values,
            fallback_indices=fallback_idx,
            specific_indices=specific_idx,
            has_specific=has_specific,
        )
        mbr_spec_lib._precursor_df["genes"] = pg_values
        mbr_spec_lib._precursor_df["proteins"] = pg_values

        return mbr_spec_lib

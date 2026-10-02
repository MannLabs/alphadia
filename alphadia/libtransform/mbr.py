import logging
from copy import copy

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep
from alphadia.libtransform.decoy import DIANN_INNER, DecoyGenerator

logger = logging.getLogger()


class IndexBuilder:
    """Build and apply lookup indices with fallback and specific matching.

    This class computes indices that map each target to a value in a lookup table.
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

    """

    def __init__(
        self,
        target_keys: np.ndarray,
        target_fallback_keys: np.ndarray,
        fallback_lookup_keys: np.ndarray,
        specific_lookup_keys: np.ndarray,
    ) -> None:
        fallback_key_to_idx = pd.Series(
            np.arange(len(fallback_lookup_keys)), index=fallback_lookup_keys
        )
        self._fallback_indices = fallback_key_to_idx.reindex(
            target_fallback_keys
        ).values.astype(np.int64)

        if len(specific_lookup_keys) > 0:
            specific_key_to_idx = pd.Series(
                np.arange(len(specific_lookup_keys)), index=specific_lookup_keys
            )
            specific_lookup = specific_key_to_idx.reindex(target_keys)
            self._specific_index_mask = ~pd.isna(specific_lookup.values)
            self._specific_indices = np.where(
                self._specific_index_mask, specific_lookup.values, 0
            ).astype(np.int64)
        else:
            n_targets = len(target_keys)
            self._specific_index_mask = np.zeros(n_targets, dtype=bool)
            self._specific_indices = np.zeros(n_targets, dtype=np.int64)

    def apply(
        self, fallback_values: np.ndarray, specific_values: np.ndarray
    ) -> np.ndarray:
        """Apply precomputed indices to get values.

        Parameters
        ----------
        fallback_values : np.ndarray
            Values from the fallback lookup table.
        specific_values : np.ndarray
            Values from the specific lookup table.

        Returns
        -------
        np.ndarray
            Result array with fallback values, overridden by specific values where available.

        """
        assert len(fallback_values) > self._fallback_indices.max(), (
            f"fallback_values length {len(fallback_values)} must be greater than "
            f"max fallback index {self._fallback_indices.max()}"
        )
        result = fallback_values[self._fallback_indices]
        if self._specific_index_mask.any():
            max_specific_idx = self._specific_indices[self._specific_index_mask].max()
            assert len(specific_values) > max_specific_idx, (
                f"specific_values length {len(specific_values)} must be greater than "
                f"max specific index {max_specific_idx}"
            )
            result[self._specific_index_mask] = specific_values[
                self._specific_indices[self._specific_index_mask]
            ]
        return result


class MbrLibraryBuilder(ProcessingStep):
    def __init__(self, fdr: float = 0.01, keep_decoys: bool = False) -> None:
        super().__init__()
        self.fdr = fdr
        self.keep_decoys = keep_decoys
        self.rt_by_run: pd.DataFrame | None = None
        self.pg_qval: pd.DataFrame | None = None

    def validate(self, psm_df: pd.DataFrame, base_library: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return True

    def _assign_rt_and_protein_groups(
        self,
        mbr_speclib: SpecLibBase,
        agg_by_eg: pd.DataFrame,
        agg_by_hash: pd.DataFrame,
    ) -> None:
        """Assign RT and protein groups to MBR library precursors.

        Parameters
        ----------
        mbr_speclib : SpecLibBase
            MBR library to update in-place.
        agg_by_eg : pd.DataFrame
            Aggregated PSM data by elution_group_idx with columns: elution_group_idx, rt, pg.
        agg_by_hash : pd.DataFrame
            Aggregated PSM data by mod_seq_charge_hash with columns: mod_seq_charge_hash, rt, pg.

        """
        index_builder = IndexBuilder(
            target_keys=mbr_speclib._precursor_df["mod_seq_charge_hash"].values,
            target_fallback_keys=mbr_speclib._precursor_df["elution_group_idx"].values,
            fallback_lookup_keys=agg_by_eg["elution_group_idx"].values,
            specific_lookup_keys=agg_by_hash["mod_seq_charge_hash"].values,
        )

        rt_values = index_builder.apply(
            fallback_values=agg_by_eg["rt"].values,
            specific_values=agg_by_hash["rt"].values,
        )
        pg_values = index_builder.apply(
            fallback_values=agg_by_eg["pg"].values,
            specific_values=agg_by_hash["pg"].values,
        )

        mbr_speclib._precursor_df["rt"] = rt_values
        mbr_speclib._precursor_df["genes"] = pg_values
        mbr_speclib._precursor_df["proteins"] = pg_values

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
        # The precursor table reaching here is filtered at the configured fdr.fdr, which entrapment and other
        # diagnostic runs raise above 1 %. Protein groups that only passed at that looser level would enter the MBR
        # library, where the MBR step's own protein FDR, computed over the pre-selected groups, cannot reject them.
        psm_df = psm_df[(psm_df["qval"] <= self.fdr) & (psm_df["pg_qval"] <= self.fdr)]

        self.pg_qval = psm_df.groupby("pg", as_index=False)["pg_qval"].max()

        self.rt_by_run = (
            psm_df[psm_df["decoy"] == 0]
            .groupby(["elution_group_idx", "run"], as_index=False)
            .agg(
                rt_observed=pd.NamedAgg(column=CalibCols.RT_OBSERVED, aggfunc="median"),
                rt_calibrated=pd.NamedAgg(
                    column=CalibCols.RT_CALIBRATED, aggfunc="median"
                ),
            )
        )

        if self.keep_decoys:
            elution_groups = psm_df["elution_group_idx"].unique()
        else:
            elution_groups = psm_df[psm_df["decoy"] == 0]["elution_group_idx"].unique()

        mbr_speclib = base_library.copy()
        mbr_speclib._precursor_df = mbr_speclib._precursor_df[
            mbr_speclib._precursor_df["elution_group_idx"].isin(elution_groups)
        ].copy()
        mbr_speclib.remove_unused_fragments()

        if self.keep_decoys:
            decoy_generator = DecoyGenerator(decoy_type=DIANN_INNER)
            mbr_speclib = decoy_generator(mbr_speclib)
            # Decoys inherit target hashes from DecoyGenerator, rehash to get unique hashes
            mbr_speclib._precursor_df = hash_precursor_df(mbr_speclib._precursor_df)

        agg_by_eg = psm_df.groupby("elution_group_idx", as_index=False).agg(
            rt=pd.NamedAgg(column=CalibCols.RT_OBSERVED, aggfunc="median"),
            pg=pd.NamedAgg(column="pg", aggfunc="first"),
        )
        agg_by_hash = psm_df.groupby("mod_seq_charge_hash", as_index=False).agg(
            rt=pd.NamedAgg(column=CalibCols.RT_OBSERVED, aggfunc="median"),
            pg=pd.NamedAgg(column="pg", aggfunc="first"),
        )

        self._assign_rt_and_protein_groups(mbr_speclib, agg_by_eg, agg_by_hash)

        return mbr_speclib


def leave_one_out_rt(
    speclib: SpecLibFlat, rt_by_run: pd.DataFrame, run: str
) -> SpecLibFlat:
    """Return the library with each elution group's RT taken from the runs other than `run`.

    The MBR library RT is the median observed RT over the runs that are searched again, so a
    first-pass identification, false ones included, would get its own apex as library RT and a
    near-zero RT error, which the decoys generated for the MBR step cannot reproduce. An elution
    group identified in `run` only keeps the first-pass calibrated library RT of that run.
    """
    other_runs = rt_by_run[rt_by_run["run"] != run]
    this_run = rt_by_run[rt_by_run["run"] == run]
    rt = (
        other_runs.groupby("elution_group_idx")["rt_observed"]
        .median()
        .combine_first(this_run.set_index("elution_group_idx")["rt_calibrated"])
    )

    precursor_df = speclib.precursor_df.copy()
    loo_rt = precursor_df["elution_group_idx"].map(rt)
    precursor_df[CalibCols.RT_LIBRARY] = loo_rt.fillna(
        precursor_df[CalibCols.RT_LIBRARY]
    ).to_numpy()

    speclib_for_run = copy(speclib)
    speclib_for_run._precursor_df = precursor_df
    return speclib_for_run


def bound_by_library_pg_qval(
    psm_df: pd.DataFrame, library_pg_qval: pd.DataFrame
) -> pd.DataFrame:
    """Raise each protein group's q-value to its q-value in the search that built the MBR library.

    The MBR step only searches the protein groups that passed the first search, with fresh decoys for them, so its
    protein FDR does not see that the false groups among them were already selected out of the whole protein space.
    A group's q-value is therefore at least the one it had there.
    """
    library_qval = psm_df["pg"].map(library_pg_qval.set_index("pg")["pg_qval"])
    psm_df["pg_qval"] = np.maximum(psm_df["pg_qval"], library_qval.fillna(0).to_numpy())
    return psm_df

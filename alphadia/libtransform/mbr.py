import logging
from copy import copy

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep
from alphadia.libtransform.decoy import (
    DIANN_INNER,
    EXTRA_DECOY_TYPES,
    DecoyGenerator,
    generate_decoy_library,
)

logger = logging.getLogger()

HASH_COLUMN = "mod_seq_charge_hash"
# 1 for the decoy of an elution group the first search admitted on its decoy: it stands for a false library entry
KEPT_DECOY_COLUMN = "mbr_kept_decoy"
# 1 for the precursors of elution groups the first search did not identify in the run being searched
TRANSFER_COLUMN = "mbr_transfer"
# the backend writes the fragment type as its character code
FRAGMENT_TYPE_CODES = {ord("b"): "b", ord("y"): "y"}
FRAGMENT_COLUMNS = [
    "precursor_idx",
    "type",
    "charge",
    "loss_type",
    "position",
    "intensity",
]
DECOY_PROCESSES = 8


def empirical_fragment_intensities(
    admitted: pd.DataFrame, fragments_by_run: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """The observed fragment pattern of each library precursor: per run max-normalised, median over the runs.

    `admitted` holds the admitted first-search PSMs (run, precursor_idx, mod_seq_charge_hash, decoy). A decoy carries
    its target's hash and fragment positions, so a group admitted on its decoy gets the decoy's pattern; where the
    target is admitted too, the target's pattern is used.

    Returns columns mod_seq_charge_hash, column (library fragment column, e.g. y_z1), position and intensity.
    """
    frames = []
    for run, fragments in fragments_by_run.items():
        psms = admitted[admitted["run"] == run][["precursor_idx", HASH_COLUMN, "decoy"]]
        f = fragments[
            (fragments["loss_type"] == 0)
            & fragments["type"].isin(list(FRAGMENT_TYPE_CODES))
            & (fragments["intensity"] > 0)
        ].merge(psms, on="precursor_idx")
        f["intensity"] = f["intensity"] / f.groupby("precursor_idx")[
            "intensity"
        ].transform("max")
        f["run"] = run
        frames.append(f)
    observed = pd.concat(frames, ignore_index=True)
    # a hash admitted as a target keeps only its target observations
    has_target = observed.groupby(HASH_COLUMN)["decoy"].transform("min") == 0
    observed = observed[~has_target | (observed["decoy"] == 0)]
    observed["column"] = (
        observed["type"].map(FRAGMENT_TYPE_CODES)
        + "_z"
        + observed["charge"].astype(str)
    )
    # a fragment without signal in a run the precursor was seen in counts as 0 there
    keys = [HASH_COLUMN, "column", "position"]
    grid = (
        observed[keys]
        .drop_duplicates()
        .merge(observed[[HASH_COLUMN, "run"]].drop_duplicates(), on=HASH_COLUMN)
        .merge(observed[[*keys, "run", "intensity"]], on=[*keys, "run"], how="left")
        .fillna({"intensity": 0.0})
    )
    summed = grid.groupby(keys, as_index=False)["intensity"].median()
    summed = summed[summed["intensity"] > 0]
    summed["intensity"] /= summed.groupby(HASH_COLUMN)["intensity"].transform("max")
    return summed[[HASH_COLUMN, "column", "position", "intensity"]]


def apply_empirical_intensities(speclib: SpecLibBase, empirical: pd.DataFrame) -> None:
    """Replace the fragment intensities of the precursors in `empirical` by their observed pattern, in place;
    fragments not in the pattern get 0."""
    precursor_df = speclib._precursor_df
    covered = precursor_df[precursor_df[HASH_COLUMN].isin(empirical[HASH_COLUMN])]
    values = speclib._fragment_intensity_df.to_numpy(copy=True)
    starts = covered["frag_start_idx"].to_numpy()
    lengths = covered["frag_stop_idx"].to_numpy() - starts
    values[
        np.repeat(starts, lengths)
        + np.arange(lengths.sum())
        - np.repeat(np.cumsum(lengths) - lengths, lengths)
    ] = 0.0
    placed = empirical.merge(
        covered[[HASH_COLUMN, "frag_start_idx", "frag_stop_idx"]], on=HASH_COLUMN
    )
    rows = placed["frag_start_idx"].to_numpy() + placed["position"].to_numpy()
    inside = rows < placed["frag_stop_idx"].to_numpy()
    columns = speclib._fragment_intensity_df.columns.get_indexer(placed["column"])
    inside &= columns >= 0
    values[rows[inside], columns[inside]] = placed["intensity"].to_numpy()[inside]
    speclib._fragment_intensity_df = pd.DataFrame(
        values, columns=speclib._fragment_intensity_df.columns
    )
    logger.info(
        f"Empirical fragment intensities for {len(covered):,} of {len(precursor_df):,} MBR library precursors"
    )


def append_extra_decoys(speclib: SpecLibBase, n_extra: int) -> SpecLibBase:
    """`speclib` (targets and one decoy each) with `n_extra` more decoys per target from `EXTRA_DECOY_TYPES`."""
    targets = speclib.copy()
    targets._precursor_df = targets._precursor_df[
        targets._precursor_df["decoy"] == 0
    ].copy()
    targets.remove_unused_fragments()
    for decoy_type in EXTRA_DECOY_TYPES[:n_extra]:
        speclib.append(generate_decoy_library(targets, decoy_type, DECOY_PROCESSES))
    speclib._precursor_df.sort_values("elution_group_idx", inplace=True)
    speclib._precursor_df.reset_index(drop=True, inplace=True)
    speclib._precursor_df["precursor_idx"] = np.arange(len(speclib._precursor_df))
    speclib.remove_unused_fragments()
    return speclib


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
    def __init__(
        self,
        fdr: float = 0.01,
        keep_decoys: bool = False,
        fragments_by_run: dict[str, pd.DataFrame] | None = None,
        decoys_per_target: int = 1,
    ) -> None:
        """`fragments_by_run` (the first search's fragment tables) gives the library its observed fragment
        intensities; `decoys_per_target` > 1 (with `keep_decoys`) adds decoys from `EXTRA_DECOY_TYPES`."""
        super().__init__()
        self.fdr = fdr
        self.keep_decoys = keep_decoys
        self.fragments_by_run = fragments_by_run
        self.decoys_per_target = decoys_per_target
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

        if self.fragments_by_run is not None:
            # before the decoys are generated, so that they copy the observed pattern of their target
            apply_empirical_intensities(
                mbr_speclib,
                empirical_fragment_intensities(psm_df, self.fragments_by_run),
            )

        if self.keep_decoys:
            decoy_generator = DecoyGenerator(decoy_type=DIANN_INNER)
            mbr_speclib = decoy_generator(mbr_speclib)
            # the first search's decoy is the DIANN_INNER copy; extra decoys appended below are fresh
            kept_groups = psm_df.loc[psm_df["decoy"] == 1, "elution_group_idx"]
            precursor_df = mbr_speclib._precursor_df
            precursor_df[KEPT_DECOY_COLUMN] = (
                (precursor_df["decoy"] == 1)
                & precursor_df["elution_group_idx"].isin(kept_groups)
            ).astype(np.int8)
            if self.decoys_per_target > 1:
                mbr_speclib = append_extra_decoys(
                    mbr_speclib, self.decoys_per_target - 1
                )
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
    precursor_df[TRANSFER_COLUMN] = (
        ~precursor_df["elution_group_idx"].isin(this_run["elution_group_idx"])
    ).astype(np.int8)

    speclib_for_run = copy(speclib)
    speclib_for_run._precursor_df = precursor_df
    return speclib_for_run

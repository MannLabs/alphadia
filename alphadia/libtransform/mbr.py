import logging
from copy import copy

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import CalibCols, DecoyType, MbrCols
from alphadia.libtransform.base import ProcessingStep
from alphadia.libtransform.decoy import DecoyGenerator

logger = logging.getLogger()

# median observed RT over the runs other than the one of the row (`_median_of_other_rows`)
_RT_OTHER_RUNS = "rt_other_runs"


def _passing_fdr(psm_df: pd.DataFrame, fdr: float) -> pd.DataFrame:
    """Return the PSMs whose precursor and protein group both pass `fdr`."""
    # The precursor table is filtered at the configured fdr.fdr, which entrapment and other diagnostic runs raise
    # above 1 %. Protein groups that only passed at that looser level would enter the MBR library, where the MBR
    # step cannot reject them any more.
    return psm_df[(psm_df["qval"] <= fdr) & (psm_df["pg_qval"] <= fdr)]


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
        decoy_type: str = DecoyType.DIANN,
    ) -> None:
        super().__init__()
        self.fdr = fdr
        self.keep_decoys = keep_decoys
        self.decoy_type = decoy_type

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
        1. Filter PSMs by FDR threshold, of the precursor and of its protein group
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
        psm_df = _passing_fdr(psm_df, self.fdr)

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
            decoy_generator = DecoyGenerator(decoy_type=self.decoy_type)
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


def get_run_identifications(psm_df: pd.DataFrame, fdr: float) -> pd.DataFrame:
    """Return the precursors each run identified at `fdr`, targets and decoys, the same PSMs `MbrLibraryBuilder` with
    this `fdr` builds its library from, with their observed and calibrated RT and their protein group with its q-value.

    `CalibCols.RT_CALIBRATED` is NaN for runs whose PSMs do not have it.
    """
    run_identifications_df = (
        _passing_fdr(psm_df, fdr)
        .reindex(
            columns=[
                "run",
                "elution_group_idx",
                "mod_seq_charge_hash",
                "decoy",
                CalibCols.RT_OBSERVED,
                CalibCols.RT_CALIBRATED,
                "pg",
                "pg_qval",
            ]
        )
        .reset_index(drop=True)
    )
    run_identifications_df["run"] = run_identifications_df["run"].astype("category")
    return run_identifications_df


def _median_of_other_rows(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """For each row, return the median of `values` over the other rows with the same key, NaN if there are none.

    After one sort, each median is read off at fixed positions, so the cost does not grow with the square of the rows
    per key, as computing a median of the other rows for each row would.
    """
    order = np.lexsort((values, keys))
    sorted_keys = keys[order]
    sorted_values = values[order]

    is_group_start = np.ones(len(keys), dtype=bool)
    is_group_start[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = np.flatnonzero(is_group_start)
    group_sizes = np.diff(np.append(group_starts, len(keys)))
    group_of_row = np.cumsum(is_group_start) - 1

    group_start_of_row = group_starts[group_of_row]
    rank_in_group = np.arange(len(keys)) - group_start_of_row
    n_other_rows = group_sizes[group_of_row] - 1

    def value_among_other_rows(position: np.ndarray) -> np.ndarray:
        # positions at or after the row itself are shifted by one to skip it
        sorted_position = (
            group_start_of_row + position + (position >= rank_in_group)
        ).clip(0, max(len(keys) - 1, 0))
        return sorted_values[sorted_position]

    median_sorted = np.where(
        n_other_rows > 0,
        (
            value_among_other_rows((n_other_rows - 1) // 2)
            + value_among_other_rows(n_other_rows // 2)
        )
        / 2,
        np.nan,
    )
    median = np.empty(len(keys))
    median[order] = median_sorted
    return median


def _get_rt_by_run(run_identifications_df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Return the median RTs of each run and `key`, with the median observed RT of the other runs in `_RT_OTHER_RUNS`."""
    rt_by_run_df = run_identifications_df.groupby(
        [key, "run"], observed=True, as_index=False
    )[[CalibCols.RT_OBSERVED, CalibCols.RT_CALIBRATED]].median()
    rt_by_run_df[_RT_OTHER_RUNS] = _median_of_other_rows(
        rt_by_run_df[key].to_numpy(), rt_by_run_df[CalibCols.RT_OBSERVED].to_numpy()
    )
    return rt_by_run_df


class PerRunMbrLibrary:
    """Adapt the MBR library to each run it is searched in, using the precursors each run identified in the search
    that built the library (`get_run_identifications`).

    The MBR library RT is the median observed RT over the runs, so a precursor that a run identified, false
    identifications included, would have that run's own apex in its library RT. With a single run, it would get a
    near-zero RT error there, which its decoy cannot reproduce. So in the elution groups a run identified, each
    precursor takes its library RT from the other runs: its own median observed RT where available, else the median
    of its elution group. Where no other run identified the elution group, the calibrated RT the run had in that
    search is used, and without that the elution group is not searched in the run.

    A run identified an elution group if its target or its decoy passed there, and the elution group medians pool
    both. A decoy kept in the MBR library then mirrors a false target: in the run where it passed, it is not searched
    at its own apex, and its target, without observations of its own, takes the decoy's RT from the other runs, as
    the fresh decoy of a false target takes the target's.
    """

    def __init__(self, run_identifications_df: pd.DataFrame) -> None:
        self._rt_by_run_and_precursor_df = _get_rt_by_run(
            run_identifications_df, "mod_seq_charge_hash"
        )
        self._rt_by_run_and_elution_group_df = _get_rt_by_run(
            run_identifications_df, "elution_group_idx"
        )
        self._rt_all_runs_by_precursor = self._rt_by_run_and_precursor_df.groupby(
            "mod_seq_charge_hash"
        )[CalibCols.RT_OBSERVED].median()

        self._precursor_rows_by_run = self._rt_by_run_and_precursor_df.groupby(
            "run", observed=True
        ).indices
        self._elution_group_rows_by_run = self._rt_by_run_and_elution_group_df.groupby(
            "run", observed=True
        ).indices

    def for_run(self, speclib: SpecLibFlat, run: str) -> SpecLibFlat:
        """Return a copy of `speclib` with the library RTs for `run` and the column `MbrCols.IDENTIFIED_IN_RUN`."""
        precursors_in_run_df = self._rt_by_run_and_precursor_df.iloc[
            self._precursor_rows_by_run.get(run, [])
        ].set_index("mod_seq_charge_hash")
        elution_groups_in_run_df = self._rt_by_run_and_elution_group_df.iloc[
            self._elution_group_rows_by_run.get(run, [])
        ].set_index("elution_group_idx")

        precursor_df = speclib.precursor_df
        precursor_hash = precursor_df["mod_seq_charge_hash"]
        elution_group = precursor_df["elution_group_idx"]

        # a precursor the run did not identify has the same observed RTs in all runs as in the other runs
        rt_other_runs = precursor_hash.map(precursors_in_run_df[_RT_OTHER_RUNS]).where(
            precursor_hash.isin(precursors_in_run_df.index),
            precursor_hash.map(self._rt_all_runs_by_precursor),
        )
        rt_for_run = (
            rt_other_runs.fillna(
                elution_group.map(elution_groups_in_run_df[_RT_OTHER_RUNS])
            )
            .fillna(precursor_hash.map(precursors_in_run_df[CalibCols.RT_CALIBRATED]))
            .fillna(
                elution_group.map(elution_groups_in_run_df[CalibCols.RT_CALIBRATED])
            )
        )

        identified_in_run = elution_group.isin(elution_groups_in_run_df.index)
        without_rt = identified_in_run & rt_for_run.isna()
        if without_rt.any():
            logger.warning(
                f"{run}: {elution_group[without_rt].nunique():,} elution groups were only identified in this run and "
                f"have no calibrated RT, they are not searched"
            )

        speclib_for_run = copy(speclib)
        speclib_for_run._precursor_df = precursor_df.assign(
            **{
                CalibCols.RT_LIBRARY: rt_for_run.where(
                    identified_in_run, precursor_df[CalibCols.RT_LIBRARY]
                ),
                MbrCols.IDENTIFIED_IN_RUN: identified_in_run.astype(np.int8),
            }
        )[~without_rt].reset_index(drop=True)
        return speclib_for_run

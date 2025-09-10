"""Score group container for managing scoring results.

This module provides container classes for organizing groups of scores
and their associated metadata during the scoring process.
"""

import logging

import numba as nb
import numpy as np

from alphadia.constants.keys import CalibCols
from alphadia.constants.settings import NUM_FEATURES
from alphadia.search.scoring.containers.candidate import Candidate, candidate_type

logger = logging.getLogger()


@nb.experimental.jitclass()
class ScoreGroup:
    elution_group_idx: nb.uint32
    score_group_idx: nb.uint32

    candidates: nb.types.ListType(candidate_type)

    def __init__(
        self, elution_group_idx: nb.uint32, score_group_idx: nb.uint32
    ) -> None:
        self.elution_group_idx = elution_group_idx
        self.score_group_idx = score_group_idx

        self.candidates = nb.typed.List.empty_list(candidate_type)

    def __getitem__(self, idx):
        return self.candidates[idx]

    def __len__(self):
        return len(self.candidates)

    def process(
        self,
        psm_proto_df,
        fragment_container,
        jit_data,
        config,
        quadrupole_calibration,
        debug,
    ) -> None:
        # get refrerence channel index
        if config.reference_channel >= 0:
            reference_channel_idx = -1
            for idx, candidate in enumerate(self.candidates):
                if candidate.channel == config.reference_channel:
                    reference_channel_idx = idx
                    break

            # return if reference channel not found
            if reference_channel_idx == -1:
                print(
                    "reference channel not found",
                    self.elution_group_idx,
                    self.score_group_idx,
                )
                return

        # process candidates
        for candidate in self.candidates:
            candidate.process(
                jit_data,
                psm_proto_df,
                fragment_container,
                config,
                quadrupole_calibration,
                debug,
            )

        # process reference channel features
        # TODO: code was unused, check if it needs re-implementation


score_group_type = ScoreGroup.class_type.instance_type


@nb.experimental.jitclass()
class ScoreGroupContainer:
    """Container for managing the scoring of precursors with defined boundaries.

    The `ScoreGroupContainer` contains all precursors that are to be scored.
    It consists of a list of `ScoreGroup` objects, which in turn contain a list of `Candidate` objects.

    For single channel experiments, each `ScoreGroup` contains a single `Candidate` object.
    For multi channel experiments, each `ScoreGroup` contains a `Candidate` object for each channel, including decoy channels.

    Structure:
    .. code-block:: none

        ScoreGroupContainer
            ScoreGroup
                Candidate
                Candidate
                Candidate
                Candidate
            ScoreGroup
                Candidate
                Candidate
                Candidate
                Candidate

    The `ScoreGroupContainer` is initialized by passing the validated columns of a candidate dataframe to the `build_from_df` method.


    Attributes
    ----------

    score_groups : nb.types.ListType(score_group_type)
        List of score groups.

    Methods
    -------

    __getitem__(self, idx: int):
        Get a score group by index.


    """

    score_groups: nb.types.ListType(score_group_type)

    def __init__(
        self,
    ) -> None:
        """Initialize the `ScoreGroupContainer` object without any score groups."""

        self.score_groups = nb.typed.List.empty_list(score_group_type)

    def __getitem__(self, idx):
        """Get a score group by index."""

        return self.score_groups[idx]

    def __len__(self):
        """Get the number of score groups."""
        return len(self.score_groups)

    def build_from_df(
        self,
        elution_group_idx: nb.uint32,
        score_group_idx: nb.uint32,
        precursor_idx: nb.uint32,
        channel: nb.uint8,
        rank: nb.uint8,
        flat_frag_start_idx: nb.uint32,
        flat_frag_stop_idx: nb.uint32,
        scan_start: nb.uint32,
        scan_stop: nb.uint32,
        scan_center: nb.uint32,
        frame_start: nb.uint32,
        frame_stop: nb.uint32,
        frame_center: nb.uint32,
        precursor_charge: nb.uint8,
        precursor_mz: nb.float32,
        precursor_isotopes: nb.float32[:, ::1],
    ):
        """Build the `ScoreGroupContainer` from a candidate dataframe.
        All relevant columns of the candidate dataframe are passed to this method as numpy arrays.

        Note
        ----

        All columns of the candidate_df need to be validated for the correct type using the `extraction.validate.candidates` schema.
        columns musst be sorted by `score_group_idx` in ascending order.

        Parameters
        ----------

        elution_group_idx : nb.uint32
            The elution group index of each precursor candidate.

        score_group_idx : nb.uint32
            The score group index of each precursor candidate.

        """
        idx = 0
        current_score_group_idx = -1
        current_precursor_idx = -1

        # iterate over all candidates
        # whenever a new score group is encountered, create a new score group
        for idx in range(len(score_group_idx)):
            if score_group_idx[idx] != current_score_group_idx:
                self.score_groups.append(
                    ScoreGroup(elution_group_idx[idx], score_group_idx[idx])
                )

                # update current score group
                current_score_group_idx = score_group_idx[idx]
                current_precursor_idx = -1

            if len(precursor_isotopes[idx]) == 0:
                raise ValueError("precursor isotopes empty")

            self.score_groups[-1].candidates.append(
                Candidate(
                    idx,
                    precursor_idx[idx],
                    channel[idx],
                    rank[idx],
                    flat_frag_start_idx[idx],
                    flat_frag_stop_idx[idx],
                    scan_start[idx],
                    scan_stop[idx],
                    scan_center[idx],
                    frame_start[idx],
                    frame_stop[idx],
                    frame_center[idx],
                    precursor_charge[idx],
                    precursor_mz[idx],
                    precursor_isotopes[idx].copy(),
                )
            )

            # check if precursor_idx is unique within a score group
            # if not, some weird "ValueError: unable to broadcast argument 1 to output array" will be raised.
            # Numba bug which took 4h to find :'(
            if current_precursor_idx == precursor_idx[idx]:
                raise ValueError("precursor_idx must be unique within a score group")

            current_precursor_idx = precursor_idx[idx]
            # idx += 1

    def get_feature_columns(self):
        """Iterate all score groups and candidates and return a list of all feature names

        Is based on the assumption that each set of features has a distinct length.

        Parameters
        ----------

        score_group_continer : list
            List of score groups

        Returns
        -------

        list
            List of feature names

        """

        known_feature_lengths = [0]
        known_feature_lengths.clear()
        known_columns = [""]
        known_columns.clear()

        for i in range(len(self)):
            for j in range(len(self[i].candidates)):
                candidate = self[i].candidates[j]
                if len(candidate.features) not in known_feature_lengths:
                    known_feature_lengths += [len(candidate.features)]
                    # add all new features to the list of known columns
                    for key in candidate.features.keys():  # noqa: SIM118
                        if key not in known_columns:
                            known_columns += [key]
        return known_columns

    def get_candidate_count(self):
        """Iterate all score groups and candidates and return the total number of candidates

        Parameters
        ----------

        score_group_continer : list
            List of score groups


        Returns
        -------

        int

        """

        candidate_count = 0
        for i in range(len(self)):
            candidate_count += len(self[i].candidates)
        return candidate_count

    def collect_features(self):
        """Iterate all score groups and candidates and return a numpy array of all features

        Parameters
        ----------

        score_group_continer : list
            List of score groups

        Returns
        -------

        np.array
            Array of features

        np.array
            Array of precursor indices

        list
            List of feature names

        """

        feature_columns = self.get_feature_columns()
        candidate_count = self.get_candidate_count()

        feature_array = np.empty((candidate_count, NUM_FEATURES), dtype=np.float32)
        feature_array[:] = np.nan

        precursor_idx_array = np.zeros(candidate_count, dtype=np.uint32)

        rank_array = np.zeros(candidate_count, dtype=np.uint8)

        candidate_idx = 0
        for i in range(len(self)):
            for j in range(len(self[i].candidates)):
                candidate = self[i].candidates[j]

                feature_array[candidate_idx] = candidate.feature_array
                # candidate.feature_array = np.empty(0, dtype=np.float32)

                precursor_idx_array[candidate_idx] = candidate.precursor_idx
                rank_array[candidate_idx] = candidate.rank
                candidate_idx += 1

        return feature_array, precursor_idx_array, rank_array, feature_columns

    def get_fragment_count(self):
        """Iterate all score groups and candidates and return the total number of fragments

        Parameters
        ----------

        Returns
        -------

        int
            Number of fragments in the score group container

        """

        fragment_count = 0
        for i in range(len(self)):
            for j in range(len(self[i].candidates)):
                if CalibCols.MZ_LIBRARY in self[i].candidates[j].fragment_feature_dict:
                    fragment_count += len(
                        self[i]
                        .candidates[j]
                        .fragment_feature_dict[CalibCols.MZ_LIBRARY]
                    )
        return fragment_count

    def collect_fragments(self):
        """Iterate all score groups and candidates and accumulate the fragment-level data in a single array

        Parameters
        ----------

        Returns
        -------

        list

        """

        fragment_columns = [
            CalibCols.MZ_LIBRARY,
            CalibCols.MZ_OBSERVED,
            "mass_error",
            "height",
            "intensity",
        ]
        fragment_count = self.get_fragment_count()
        fragment_array = np.zeros((fragment_count, len(fragment_columns)))
        fragment_array[:] = np.nan

        precursor_idx_array = np.zeros(fragment_count, dtype=np.uint32)

        rank_array = np.zeros(fragment_count, dtype=np.uint8)

        fragment_start_idx = 0

        # iterate all score groups and candidates
        for i in range(len(self)):
            for j in range(len(self[i].candidates)):
                candidate = self[i].candidates[j]

                # if the candidate has fragments, add them to the array
                if CalibCols.MZ_LIBRARY in candidate.fragment_feature_dict:
                    candidate_fragment_count = len(
                        candidate.fragment_feature_dict[CalibCols.MZ_LIBRARY]
                    )
                    for k, col in enumerate(fragment_columns):
                        fragment_array[
                            fragment_start_idx : fragment_start_idx
                            + candidate_fragment_count,
                            k,
                        ] = candidate.fragment_feature_dict[col]
                        precursor_idx_array[
                            fragment_start_idx : fragment_start_idx
                            + candidate_fragment_count
                        ] = candidate.precursor_idx
                        rank_array[
                            fragment_start_idx : fragment_start_idx
                            + candidate_fragment_count
                        ] = candidate.rank

                    fragment_start_idx += candidate_fragment_count

        return fragment_array, precursor_idx_array, rank_array, fragment_columns


# TODO: why is this necessary?
ScoreGroupContainer.__module__ = "alphatims.extraction.plexscoring"

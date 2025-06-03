"""Data Containers for Candidate Scoring."""

# native imports
import logging

# alpha family imports
import numba as nb
import numpy as np

# third party imports
# alphadia imports
from alphadia import quadrupole
from alphadia.constants.settings import NUM_FEATURES
from alphadia.numba import config, fragments
from alphadia.plexscoring.features_ import features
from alphadia.plotting.cycle import plot_cycle
from alphadia.plotting.debug import (
    plot_fragment_profile,
    plot_fragments,
    plot_precursor,
    plot_template,
)

logger = logging.getLogger()


@nb.experimental.jitclass()
class Candidate:
    """
    __init__ will be called single threaded, initialize will later be called multithreaded.
    Therefore as much as possible should be done in initialize.

    """

    failed: nb.boolean

    output_idx: nb.uint32

    # input columns
    precursor_idx: nb.uint32
    channel: nb.uint8
    rank: nb.uint8

    frag_start_idx: nb.uint32
    frag_stop_idx: nb.uint32

    scan_start: nb.int64
    scan_stop: nb.int64
    scan_center: nb.int64
    frame_start: nb.int64
    frame_stop: nb.int64
    frame_center: nb.int64

    charge: nb.uint8
    precursor_mz: nb.float32
    isotope_intensity: nb.float32[::1]

    # calculated properties
    isotope_mz: nb.float32[::1]

    # object properties
    fragments: fragments.FragmentContainer.class_type.instance_type

    fragment_feature_dict: nb.types.DictType(nb.types.unicode_type, nb.float32[:])

    feature_array: nb.float32[::1]

    dense_fragments: nb.float32[:, :, :, :, ::1]
    dense_precursors: nb.float32[:, :, :, :, ::1]

    fragments_frame_profile: nb.float32[:, :, ::1]
    fragments_scan_profile: nb.float32[:, :, ::1]

    template_frame_profile: nb.float32[:, ::1]
    template_scan_profile: nb.float32[:, ::1]

    observation_importance: nb.float32[::1]
    template: nb.float32[:, :, ::1]

    def __init__(
        self,
        output_idx: nb.uint32,
        precursor_idx: nb.uint32,
        channel: nb.uint8,
        rank: nb.uint8,
        frag_start_idx: nb.uint32,
        frag_stop_idx: nb.uint32,
        scan_start: nb.int64,
        scan_stop: nb.int64,
        scan_center: nb.int64,
        frame_start: nb.int64,
        frame_stop: nb.int64,
        frame_center: nb.int64,
        charge: nb.uint8,
        precursor_mz: nb.float32,
        isotope_intensity: nb.float32[::1],
    ) -> None:
        self.output_idx = output_idx

        self.precursor_idx = precursor_idx
        self.channel = channel
        self.rank = rank

        self.frag_start_idx = frag_start_idx
        self.frag_stop_idx = frag_stop_idx

        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.scan_center = scan_center
        self.frame_start = frame_start
        self.frame_stop = frame_stop
        self.frame_center = frame_center

        self.charge = charge
        self.precursor_mz = precursor_mz
        self.isotope_intensity = isotope_intensity

        self.failed = False

    def __str__(self):
        string = "Candidate Object ("
        string += "precursor_idx: " + str(self.precursor_idx)
        string += ", channel: " + str(self.channel) + ")"
        return string

    def initialize(self, fragment_container, config):
        # initialize all required dicts
        # accessing uninitialized dicts in numba will result in a kernel crash :)

        self.fragment_feature_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=float_array
        )

        self.assemble_isotope_mz(config)

        self.assemble_isotope_mz(config)

    def assemble_isotope_mz(self, config):
        """
        Assemble the isotope m/z values from the precursor m/z and the isotope
        offsets.
        """
        n_isotopes = min(self.isotope_intensity.shape[0], config.top_k_isotopes)
        self.isotope_intensity = self.isotope_intensity[:n_isotopes]
        offset = (
            np.arange(self.isotope_intensity.shape[0])
            * 1.0033548350700006
            / self.charge
        )
        return offset.astype(nb.float32) + self.precursor_mz

    def process(
        self,
        jit_data,
        psm_proto_df,
        fragment_container,
        config,
        quadrupole_calibration,
        debug,
    ) -> None:
        psm_proto_df.precursor_idx[self.output_idx] = self.precursor_idx
        psm_proto_df.rank[self.output_idx] = self.rank

        isotope_mz = self.assemble_isotope_mz(config)

        # build fragment container
        fragments = fragment_container.slice(
            np.array([[self.frag_start_idx, self.frag_stop_idx, 1]])
        )
        if config.exclude_shared_ions:
            fragments.filter_by_cardinality(1)

        fragments.filter_top_k(config.top_k_fragments)
        fragments.sort_by_mz()

        if len(fragments.mz) <= 3:
            self.failed = True
            return

        if debug:
            print("precursor", self.precursor_idx, "channel", self.channel)

        frame_limit = np.array(
            [[self.frame_start, self.frame_stop, 1]], dtype=np.uint64
        )

        scan_limit = np.array([[self.scan_start, self.scan_stop, 1]], dtype=np.uint64)

        quadrupole_limit = np.array(
            [[np.min(isotope_mz) - 0.5, np.max(isotope_mz) + 0.5]], dtype=np.float32
        )

        if debug:
            self.visualize_window(
                quadrupole_calibration.cycle_calibrated,
                quadrupole_limit[0, 0],
                quadrupole_limit[0, 1],
                self.scan_start,
                self.scan_stop,
            )

        dense_fragments, frag_precursor_index = jit_data.get_dense(
            frame_limit,
            scan_limit,
            fragments.mz,
            config.fragment_mz_tolerance,
            quadrupole_limit,
            absolute_masses=True,
        )

        # DEBUG only used for debugging
        # self.dense_fragments = dense_fragments

        # check if an empty array is returned
        # scan and quadrupole limits of the fragments candidate are outside the acquisition range
        if dense_fragments.shape[-1] == 0:
            self.failed = True
            return

        # only one fragment is found
        if dense_fragments.shape[1] <= 1:
            self.failed = True
            return

        _dense_precursors, prec_precursor_index = jit_data.get_dense(
            frame_limit,
            scan_limit,
            isotope_mz,
            config.precursor_mz_tolerance,
            np.array([[-1.0, -1.0]]),
            absolute_masses=True,
        )

        # sum precursors to remove multiple observations
        dense_precursors = np.zeros(
            (
                2,
                _dense_precursors.shape[1],
                1,
                _dense_precursors.shape[3],
                _dense_precursors.shape[4],
            ),
            dtype=np.float32,
        )
        for i in range(_dense_precursors.shape[1]):
            dense_precursors[0, i, 0] = np.sum(_dense_precursors[0, i], axis=0)
            for k in range(_dense_precursors.shape[3]):
                for ll in range(_dense_precursors.shape[4]):
                    sum = 0
                    count = 0
                    for j in range(_dense_precursors.shape[2]):
                        sum += _dense_precursors[1, i, j, k, ll]
                        if _dense_precursors[1, i, j, k, ll] > 0:
                            count += 1
                    dense_precursors[1, i, 0, k, ll] = sum / (count + 1e-6)

        # DEBUG only used for debugging
        # self.dense_precursors = dense_precursors

        if debug:
            # self.visualize_precursor(dense_precursors)
            self.visualize_fragments(dense_fragments, fragments)

        # (n_isotopes, n_observations, n_scans)
        qtf = quadrupole.quadrupole_transfer_function_single(
            quadrupole_calibration,
            frag_precursor_index,
            np.arange(int(self.scan_start), int(self.scan_stop)),
            isotope_mz,
        )

        # mask fragments by qtf
        qtf_mask = np.reshape(
            np.sum(qtf, axis=0) / qtf.shape[0], (1, qtf.shape[1], qtf.shape[2], 1)
        ).astype(np.float32)
        dense_fragments[0] = dense_fragments[0] * qtf_mask

        # (n_observation, n_scans, n_frames)
        template = quadrupole.calculate_template_single(
            qtf, dense_precursors, self.isotope_intensity
        )

        if debug:
            self.visualize_template(
                dense_precursors, qtf, template, self.isotope_intensity
            )

        observation_importance = quadrupole.calculate_observation_importance_single(
            template,
        )

        self.observation_importance = observation_importance

        # DEBUG only used for debugging
        # self.template = template

        if dense_fragments.shape[0] == 0:
            self.failed = True
            return

        if dense_precursors.shape[0] == 0:
            self.failed = True
            return

        fragment_mask_1d = (
            np.sum(np.sum(np.sum(dense_fragments[0], axis=-1), axis=-1), axis=-1) > 0
        )

        if np.sum(fragment_mask_1d) < 2:
            self.failed = True
            return

        # (2, n_valid_fragments, n_observations, n_scans, n_frames)
        dense_fragments = dense_fragments[:, fragment_mask_1d]
        fragments.apply_mask(fragment_mask_1d)

        # (n_fragments, n_observations, n_frames)

        fragments_frame_profile = features.frame_profile_2d(dense_fragments[0])
        # features.center_envelope(fragments_frame_profile)

        cycle_len = jit_data.cycle.shape[1]

        frame_rt = jit_data.rt_values[self.frame_start : self.frame_stop : cycle_len]

        # (n_observations, n_frames)
        template_frame_profile = features.or_envelope_1d(
            features.frame_profile_1d(template)
        )

        # (n_fragments, n_observations, n_scans)
        fragments_scan_profile = features.or_envelope_2d(
            features.scan_profile_2d(dense_fragments[0])
        )

        # (n_observations, n_scans)
        template_scan_profile = features.or_envelope_1d(
            features.scan_profile_1d(template)
        )

        if debug:
            self.visualize_profiles(
                template,
                fragments_scan_profile,
                fragments_frame_profile,
                template_frame_profile,
                template_scan_profile,
                jit_data.has_mobility,
            )

        # from here on features are being accumulated in the feature_array
        # (n_features)
        feature_array = np.zeros(NUM_FEATURES, dtype=np.float32)
        feature_array[28] = np.mean(fragment_mask_1d)

        # works
        features.location_features(
            jit_data,
            self.scan_start,
            self.scan_stop,
            self.scan_center,
            self.frame_start,
            self.frame_stop,
            self.frame_center,
            feature_array,
        )

        features.precursor_features(
            isotope_mz,
            self.isotope_intensity,
            dense_precursors,
            observation_importance,
            template,
            feature_array,
        )
        # works

        # retrive first fragment features
        # (n_valid_fragments)

        mz_observed, mass_error, height, intensity = features.fragment_features(
            dense_fragments,
            fragments_frame_profile,
            frame_rt,
            observation_importance,
            template,
            fragments,
            feature_array,
            quant_window=config.quant_window,
            quant_all=config.quant_all,
        )

        # store fragment features if requested
        # only target precursors are stored
        if config.collect_fragments:
            psm_proto_df.fragment_precursor_idx[self.output_idx, : len(mz_observed)] = [
                self.precursor_idx
            ] * len(mz_observed)
            psm_proto_df.fragment_rank[self.output_idx, : len(mz_observed)] = [
                self.rank
            ] * len(mz_observed)
            psm_proto_df.fragment_mz_library[
                self.output_idx, : len(fragments.mz_library)
            ] = fragments.mz_library
            psm_proto_df.fragment_mz[self.output_idx, : len(fragments.mz)] = (
                fragments.mz
            )
            psm_proto_df.fragment_mz_observed[self.output_idx, : len(mz_observed)] = (
                mz_observed
            )

            psm_proto_df.fragment_height[self.output_idx, : len(height)] = height
            psm_proto_df.fragment_intensity[self.output_idx, : len(intensity)] = (
                intensity
            )

            psm_proto_df.fragment_mass_error[self.output_idx, : len(mass_error)] = (
                mass_error
            )
            psm_proto_df.fragment_position[
                self.output_idx, : len(fragments.position)
            ] = fragments.position
            psm_proto_df.fragment_number[self.output_idx, : len(fragments.number)] = (
                fragments.number
            )
            psm_proto_df.fragment_type[self.output_idx, : len(fragments.type)] = (
                fragments.type
            )
            psm_proto_df.fragment_charge[self.output_idx, : len(fragments.charge)] = (
                fragments.charge
            )
            psm_proto_df.fragment_loss_type[
                self.output_idx, : len(fragments.loss_type)
            ] = fragments.loss_type

        # ============= FRAGMENT MOBILITY CORRELATIONS =============
        # will be skipped if no mobility dimension is present
        if jit_data.has_mobility:
            (
                feature_array[29],
                feature_array[30],
            ) = features.fragment_mobility_correlation(
                fragments_scan_profile,
                template_scan_profile,
                observation_importance,
                fragments.intensity,
            )

        # (n_valid_fragments)
        correlation = features.profile_features(
            jit_data,
            fragments.intensity,
            fragments.type,
            observation_importance,
            fragments_scan_profile,
            fragments_frame_profile,
            template_scan_profile,
            template_frame_profile,
            self.scan_start,
            self.scan_stop,
            self.frame_start,
            self.frame_stop,
            feature_array,
            config.experimental_xic,
        )

        if config.collect_fragments:
            psm_proto_df.fragment_correlation[self.output_idx, : len(correlation)] = (
                correlation
            )

        psm_proto_df.features[self.output_idx] = feature_array
        psm_proto_df.valid[self.output_idx] = True

    def process_reference_channel(self, reference_candidate, fragment_container):
        fragments = fragment_container.slice(
            np.array([[self.frag_start_idx, self.frag_stop_idx, 1]])
        )
        if config.exclude_shared_ions:
            fragments.filter_by_cardinality(1)

        fragments.filter_top_k(config.top_k_fragments)
        fragments.sort_by_mz()

        self.features.update(
            features.reference_features(
                reference_candidate.observation_importance,
                reference_candidate.fragments_scan_profile,
                reference_candidate.fragments_frame_profile,
                reference_candidate.template_scan_profile,
                reference_candidate.template_frame_profile,
                self.observation_importance,
                self.fragments_scan_profile,
                self.fragments_frame_profile,
                self.template_scan_profile,
                self.template_frame_profile,
                fragments.intensity,
            )
        )

    def visualize_window(self, *args):
        with nb.objmode:
            plot_cycle(*args)

    def visualize_precursor(self, *args):
        with nb.objmode:
            plot_precursor(*args)

    def visualize_fragments(self, *args):
        with nb.objmode:
            plot_fragments(*args)

    def visualize_template(self, *args):
        with nb.objmode:
            plot_template(*args)

    def visualize_profiles(self, *args):
        with nb.objmode:
            plot_fragment_profile(*args)


candidate_type = Candidate.class_type.instance_type


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
        if config.reference_channel >= 0:
            for idx, _ in enumerate(self.candidates):
                if idx == reference_channel_idx:
                    continue
                # candidate.process_reference_channel(
                #    self.candidates[reference_channel_idx]
                # )

                # update rank features
                # candidate.features.update(
                #    features.rank_features(idx, self.candidates)
                # )


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
                if "mz_library" in self[i].candidates[j].fragment_feature_dict:
                    fragment_count += len(
                        self[i].candidates[j].fragment_feature_dict["mz_library"]
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
            "mz_library",
            "mz_observed",
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
                if "mz_library" in candidate.fragment_feature_dict:
                    candidate_fragment_count = len(
                        candidate.fragment_feature_dict["mz_library"]
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


ScoreGroupContainer.__module__ = "alphatims.extraction.plexscoring"

float_array = nb.types.float32[:]

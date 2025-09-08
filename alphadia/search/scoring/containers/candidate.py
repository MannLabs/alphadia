"""Data Containers for Candidate Scoring."""

import logging

import numba as nb
import numpy as np

from alphadia.constants.settings import NUM_FEATURES
from alphadia.search.jitclasses.fragment_container import FragmentContainer
from alphadia.search.scoring import quadrupole
from alphadia.search.scoring.features.fragment_features import (
    fragment_features,
    fragment_mobility_correlation,
)
from alphadia.search.scoring.features.location_features import location_features
from alphadia.search.scoring.features.precursor_features import precursor_features
from alphadia.search.scoring.features.profile_features import profile_features
from alphadia.search.scoring.plotting.cycle import plot_cycle
from alphadia.search.scoring.plotting.debug import (
    plot_fragment_profile,
    plot_fragments,
    plot_precursor,
    plot_template,
)
from alphadia.search.scoring.utils import (
    frame_profile_1d,
    frame_profile_2d,
    or_envelope_1d,
    or_envelope_2d,
    scan_profile_1d,
    scan_profile_2d,
)

logger = logging.getLogger()


float_array = nb.types.float32[:]


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
    fragments: FragmentContainer.class_type.instance_type

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

    # TODO this needs to be split! also, this is a lot of logic for a "container"
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

        fragments_frame_profile = frame_profile_2d(dense_fragments[0])
        # features.center_envelope(fragments_frame_profile)

        cycle_len = jit_data.cycle.shape[1]

        frame_rt = jit_data.rt_values[self.frame_start : self.frame_stop : cycle_len]

        # (n_observations, n_frames)
        template_frame_profile = or_envelope_1d(frame_profile_1d(template))

        # (n_fragments, n_observations, n_scans)
        fragments_scan_profile = or_envelope_2d(scan_profile_2d(dense_fragments[0]))

        # (n_observations, n_scans)
        template_scan_profile = or_envelope_1d(scan_profile_1d(template))

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
        location_features(
            jit_data,
            self.scan_start,
            self.scan_stop,
            self.scan_center,
            self.frame_start,
            self.frame_stop,
            self.frame_center,
            feature_array,
        )

        precursor_features(
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

        mz_observed, mass_error, height, intensity = fragment_features(
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
            ) = fragment_mobility_correlation(
                fragments_scan_profile,
                template_scan_profile,
                observation_importance,
                fragments.intensity,
            )

        # (n_valid_fragments)
        correlation = profile_features(
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

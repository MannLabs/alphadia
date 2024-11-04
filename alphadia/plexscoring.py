# native imports
import logging

# alpha family imports
import alphatims.utils
import numba as nb
import numpy as np

# third party imports
import pandas as pd

# alphadia imports
from alphadia import features, quadrupole, utils, validate
from alphadia.data import alpharaw, bruker
from alphadia.numba import config, fragments
from alphadia.plotting.cycle import plot_cycle
from alphadia.plotting.debug import (
    plot_fragment_profile,
    plot_fragments,
    plot_precursor,
    plot_template,
)

logger = logging.getLogger()

NUM_FEATURES = 46


def candidate_features_to_candidates(
    candidate_features_df: pd.DataFrame,
    optional_columns: list[str] | None = None,
):
    """create candidates_df from candidate_features_df

    Parameters
    ----------

    candidate_features_df : pd.DataFrame
        candidate_features_df

    Returns
    -------

    candidate_df : pd.DataFrame
        candidates_df
    """

    # validate candidate_features_df input
    if optional_columns is None:
        optional_columns = ["proba"]
    validate.candidate_features_df(candidate_features_df.copy())

    required_columns = [
        "elution_group_idx",
        "precursor_idx",
        "rank",
        "scan_start",
        "scan_stop",
        "scan_center",
        "frame_start",
        "frame_stop",
        "frame_center",
    ]

    # select required columns
    candidate_df = candidate_features_df[required_columns + optional_columns].copy()
    # validate candidate_df output
    validate.candidates_df(candidate_df)

    return candidate_df


def multiplex_candidates(
    candidates_df: pd.DataFrame,
    precursors_flat_df: pd.DataFrame,
    remove_decoys: bool = True,
    channels: list[int] | None = None,
):
    """Takes a candidates dataframe and a precursors dataframe and returns a multiplexed candidates dataframe.
    All original candidates will be retained. For missing candidates, the best scoring candidate in the elution group will be used and multiplexed across all missing channels.

    Parameters
    ----------

    candidates_df : pd.DataFrame
        Candidates dataframe as returned by `hybridselection.HybridCandidateSelection`

    precursors_flat_df : pd.DataFrame
        Precursors dataframe

    remove_decoys : bool, optional
        If True, remove decoys from the precursors dataframe, by default True

    channels : typing.List[int], optional
        List of channels to include in the multiplexed candidates dataframe, by default [0,4,8,12]

    Returns
    -------

    pd.DataFrame
        Multiplexed candidates dataframe

    """
    if channels is None:
        channels = [0, 4, 8, 12]
    precursors_flat_view = precursors_flat_df.copy()
    best_candidate_view = candidates_df.copy()

    validate.precursors_flat(precursors_flat_view)
    validate.candidates_df(best_candidate_view)

    # remove decoys if requested
    if remove_decoys:
        precursors_flat_view = precursors_flat_df[precursors_flat_df["decoy"] == 0]
        if "decoy" in best_candidate_view.columns:
            best_candidate_view = best_candidate_view[best_candidate_view["decoy"] == 0]

    # original precursors are forbidden as they will be concatenated in the end
    # the candidate used for multiplexing is the best scoring candidate in each elution group
    best_candidate_view = (
        best_candidate_view.sort_values("proba")
        .groupby("elution_group_idx")
        .first()
        .reset_index()
    )

    # get all candidate elution group
    candidate_elution_group_idxs = best_candidate_view["elution_group_idx"].unique()

    # restrict precursors to channels and candidate elution groups
    precursors_flat_view = precursors_flat_view[
        precursors_flat_view["channel"].isin(channels)
    ]

    precursors_flat_view = precursors_flat_view[
        precursors_flat_view["elution_group_idx"].isin(candidate_elution_group_idxs)
    ]
    # remove original precursors
    precursors_flat_view = precursors_flat_view[
        ["elution_group_idx", "precursor_idx", "channel"]
    ]
    # reduce precursors to the elution group level
    best_candidate_view = best_candidate_view.drop(columns=["precursor_idx"])
    if "channel" in best_candidate_view.columns:
        best_candidate_view = best_candidate_view.drop(columns=["channel"])

    # merge candidates and precursors
    multiplexed_candidates_df = precursors_flat_view.merge(
        best_candidate_view, on="elution_group_idx", how="left"
    )

    # append original candidates
    # multiplexed_candidates_df = pd.concat([multiplexed_candidates_df, candidates_view]).sort_values('precursor_idx')
    validate.candidates_df(multiplexed_candidates_df)

    return multiplexed_candidates_df


@nb.experimental.jitclass()
class CandidateConfigJIT:
    collect_fragments: nb.boolean
    score_grouped: nb.boolean
    exclude_shared_ions: nb.boolean
    top_k_fragments: nb.uint32
    top_k_isotopes: nb.uint32
    reference_channel: nb.int16
    quant_window: nb.uint32
    quant_all: nb.boolean

    precursor_mz_tolerance: nb.float32
    fragment_mz_tolerance: nb.float32

    def __init__(
        self,
        collect_fragments: nb.boolean,
        score_grouped: nb.boolean,
        exclude_shared_ions: nb.types.bool_,
        top_k_fragments: nb.uint32,
        top_k_isotopes: nb.uint32,
        reference_channel: nb.int16,
        quant_window: nb.uint32,
        quant_all: nb.boolean,
        precursor_mz_tolerance: nb.float32,
        fragment_mz_tolerance: nb.float32,
    ) -> None:
        """Numba JIT compatible config object for CandidateScoring.
        Will be emitted when `CandidateConfig.jitclass()` is called.

        Please refer to :class:`.alphadia.plexscoring.CandidateConfig` for documentation.
        """

        self.collect_fragments = collect_fragments
        self.score_grouped = score_grouped
        self.exclude_shared_ions = exclude_shared_ions
        self.top_k_fragments = top_k_fragments
        self.top_k_isotopes = top_k_isotopes
        self.reference_channel = reference_channel
        self.quant_window = quant_window
        self.quant_all = quant_all

        self.precursor_mz_tolerance = precursor_mz_tolerance
        self.fragment_mz_tolerance = fragment_mz_tolerance


candidate_config_type = CandidateConfigJIT.class_type.instance_type


class CandidateConfig(config.JITConfig):
    """Config object for CandidateScoring."""

    def __init__(self):
        """Create default config for CandidateScoring"""
        self.collect_fragments = True
        self.score_grouped = False
        self.exclude_shared_ions = True
        self.top_k_fragments = 12
        self.top_k_isotopes = 4
        self.reference_channel = -1
        self.quant_window = 3
        self.quant_all = False
        self.precursor_mz_tolerance = 15
        self.fragment_mz_tolerance = 15

    @property
    def jit_container(self):
        """The numba jitclass for this config object."""
        return CandidateConfigJIT

    @property
    def collect_fragments(self) -> bool:
        """Collect fragment features.
        Default: `collect_fragments = False`"""
        return self._collect_fragments

    @collect_fragments.setter
    def collect_fragments(self, value):
        self._collect_fragments = value

    @property
    def score_grouped(self) -> bool:
        """When multiplexing is used, some grouped features are calculated taking into account all channels.
        Default: `score_grouped = False`"""
        return self._score_grouped

    @score_grouped.setter
    def score_grouped(self, value):
        self._score_grouped = value

    @property
    def exclude_shared_ions(self) -> int:
        """When multiplexing is used, some fragments are shared for the same peptide with different labels.
        This setting removes fragments who are shared by more than one channel.
        Default: `exclude_shared_ions = True`"""
        return self._exclude_shared_ions

    @exclude_shared_ions.setter
    def exclude_shared_ions(self, value):
        self._exclude_shared_ions = value

    @property
    def top_k_fragments(self) -> int:
        """The number of fragments to consider for scoring. The top_k_fragments most intense fragments are used.
        Default: `top_k_fragments = 12`"""
        return self._top_k_fragments

    @top_k_fragments.setter
    def top_k_fragments(self, value):
        self._top_k_fragments = value

    @property
    def top_k_isotopes(self) -> int:
        """The number of precursor isotopes to consider for scoring. The first top_k_isotopes most intense isotopes are used.
        Default: `top_k_isotopes = 4`"""
        return self._top_k_isotopes

    @top_k_isotopes.setter
    def top_k_isotopes(self, value):
        self._top_k_isotopes = value

    @property
    def reference_channel(self) -> int:
        """When multiplexing is being used, a reference channel can be defined for calculating reference channel deopendent features.
        The channel information is used as defined in the `channel` column in the precursor dataframe. If set to -1, no reference channel is used.
        Default: `reference_channel = -1`"""
        return self._reference_channel

    @reference_channel.setter
    def reference_channel(self, value):
        self._reference_channel = value

    @property
    def quant_window(self) -> int:
        """The quantification window size in cycles.
        the area will be calculated from `scan_center - quant_window` to `scan_center + quant_window`.
        Default: `quant_window = 3`"""
        return self._quant_window

    @quant_window.setter
    def quant_window(self, value):
        self._quant_window = value

    @property
    def quant_all(self) -> bool:
        """Quantify all fragments in the quantification window.
        Default: `quant_all = False`"""
        return self._quant_all

    @quant_all.setter
    def quant_all(self, value):
        self._quant_all = value

    @property
    def precursor_mz_tolerance(self) -> float:
        """The precursor m/z tolerance in ppm.
        Default: `precursor_mz_tolerance = 10`"""
        return self._precursor_mz_tolerance

    @precursor_mz_tolerance.setter
    def precursor_mz_tolerance(self, value):
        self._precursor_mz_tolerance = value

    @property
    def fragment_mz_tolerance(self) -> float:
        """The fragment m/z tolerance in ppm.
        Default: `fragment_mz_tolerance = 15`"""
        return self._fragment_mz_tolerance

    @fragment_mz_tolerance.setter
    def fragment_mz_tolerance(self, value):
        self._fragment_mz_tolerance = value

    def validate(self):
        """Validate all properties of the config object.
        Should be called whenever a property is changed."""

        assert isinstance(self.score_grouped, bool), "score_grouped must be a boolean"
        assert self.top_k_fragments > 0, "top_k_fragments must be greater than 0"
        assert self.top_k_isotopes > 0, "top_k_isotopes must be greater than 0"
        assert (
            self.reference_channel >= -1
        ), "reference_channel must be greater than or equal to -1"
        # assert not (self.score_grouped == True and self.reference_channel == -1), 'for grouped scoring, reference_channel must be set to a valid channel'

        assert (
            self.precursor_mz_tolerance >= 0
        ), "precursor_mz_tolerance must be greater than or equal to 0"
        assert (
            self.precursor_mz_tolerance < 200
        ), "precursor_mz_tolerance must be less than 200"
        assert (
            self.fragment_mz_tolerance >= 0
        ), "fragment_mz_tolerance must be greater than or equal to 0"
        assert (
            self.fragment_mz_tolerance < 200
        ), "fragment_mz_tolerance must be less than 200"

    def copy(self):
        """Create a copy of the config object."""
        return CandidateConfig.from_dict(self.to_dict())


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


@nb.experimental.jitclass()
class OuptutPsmDF:
    valid: nb.boolean[::1]
    precursor_idx: nb.uint32[::1]
    rank: nb.uint8[::1]

    features: nb.float32[:, ::1]

    fragment_precursor_idx: nb.uint32[:, ::1]
    fragment_rank: nb.uint8[:, ::1]

    fragment_mz_library: nb.float32[:, ::1]
    fragment_mz: nb.float32[:, ::1]
    fragment_mz_observed: nb.float32[:, ::1]

    fragment_height: nb.float32[:, ::1]
    fragment_intensity: nb.float32[:, ::1]

    fragment_mass_error: nb.float32[:, ::1]
    fragment_correlation: nb.float32[:, ::1]

    fragment_position: nb.uint8[:, ::1]
    fragment_number: nb.uint8[:, ::1]
    fragment_type: nb.uint8[:, ::1]
    fragment_charge: nb.uint8[:, ::1]

    def __init__(self, n_psm, top_k_fragments):
        self.valid = np.zeros(n_psm, dtype=np.bool_)
        self.precursor_idx = np.zeros(n_psm, dtype=np.uint32)
        self.rank = np.zeros(n_psm, dtype=np.uint8)

        self.features = np.zeros((n_psm, NUM_FEATURES), dtype=np.float32)

        self.fragment_precursor_idx = np.zeros(
            (n_psm, top_k_fragments), dtype=np.uint32
        )
        self.fragment_rank = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)

        self.fragment_mz_library = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_mz = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_mz_observed = np.zeros((n_psm, top_k_fragments), dtype=np.float32)

        self.fragment_height = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_intensity = np.zeros((n_psm, top_k_fragments), dtype=np.float32)

        self.fragment_mass_error = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_correlation = np.zeros((n_psm, top_k_fragments), dtype=np.float32)

        self.fragment_position = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)
        self.fragment_number = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)
        self.fragment_type = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)
        self.fragment_charge = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)

    def to_fragment_df(self):
        mask = self.fragment_mz_library.flatten() > 0

        return (
            self.fragment_precursor_idx.flatten()[mask],
            self.fragment_rank.flatten()[mask],
            self.fragment_mz_library.flatten()[mask],
            self.fragment_mz.flatten()[mask],
            self.fragment_mz_observed.flatten()[mask],
            self.fragment_height.flatten()[mask],
            self.fragment_intensity.flatten()[mask],
            self.fragment_mass_error.flatten()[mask],
            self.fragment_correlation.flatten()[mask],
            self.fragment_position.flatten()[mask],
            self.fragment_number.flatten()[mask],
            self.fragment_type.flatten()[mask],
            self.fragment_charge.flatten()[mask],
        )

    def to_precursor_df(self):
        return (
            self.precursor_idx[self.valid],
            self.rank[self.valid],
            self.features[self.valid],
        )


@alphatims.utils.pjit()
def _executor(
    i,
    sg_container,
    psm_proto_df,
    fragment_container,
    jit_data,
    config,
    quadrupole_calibration,
    debug,
):
    """
    Helper function.
    Is decorated with alphatims.utils.pjit to enable parallel execution of HybridElutionGroup.process.
    """

    sg_container[i].process(
        psm_proto_df,
        fragment_container,
        jit_data,
        config,
        quadrupole_calibration,
        debug,
    )


@alphatims.utils.pjit()
def transfer_feature(
    idx, score_group_container, feature_array, precursor_idx_array, rank_array
):
    feature_array[idx] = score_group_container[idx].candidates[0].feature_array
    precursor_idx_array[idx] = score_group_container[idx].candidates[0].precursor_idx
    rank_array[idx] = score_group_container[idx].candidates[0].rank


class CandidateScoring:
    """Calculate features for each precursor candidate used in scoring."""

    def __init__(
        self,
        dia_data: bruker.TimsTOFTransposeJIT | alpharaw.AlphaRawJIT,
        precursors_flat: pd.DataFrame,
        fragments_flat: pd.DataFrame,
        quadrupole_calibration: quadrupole.SimpleQuadrupole = None,
        config: CandidateConfig = None,
        rt_column: str = "rt_library",
        mobility_column: str = "mobility_library",
        precursor_mz_column: str = "mz_library",
        fragment_mz_column: str = "mz_library",
    ):
        """Initialize candidate scoring step.
        The features can then be used for scoring, calibration and quantification.

        Parameters
        ----------

        dia_data : data.TimsTOFTransposeJIT
            The raw mass spec data as a TimsTOFTransposeJIT object.

        precursors_flat : pd.DataFrame
            A DataFrame containing precursor information.
            The DataFrame will be validated by using the `alphadia.validate.precursors_flat` schema.

        fragments_flat : pd.DataFrame
            A DataFrame containing fragment information.
            The DataFrame will be validated by using the `alphadia.validate.fragments_flat` schema.

        quadrupole_calibration : quadrupole.SimpleQuadrupole, default=None
            An object containing the quadrupole calibration information.
            If None, an uncalibrated quadrupole will be used.
            The object musst have a `jit` method which returns a Numba JIT compiled instance of the calibration function.

        config : CandidateConfig, default = None
            A Numba jit compatible object containing the configuration for the candidate scoring.
            If None, the default configuration will be used.

        rt_column : str, default='rt_library'
            The name of the column in `precursors_flat` containing the RT information.
            This property needs to be changed to `rt_calibrated` if the data has been calibrated.

        mobility_column : str, default='mobility_library'
            The name of the column in `precursors_flat` containing the mobility information.
            This property needs to be changed to `mobility_calibrated` if the data has been calibrated.

        precursor_mz_column : str, default='mz_library'
            The name of the column in `precursors_flat` containing the precursor m/z information.
            This property needs to be changed to `mz_calibrated` if the data has been calibrated.

        fragment_mz_column : str, default='mz_library'
            The name of the column in `fragments_flat` containing the fragment m/z information.
            This property needs to be changed to `mz_calibrated` if the data has been calibrated.
        """

        self._dia_data = dia_data

        # validate precursors_flat
        validate.precursors_flat(precursors_flat)
        self.precursors_flat_df = precursors_flat

        # validate fragments_flat
        validate.fragments_flat(fragments_flat)
        self.fragments_flat = fragments_flat

        # check if a valid quadrupole calibration is provided
        if quadrupole_calibration is None:
            self.quadrupole_calibration = quadrupole.SimpleQuadrupole(dia_data.cycle)
        else:
            self.quadrupole_calibration = quadrupole_calibration

        # check if a valid config is provided
        if config is None:
            self.config = CandidateConfig()
        else:
            self.config = config

        self.rt_column = rt_column
        self.mobility_column = mobility_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column

    @property
    def dia_data(self):
        """Get the raw mass spec data as a TimsTOFTransposeJIT object."""
        return self._dia_data

    @property
    def precursors_flat_df(self):
        """Get the DataFrame containing precursor information."""
        return self._precursors_flat_df

    @precursors_flat_df.setter
    def precursors_flat_df(self, precursors_flat_df):
        validate.precursors_flat(precursors_flat_df)
        self._precursors_flat_df = precursors_flat_df.sort_values(by="precursor_idx")

    @property
    def fragments_flat_df(self):
        """Get the DataFrame containing fragment information."""
        return self._fragments_flat

    @fragments_flat_df.setter
    def fragments_flat_df(self, fragments_flat):
        validate.fragments_flat(fragments_flat)
        self._fragments_flat = fragments_flat

    @property
    def quadrupole_calibration(self):
        """Get the quadrupole calibration object."""
        return self._quadrupole_calibration

    @quadrupole_calibration.setter
    def quadrupole_calibration(self, quadrupole_calibration):
        if not hasattr(quadrupole_calibration, "jit"):
            raise AttributeError("quadrupole_calibration must have a jit method")
        self._quadrupole_calibration = quadrupole_calibration

    @property
    def config(self):
        """Get the configuration object."""
        return self._config

    @config.setter
    def config(self, config):
        config.validate()
        self._config = config

    def assemble_score_group_container(
        self, candidates_df: pd.DataFrame
    ) -> ScoreGroupContainer:
        """Assemble the Numba JIT compatible score group container from a candidate dataframe.

        If not present, the `rank` column will be added to the candidate dataframe.
        Then score groups are calculated using :func:`.calculate_score_groups` function.
        If configured in :attr:`.CandidateConfig.score_grouped`, all channels will be grouped into a single score group.
        Otherwise, each channel will be scored separately.

        The candidate dataframe is validated using the :func:`.validate.candidates` schema.

        Parameters
        ----------

        candidates_df : pd.DataFrame
            A DataFrame containing the candidates.

        Returns
        -------

        score_group_container : ScoreGroupContainer
            A Numba JIT compatible score group container.

        """

        precursor_columns = [
            "channel",
            "flat_frag_start_idx",
            "flat_frag_stop_idx",
            "charge",
            "decoy",
            "channel",
            self.precursor_mz_column,
        ] + utils.get_isotope_column_names(self.precursors_flat_df.columns)

        candidates_df = utils.merge_missing_columns(
            candidates_df,
            self.precursors_flat_df,
            precursor_columns,
            on=["precursor_idx"],
            how="left",
        )

        # check if channel column is present
        if "channel" not in candidates_df.columns:
            candidates_df["channel"] = np.zeros(len(candidates_df), dtype=np.uint8)

        # check if monoisotopic abundance column is present
        if "i_0" not in candidates_df.columns:
            candidates_df["i_0"] = np.ones(len(candidates_df), dtype=np.float32)

        # calculate score groups
        candidates_df = utils.calculate_score_groups(
            candidates_df, group_channels=self.config.score_grouped
        )

        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.candidates_df(candidates_df)

        score_group_container = ScoreGroupContainer()
        score_group_container.build_from_df(
            candidates_df["elution_group_idx"].values,
            candidates_df["score_group_idx"].values,
            candidates_df["precursor_idx"].values,
            candidates_df["channel"].values,
            candidates_df["rank"].values,
            candidates_df["flat_frag_start_idx"].values,
            candidates_df["flat_frag_stop_idx"].values,
            candidates_df["scan_start"].values,
            candidates_df["scan_stop"].values,
            candidates_df["scan_center"].values,
            candidates_df["frame_start"].values,
            candidates_df["frame_stop"].values,
            candidates_df["frame_center"].values,
            candidates_df["charge"].values,
            candidates_df[self.precursor_mz_column].values,
            candidates_df[utils.get_isotope_column_names(candidates_df.columns)].values,
        )

        return score_group_container

    def assemble_fragments(self) -> fragments.FragmentContainer:
        """Assemble the Numba JIT compatible fragment container from a fragment dataframe.

        If not present, the `cardinality` column will be added to the fragment dataframe and set to 1.
        Then the fragment dataframe is validated using the :func:`.validate.fragments_flat` schema.

        Returns
        -------

        fragment_container : fragments.FragmentContainer
            A Numba JIT compatible fragment container.
        """

        # set cardinality to 1 if not present
        if "cardinality" in self.fragments_flat.columns:
            pass

        else:
            logger.warning(
                "Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1."
            )
            self.fragments_flat["cardinality"] = np.ones(
                len(self.fragments_flat), dtype=np.uint8
            )

        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.fragments_flat(self.fragments_flat)

        return fragments.FragmentContainer(
            self.fragments_flat["mz_library"].values,
            self.fragments_flat[self.fragment_mz_column].values,
            self.fragments_flat["intensity"].values,
            self.fragments_flat["type"].values,
            self.fragments_flat["loss_type"].values,
            self.fragments_flat["charge"].values,
            self.fragments_flat["number"].values,
            self.fragments_flat["position"].values,
            self.fragments_flat["cardinality"].values,
        )

    def collect_candidates(
        self, candidates_df: pd.DataFrame, psm_proto_df
    ) -> pd.DataFrame:
        """Collect the features from the score group container and return a DataFrame.

        Parameters
        ----------

        score_group_container : ScoreGroupContainer
            A Numba JIT compatible score group container.

        candidates_df : pd.DataFrame
            A DataFrame containing the features for each candidate.

        Returns
        -------

        candidates_psm_df : pd.DataFrame
            A DataFrame containing the features for each candidate.
        """

        feature_columns = [
            "base_width_mobility",
            "base_width_rt",
            "rt_observed",
            "mobility_observed",
            "mono_ms1_intensity",
            "top_ms1_intensity",
            "sum_ms1_intensity",
            "weighted_ms1_intensity",
            "weighted_mass_deviation",
            "weighted_mass_error",
            "mz_observed",
            "mono_ms1_height",
            "top_ms1_height",
            "sum_ms1_height",
            "weighted_ms1_height",
            "isotope_intensity_correlation",
            "isotope_height_correlation",
            "n_observations",
            "intensity_correlation",
            "height_correlation",
            "intensity_fraction",
            "height_fraction",
            "intensity_fraction_weighted",
            "height_fraction_weighted",
            "mean_observation_score",
            "sum_b_ion_intensity",
            "sum_y_ion_intensity",
            "diff_b_y_ion_intensity",
            "f_masked",
            "fragment_scan_correlation",
            "template_scan_correlation",
            "fragment_frame_correlation",
            "top3_frame_correlation",
            "template_frame_correlation",
            "top3_b_ion_correlation",
            "n_b_ions",
            "top3_y_ion_correlation",
            "n_y_ions",
            "cycle_fwhm",
            "mobility_fwhm",
            "delta_frame_peak",
            "top_3_ms2_mass_error",
            "mean_ms2_mass_error",
            "n_overlapping",
            "mean_overlapping_intensity",
            "mean_overlapping_mass_error",
        ]

        precursor_idx, rank, features = psm_proto_df.to_precursor_df()

        df = pd.DataFrame(features, columns=feature_columns)
        df["precursor_idx"] = precursor_idx
        df["rank"] = rank

        # join candidate columns
        candidate_df_columns = [
            "elution_group_idx",
            "scan_center",
            "scan_start",
            "scan_stop",
            "frame_center",
            "frame_start",
            "frame_stop",
            "score",
        ]
        df = utils.merge_missing_columns(
            df,
            candidates_df,
            candidate_df_columns,
            on=["precursor_idx", "rank"],
            how="left",
        )

        # join precursor columns
        precursor_df_columns = [
            "rt_library",
            "mobility_library",
            "mz_library",
            "charge",
            "decoy",
            "channel",
            "flat_frag_start_idx",
            "flat_frag_stop_idx",
            "proteins",
            "genes",
            "sequence",
            "mods",
            "mod_sites",
        ] + utils.get_isotope_column_names(self.precursors_flat_df.columns)

        precursor_df_columns += (
            [self.rt_column] if self.rt_column not in precursor_df_columns else []
        )
        precursor_df_columns += (
            [self.mobility_column]
            if self.mobility_column not in precursor_df_columns
            else []
        )
        precursor_df_columns += (
            [self.precursor_mz_column]
            if self.precursor_mz_column not in precursor_df_columns
            else []
        )

        df = utils.merge_missing_columns(
            df,
            self.precursors_flat_df,
            precursor_df_columns,
            on=["precursor_idx"],
            how="left",
        )

        # calculate delta_rt

        if self.rt_column == "rt_library":
            df["delta_rt"] = df["rt_observed"] - df["rt_library"]
        else:
            df["delta_rt"] = df["rt_observed"] - df[self.rt_column]

        # calculate number of K in sequence
        df["n_K"] = df["sequence"].str.count("K")
        df["n_R"] = df["sequence"].str.count("R")
        df["n_P"] = df["sequence"].str.count("P")

        return df

    def collect_fragments(
        self, candidates_df: pd.DataFrame, psm_proto_df
    ) -> pd.DataFrame:
        """Collect the fragment-level features from the score group container and return a DataFrame.

        Parameters
        ----------

        score_group_container : ScoreGroupContainer
            A Numba JIT compatible score group container.

        candidates_df : pd.DataFrame
            A DataFrame containing the features for each candidate.

        Returns
        -------

        fragment_psm_df : pd.DataFrame
            A DataFrame containing the features for each fragment.

        """

        colnames = [
            "precursor_idx",
            "rank",
            "mz_library",
            "mz",
            "mz_observed",
            "height",
            "intensity",
            "mass_error",
            "correlation",
            "position",
            "number",
            "type",
            "charge",
        ]
        df = pd.DataFrame(
            {
                key: value
                for value, key in zip(
                    psm_proto_df.to_fragment_df(), colnames, strict=True
                )
            }
        )

        # join precursor columns
        precursor_df_columns = [
            "elution_group_idx",
            "decoy",
        ]
        df = utils.merge_missing_columns(
            df,
            self.precursors_flat_df,
            precursor_df_columns,
            on=["precursor_idx"],
            how="left",
        )

        return df

    def __call__(
        self,
        candidates_df,
        thread_count=10,
        debug=False,
        include_decoy_fragment_features=False,
    ):
        """Calculate features for each precursor candidate used for scoring.

        Parameters
        ----------

        candidates_df : pd.DataFrame
            A DataFrame containing the candidates.

        thread_count : int, default=10
            The number of threads to use for parallel processing.

        debug : bool, default=False
            Process only the first 10 elution groups and display full debug information.

        include_decoy_fragment_features : bool, default=False
            Include fragment features for decoy candidates.

        Returns
        -------

        candidate_features_df : pd.DataFrame
            A DataFrame containing the features for each candidate.

        fragment_features_df : pd.DataFrame
            A DataFrame containing the features for each fragment.

        """
        logger.info("Starting candidate scoring")

        fragment_container = self.assemble_fragments()
        validate.candidates_df(candidates_df)

        score_group_container = self.assemble_score_group_container(candidates_df)
        n_candidates = score_group_container.get_candidate_count()
        psm_proto_df = OuptutPsmDF(n_candidates, self.config.top_k_fragments)

        # if debug mode, only iterate over 10 elution groups
        iterator_len = (
            min(10, len(score_group_container)) if debug else len(score_group_container)
        )
        thread_count = 1 if debug else thread_count

        alphatims.utils.set_threads(thread_count)
        _executor(
            range(iterator_len),
            score_group_container,
            psm_proto_df,
            fragment_container,
            self.dia_data,
            self.config.jitclass(),
            self.quadrupole_calibration.jit,
            debug,
        )

        logger.info("Finished candidate processing")
        logger.info("Collecting candidate features")
        candidate_features_df = self.collect_candidates(candidates_df, psm_proto_df)
        validate.candidate_features_df(candidate_features_df)

        logger.info("Collecting fragment features")
        fragment_features_df = self.collect_fragments(candidates_df, psm_proto_df)
        validate.fragment_features_df(fragment_features_df)

        logger.info("Finished candidate scoring")

        del score_group_container
        del fragment_container

        return candidate_features_df, fragment_features_df

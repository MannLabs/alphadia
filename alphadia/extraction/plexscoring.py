from alphadia.extraction import validate, utils, features, plotting, quadrupole, data
from alphadia.extraction.numba import fragments

import alphatims.utils

import pandas as pd
import numpy as np
import numba as nb
import logging

import typing

def candidate_features_to_candidates(
    candidate_features_df : pd.DataFrame,
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
    validate.candidate_features_df(candidate_features_df)

    required_columns = [
        'elution_group_idx',
        'precursor_idx',
        'rank',
        'scan_start',
        'scan_stop',
        'scan_center',
        'frame_start',
        'frame_stop',
        'frame_center'
    ]

    # select required columns
    candidate_df = candidate_features_df[required_columns].copy()

    # validate candidate_df output
    validate.candidates_df(candidate_df)

    return candidate_df

def multiplex_candidates(
    candidates_df: pd.DataFrame,
    precursors_flat_df: pd.DataFrame,
    remove_decoys: bool = True,
    channels: typing.List[int] = [0,4,8,12],
    ):

    """Takes a candidates dataframe and a precursors dataframe and returns a multiplexed candidates dataframe.

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

    validate.candidates_df(candidates_df)
    validate.precursors_flat(precursors_flat_df)

    precursors_flat_view = precursors_flat_df
    candidates_view = candidates_df.copy()

    # remove decoys if requested
    if remove_decoys:
        precursors_flat_view = precursors_flat_df[precursors_flat_df['decoy'] == 0]

    # get all candidate elution group 
    candidate_elution_group_idxs = candidates_view['elution_group_idx'].unique()

    # restrict precursors to channels and candidate elution groups
    precursors_flat_view = precursors_flat_view[precursors_flat_view['channel'].isin(channels)]
    precursors_flat_view = precursors_flat_view[precursors_flat_view['elution_group_idx'].isin(candidate_elution_group_idxs)]
    precursors_flat_view = precursors_flat_view[['elution_group_idx','precursor_idx','channel']]

    # reduce precursors to the elution group level
    candidates_view = candidates_view.drop(columns=['precursor_idx'])

    # merge candidates and precursors
    multiplexed_candidates_df = precursors_flat_view.merge(candidates_view, on='elution_group_idx', how='left')
    validate.candidates_df(multiplexed_candidates_df)

    return multiplexed_candidates_df

    
from alphadia.extraction.numba import config



@nb.experimental.jitclass()
class CandidateConfigJIT:
    
     
    score_grouped: nb.boolean
    max_cardinality: nb.uint8
    top_k_fragments: nb.uint32
    top_k_isotopes: nb.uint32
    reference_channel: nb.int16
    
    precursor_mz_tolerance: nb.float32
    fragment_mz_tolerance: nb.float32


    def __init__(self,
            score_grouped: nb.boolean,
            max_cardinality: nb.uint8,
            top_k_fragments: nb.uint32,
            top_k_isotopes: nb.uint32,
            reference_channel: nb.int16,

            precursor_mz_tolerance: nb.float32,
            fragment_mz_tolerance: nb.float32
        ) -> None:
        """Numba JIT compatible config object for CandidateScoring.
        Will be emitted when `CandidateConfig.jitclass()` is called.

        Please refer to :class:`.alphadia.extraction.plexscoring.CandidateConfig` for documentation.
        """

        self.score_grouped = score_grouped
        self.max_cardinality = max_cardinality
        self.top_k_fragments = top_k_fragments
        self.top_k_isotopes = top_k_isotopes
        self.reference_channel = reference_channel

        self.precursor_mz_tolerance = precursor_mz_tolerance
        self.fragment_mz_tolerance = fragment_mz_tolerance

candidate_config_type = CandidateConfigJIT.class_type.instance_type

class CandidateConfig(config.JITConfig):
    """Config object for CandidateScoring."""

    def __init__(self):
        """Create default config for CandidateScoring"""
        self.score_grouped = False
        self.max_cardinality = 10
        self.top_k_fragments = 16
        self.top_k_isotopes = 4
        self.reference_channel = -1
        self.precursor_mz_tolerance = 10
        self.fragment_mz_tolerance = 15

    @property
    def jit_container(self):
        """The numba jitclass for this config object."""
        return CandidateConfigJIT

    @property
    def score_grouped(self) -> bool:
        """When multiplexing is used, some grouped features are calculated taking into account all channels. 
        Default: `score_grouped = False`"""
        return self._score_grouped
    
    @score_grouped.setter
    def score_grouped(self, value):
        self._score_grouped = value

    @property
    def max_cardinality(self) -> int:
        """When multiplexing is used, some fragments are shared for the same peptide with different labels.
        This setting removes fragments who are shared by more than max_cardinality precursors. 
        Default: `max_cardinality = 10`"""
        return self._max_cardinality
    
    @max_cardinality.setter
    def max_cardinality(self, value):
        self._max_cardinality = value

    @property
    def top_k_fragments(self) -> int:
        """The number of fragments to consider for scoring. The top_k_fragments most intense fragments are used. 
        Default: `top_k_fragments = 16`"""
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

        assert isinstance(self.score_grouped, bool), 'score_grouped must be a boolean'
        assert self.max_cardinality > 0, 'max_cardinality must be greater than 0'
        assert self.top_k_fragments > 0, 'top_k_fragments must be greater than 0'
        assert self.top_k_isotopes > 0, 'top_k_isotopes must be greater than 0'
        assert self.reference_channel >= -1, 'reference_channel must be greater than or equal to -1'
        assert not (self.score_grouped == True and self.reference_channel == -1), 'for grouped scoring, reference_channel must be set to a valid channel'

        assert self.precursor_mz_tolerance >= 0, 'precursor_mz_tolerance must be greater than or equal to 0'
        assert self.precursor_mz_tolerance < 200, 'precursor_mz_tolerance must be less than 200'
        assert self.fragment_mz_tolerance >= 0, 'fragment_mz_tolerance must be greater than or equal to 0'
        assert self.fragment_mz_tolerance < 200, 'fragment_mz_tolerance must be less than 200'

float_array = nb.types.float32[:]

@nb.experimental.jitclass()
class Candidate:

    """
    __init__ will be called single threaded, initialize will later be called multithreaded.
    Therefore as much as possible should be done in initialize.

    """
    failed: nb.boolean

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
    
    #calculated properties
    isotope_mz: nb.float32[::1]
    
    # object properties
    fragments: fragments.FragmentContainer.class_type.instance_type

    features: nb.types.DictType(nb.types.unicode_type, nb.float32)
    fragment_feature_dict: nb.types.DictType(nb.types.unicode_type, nb.float32[:])

    dense_fragments : nb.float32[:, :, :, :, ::1]
    dense_precursors : nb.float32[:, :, :, :, ::1]

    fragments_frame_profile : nb.float32[:, :, ::1]
    fragments_scan_profile : nb.float32[:, :, ::1]

    template_frame_profile : nb.float32[:, ::1]
    template_scan_profile : nb.float32[:, ::1]

    observation_importance : nb.float32[::1]
    template : nb.float32[:, :, ::1]

    def __init__(
            self,
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
            isotope_intensity: nb.float32[::1]
        ) -> None:

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
        string = 'Candidate Object ('
        string += 'precursor_idx: ' + str(self.precursor_idx)
        string += ', channel: ' + str(self.channel) + ')'
        return string        
    
    def initialize(
            self,
            fragment_container,
            config
        ):
        
        # initialize all required dicts
        # accessing uninitialized dicts in numba will result in a kernel crash :)
        self.features = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.types.float32,
        )

        self.fragment_feature_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=float_array
        )

        self.fragments = fragment_container.slice(np.array([[self.frag_start_idx, self.frag_stop_idx, 1]]))
        self.fragments.filter_by_cardinality(config.max_cardinality)
        self.fragments.filter_top_k(config.top_k_fragments)
        self.fragments.sort_by_mz()

        self.assemble_isotope_mz(config)

    def assemble_isotope_mz(self, config):
        """
        Assemble the isotope m/z values from the precursor m/z and the isotope
        offsets.
        """
        n_isotopes = min(self.isotope_intensity.shape[0], config.top_k_isotopes)
        self.isotope_intensity = self.isotope_intensity[:n_isotopes]
        offset = np.arange(self.isotope_intensity.shape[0]) * 1.0033548350700006 / self.charge
        self.isotope_mz = offset.astype(nb.float32) + self.precursor_mz

    def build_profiles(
        self,
        dense_fragments,
        template
    ):
        
        # (n_fragments, n_observations, n_frames)
        self.fragments_frame_profile = features.or_envelope_2d(features.frame_profile_2d(dense_fragments[0]))
        
        # (n_observations, n_frames)
        self.template_frame_profile = features.or_envelope_1d(features.frame_profile_1d(template))

        # (n_fragments, n_observations, n_scans)
        self.fragments_scan_profile = features.or_envelope_2d(features.scan_profile_2d(dense_fragments[0]))

        # (n_observations, n_scans)
        self.template_scan_profile = features.or_envelope_1d(features.scan_profile_1d(template))
    

    def process(
        self,
        jit_data,
        config,
        quadrupole_calibration,
        debug
    ) -> None:
        
        if debug:
            print('precursor', self.precursor_idx, 'channel', self.channel)
        
        frame_limit = np.array(
            [[
                self.frame_start,
                self.frame_stop,
                1
            ]], dtype=np.uint64
        )

        scan_limit = np.array(
            [[
                self.scan_start,
                self.scan_stop,
                1
            ]],dtype=np.uint64
        )

        quadrupole_limit = np.array(
            [[
                np.min(self.isotope_mz)-0.5,
                np.max(self.isotope_mz)+0.5
            ]], dtype=np.float32
        )

        if debug:
            self.visualize_window(
                quadrupole_calibration.cycle_calibrated,
                self.scan_start, self.scan_stop,
                quadrupole_limit[0,0], quadrupole_limit[0,1]
                )
            
        dense_fragments, frag_precursor_index = jit_data.get_dense(
            frame_limit,
            scan_limit,
            self.fragments.mz,
            config.fragment_mz_tolerance,
            quadrupole_limit,
            absolute_masses = True
        )

        self.dense_fragments = dense_fragments

        # check if an empty array is returned
        # scan and quadrupole limits of the fragments candidate are outside the acquisition range
        if dense_fragments.shape[-1] == 0:
            self.failed = True
            return
        
        # only one fragment is found
        if dense_fragments.shape[1] <= 1:
            self.failed = True
            return

        dense_precursors, prec_precursor_index = jit_data.get_dense(
            frame_limit,
            scan_limit,
            self.isotope_mz,
            config.precursor_mz_tolerance,
            np.array([[-1.,-1.]]),
            absolute_masses = True
        )

        self.dense_precursors = dense_precursors

        if debug:
            #self.visualize_precursor(dense_precursors)
            self.visualize_fragments(dense_fragments, self.fragments)

        # (n_isotopes, n_observations, n_scans)
        qtf = quadrupole.quadrupole_transfer_function_single(
            quadrupole_calibration,
            frag_precursor_index,
            np.arange(int(self.scan_start), int(self.scan_stop)),
            self.isotope_mz
        )


        # (n_observation, n_scans, n_frames)
        template = quadrupole.calculate_template_single(
            qtf,
            dense_precursors,
            self.isotope_intensity
        )

        if debug:
            self.visualize_template(
                dense_precursors,
                qtf,
                template,
                self.isotope_intensity
            )

        observation_importance = quadrupole.calculate_observation_importance_single(
            template,
        )
        
        self.observation_importance = observation_importance
        self.template = template

        self.build_profiles(
            dense_fragments,
            template
        )

        if dense_fragments.shape[0] == 0:
            self.failed = True
            return
        
        if dense_precursors.shape[0] == 0:
            self.failed = True
            return

        if debug:
            self.visualize_profiles(
                template,
                self.fragments_scan_profile,
                self.fragments_frame_profile,
                self.template_frame_profile,
                self.template_scan_profile,
            )
        
        
        self.features.update(
            features.location_features(
                jit_data,
                self.scan_start,
                self.scan_stop,
                self.scan_center,
                self.frame_start,
                self.frame_stop,
                self.frame_center,
            )
        )
        
        self.features.update(
            features.precursor_features(
                self.isotope_mz, 
                self.isotope_intensity, 
                dense_precursors, 
                observation_importance,
                template
            )
        )
        
        feature_dict, self.fragment_feature_dict = features.fragment_features(
                dense_fragments,
                observation_importance,
                template,
                self.fragments
            )

        self.features.update(
            feature_dict
        )
        
        
        self.features.update(
            features.profile_features(
                jit_data,
                self.fragments.intensity,
                self.fragments.type,
                observation_importance,
                self.fragments_scan_profile,
                self.fragments_frame_profile,
                self.template_scan_profile,
                self.template_frame_profile,
                self.scan_start,
                self.scan_stop,
                self.frame_start,
                self.frame_stop,
            )
        )
        
        
    def process_reference_channel(
        self,
        reference_candidate
        ):
        
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
                self.fragments.intensity,
            )
        )

    def visualize_window(
            self,
            *args
        ):
        with nb.objmode:
            plotting.plot_dia_window(
                *args
            )

    def visualize_precursor(
            self,
            *args
        ):

        with nb.objmode:
            plotting.plot_precursor(
                *args
            )

    def visualize_fragments(
            self,
            *args
        ):
        with nb.objmode:
            plotting.plot_fragments(
                *args
            )

    def visualize_template(
        self,
        *args
    ):
        with nb.objmode:
            plotting.plot_template(
                *args
            )

    def visualize_profiles(
        self,
        *args
    ):
        with nb.objmode:
            plotting.plot_fragment_profile(
                *args
            )


candidate_type = Candidate.class_type.instance_type

@nb.experimental.jitclass()
class ScoreGroup:
    elution_group_idx: nb.uint32
    score_group_idx: nb.uint32

    candidates: nb.types.ListType(candidate_type)

    def __init__(self,
            elution_group_idx: nb.uint32,
            score_group_idx: nb.uint32
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
        fragment_container,
        jit_data,
        config,
        quadrupole_calibration,
        debug
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
                print('reference channel not found', self.elution_group_idx, self.score_group_idx)
                return

        # process candidates
        for candidate in self.candidates:
            candidate.initialize(
                fragment_container,
                config
            )
            candidate.process(
                jit_data,
                config,
                quadrupole_calibration,
                debug
            )

        # process reference channel features
        if config.reference_channel >= 0:
        
            for idx, candidate in enumerate(self.candidates):
                if idx == reference_channel_idx:
                    continue
                candidate.process_reference_channel(
                    self.candidates[reference_channel_idx]
                )

                # update rank features

                candidate.features.update(
                    features.rank_features(idx, self.candidates)
                )
    
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
        """Initialize the `ScoreGroupContainer` object without any score groups.
        """

        self.score_groups = nb.typed.List.empty_list(score_group_type)

    def __getitem__(self, idx):
        """Get a score group by index.
        """

        return self.score_groups[idx]

    def __len__(self):
        """Get the number of score groups.
        """
        return len(self.score_groups)
    
    def build_from_df(
        self,
        elution_group_idx : nb.uint32,
        score_group_idx : nb.uint32,
        precursor_idx : nb.uint32,
        channel : nb.uint8,
        rank : nb.uint8,

        flat_frag_start_idx : nb.uint32,
        flat_frag_stop_idx : nb.uint32,

        scan_start : nb.uint32,
        scan_stop : nb.uint32,
        scan_center : nb.uint32,
        frame_start : nb.uint32,
        frame_stop : nb.uint32,
        frame_center : nb.uint32,

        precursor_charge : nb.uint8,
        precursor_mz : nb.float32,
        precursor_isotopes : nb.float32[:,::1]
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

                self.score_groups.append(ScoreGroup(
                    elution_group_idx[idx],
                    score_group_idx[idx]
                ))

                # update current score group
                current_score_group_idx = score_group_idx[idx]
                current_precursor_idx = -1

            if len(precursor_isotopes[idx]) == 0:
                raise ValueError('precursor isotopes empty')

            self.score_groups[-1].candidates.append(Candidate(
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
                precursor_isotopes[idx].copy()
            ))

            # check if precursor_idx is unique within a score group
            # if not, some weird "ValueError: unable to broadcast argument 1 to output array" will be raised.
            # Numba bug which took 4h to find :'(
            if current_precursor_idx == precursor_idx[idx]:
                raise ValueError('precursor_idx must be unique within a score group')
            
            current_precursor_idx = precursor_idx[idx]
            idx += 1

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
        known_columns = ['']
        known_columns.clear()

        for i in range(len(self)):
            for j in range(len(self[i].candidates)):
                candidate = self[i].candidates[j]
                if len(candidate.features) not in known_feature_lengths:
                    known_feature_lengths += [len(candidate.features)]
                    # add all new features to the list of known columns
                    for key in candidate.features.keys():
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

        feature_array = np.empty((candidate_count, len(feature_columns)), dtype=np.float32)
        feature_array[:] = np.nan

        precursor_idx_array = np.zeros(candidate_count, dtype=np.uint32)

        rank_array = np.zeros(candidate_count, dtype=np.uint8)

        candidate_idx = 0
        for i in range(len(self)):
            for j in range(len(self[i].candidates)):
                candidate = self[i].candidates[j]

                # iterate all features and add them to the feature array
                for key, value in candidate.features.items():
                        
                        # get the column index for the feature
                        for k in range(len(feature_columns)):
                            if feature_columns[k] == key:
                                feature_array[candidate_idx, k] = value
                                break

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
                if 'mz_library' in self[i].candidates[j].fragment_feature_dict:
                    fragment_count += len(self[i].candidates[j].fragment_feature_dict['mz_library'])
        return fragment_count

    def collect_fragments(self):
        """Iterate all score groups and candidates and accumulate the fragment-level data in a single array

        Parameters
        ----------

        Returns
        -------

        list
        
        """

        fragment_columns = ['mz_library','mz_observed','mass_error','height','intensity']
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
                if 'mz_library' in candidate.fragment_feature_dict:
                    
                    candidate_fragment_count = len(candidate.fragment_feature_dict['mz_library'])
                    for k, col in enumerate(fragment_columns):
                        fragment_array[fragment_start_idx:fragment_start_idx+candidate_fragment_count, k] = candidate.fragment_feature_dict[col]
                        precursor_idx_array[fragment_start_idx:fragment_start_idx+candidate_fragment_count] = candidate.precursor_idx
                        rank_array[fragment_start_idx:fragment_start_idx+candidate_fragment_count] = candidate.rank
                
                    fragment_start_idx += candidate_fragment_count

        return fragment_array, precursor_idx_array, rank_array, fragment_columns
        
ScoreGroupContainer.__module__ = 'alphatims.extraction.plexscoring'

@alphatims.utils.pjit()
def _executor(
        i,
        sg_container,
        fragment_container,
        
        jit_data,
        config,
        quadrupole_calibration,
        debug
    ):
    """
    Helper function.
    Is decorated with alphatims.utils.pjit to enable parallel execution of HybridElutionGroup.process.
    """

    sg_container[i].process(
        fragment_container,
        jit_data,
        config,
        quadrupole_calibration,
        debug
    )

class CandidateScoring():
    """Calculate features for each precursor candidate used in scoring."""
    def __init__(self, 
                dia_data : data.TimsTOFTransposeJIT,
                precursors_flat : pd.DataFrame,
                fragments_flat : pd.DataFrame,
                quadrupole_calibration : quadrupole.SimpleQuadrupole = None,
                config : CandidateConfig = None,
                rt_column : str = 'rt_library',
                mobility_column : str = 'mobility_library',
                precursor_mz_column : str = 'mz_library',
                fragment_mz_column : str = 'mz_library'
                ):
        
        """Initialize candidate scoring step.
        The features can then be used for scoring, calibration and quantification.

        Parameters
        ----------

        dia_data : data.TimsTOFTransposeJIT
            The raw mass spec data as a TimsTOFTransposeJIT object.

        precursors_flat : pd.DataFrame
            A DataFrame containing precursor information. 
            The DataFrame will be validated by using the `alphadia.extraction.validate.precursors_flat` schema.

        fragments_flat : pd.DataFrame
            A DataFrame containing fragment information.
            The DataFrame will be validated by using the `alphadia.extraction.validate.fragments_flat` schema.

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
        self._precursors_flat_df = precursors_flat_df.sort_values(by='precursor_idx')
    
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
        if not hasattr(quadrupole_calibration, 'jit'):
            raise AttributeError('quadrupole_calibration must have a jit method')
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
        self, 
        candidates_df : pd.DataFrame
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

        validate.candidates_df(candidates_df)

        precursor_columns = [
            'channel',
            'flat_frag_start_idx',
            'flat_frag_stop_idx',
            'charge',
            'decoy',
            'channel',
            self.precursor_mz_column
        ] + utils.get_isotope_column_names(self.precursors_flat_df.columns)

        candidates_df = utils.merge_missing_columns(
            candidates_df,
            self.precursors_flat_df,
            precursor_columns,
            on = ['precursor_idx'],
            how = 'left'
        )

        # check if rank column is present
        if not 'rank' in candidates_df.columns:
            candidates_df['rank'] = np.zeros(len(candidates_df), dtype=np.uint8)

        # check if channel column is present
        if not 'channel' in candidates_df.columns:
            candidates_df['channel'] = np.zeros(len(candidates_df), dtype=np.uint8)

        # check if monoisotopic abundance column is present
        if not 'i_0' in candidates_df.columns:
            candidates_df['i_0'] = np.ones(len(candidates_df), dtype=np.float32)

        # calculate score groups
        candidates_df = utils.calculate_score_groups(
            candidates_df, 
            group_channels=self.config.score_grouped
        )

        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.candidates_df(candidates_df)

        score_group_container = ScoreGroupContainer()
        score_group_container.build_from_df(
            candidates_df['elution_group_idx'].values,
            candidates_df['score_group_idx'].values,
            candidates_df['precursor_idx'].values,
            candidates_df['channel'].values,
            candidates_df['rank'].values,

            candidates_df['flat_frag_start_idx'].values,
            candidates_df['flat_frag_stop_idx'].values,

            candidates_df['scan_start'].values,
            candidates_df['scan_stop'].values,
            candidates_df['scan_center'].values,
            candidates_df['frame_start'].values,
            candidates_df['frame_stop'].values,
            candidates_df['frame_center'].values,

            candidates_df['charge'].values,
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
        if 'cardinality' in self.fragments_flat.columns:
            pass
        
        else:
            logging.warning('Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1.')
            self.fragments_flat['cardinality'] = np.ones(len(self.fragments_flat), dtype=np.uint8)
        
        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.fragments_flat(self.fragments_flat)

        return fragments.FragmentContainer(
            self.fragments_flat['mz_library'].values,
            self.fragments_flat[self.fragment_mz_column].values,
            self.fragments_flat['intensity'].values,
            self.fragments_flat['type'].values,
            self.fragments_flat['loss_type'].values,
            self.fragments_flat['charge'].values,
            self.fragments_flat['number'].values,
            self.fragments_flat['position'].values,
            self.fragments_flat['cardinality'].values
        )
    
    def collect_candidates(
        self,
        candidates_df : pd.DataFrame,
        score_group_container : ScoreGroupContainer
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

        feature_array, precursor_idx_array, rank_array, feature_columns = score_group_container.collect_features()
        
        df = pd.DataFrame(feature_array, columns=feature_columns)
        df['precursor_idx'] = precursor_idx_array
        df['rank'] = rank_array

        # join candidate columns
        candidate_df_columns = [
            'elution_group_idx',
            'scan_center',
            'scan_start',
            'scan_stop',
            'frame_center',
            'frame_start',
            'frame_stop'
        ]
        df = utils.merge_missing_columns(
            df,
            candidates_df,
            candidate_df_columns,
            on = ['precursor_idx','rank'],
            how = 'left'
        )

        # join precursor columns
        precursor_df_columns = [
            'rt_library',
            'mobility_library',
            'mz_library',
            'charge',
            'decoy',
            'channel',
            'flat_frag_start_idx',
            'flat_frag_stop_idx'
        ] + utils.get_isotope_column_names(self.precursors_flat_df.columns)
        df = utils.merge_missing_columns(
            df,
            self.precursors_flat_df,
            precursor_df_columns,
            on = ['precursor_idx'],
            how = 'left'
        )

        df.drop(columns=['top3_b_ion_correlation','top3_y_ion_correlation'], inplace=True)

        return df
    
    def collect_fragments(
        self,
        candidates_df : pd.DataFrame,
        score_group_container : ScoreGroupContainer
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

        fragment_array, precursor_idx_array, rank_array, fragment_columns = score_group_container.collect_fragments()

        df = pd.DataFrame(fragment_array, columns=fragment_columns)
        df['precursor_idx'] = precursor_idx_array
        df['rank'] = rank_array

        # join precursor columns
        precursor_df_columns = [
            'elution_group_idx',
            'decoy',
        ]
        df = utils.merge_missing_columns(
            df,
            self.precursors_flat_df,
            precursor_df_columns,
            on = ['precursor_idx'],
            how = 'left'
        )

        return df

    def __call__(
            self, 
            candidates_df, 
            thread_count = 10, 
            debug = False
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

        Returns
        -------

        candidate_features_df : pd.DataFrame
            A DataFrame containing the features for each candidate.

        fragment_features_df : pd.DataFrame
            A DataFrame containing the features for each fragment.

        """

        score_group_container = self.assemble_score_group_container(candidates_df)
        fragment_container = self.assemble_fragments()

        # if debug mode, only iterate over 10 elution groups
        iterator_len = min(10,len(score_group_container)) if debug else len(score_group_container)
        thread_count = 1 if debug else thread_count

        alphatims.utils.set_threads(thread_count)
        _executor(
            range(iterator_len), 
            score_group_container,
            fragment_container,
            self.dia_data,
            self.config.jitclass(),
            self.quadrupole_calibration.jit,
            debug
        )

        candidate_features_df = self.collect_candidates(candidates_df, score_group_container)
        validate.candidate_features_df(candidate_features_df)
        fragment_features_df = self.collect_fragments(candidates_df, score_group_container)
        validate.fragment_features_df(fragment_features_df)

        return candidate_features_df, fragment_features_df
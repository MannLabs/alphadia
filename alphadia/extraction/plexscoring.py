from alphadia.extraction import validate, utils, features, plotting, quadrupole
from alphadia.extraction.numba import fragments

import alphatims.utils

import pandas as pd
import numpy as np
import numba as nb
import logging

class Multiplexer():

    def __init__(self,
        precursors_flat: pd.DataFrame,
        fragments_flat: pd.DataFrame,
        psm_df: pd.DataFrame,
        mz_column: str = 'mz_calibrated',
        ) -> None:

        self.precursors_flat = precursors_flat
        self.fragments_flat = fragments_flat
        self.psm_df = psm_df

        self.mz_column = mz_column

    def __call__(self):
        # make sure input psm's have all required columns
        self.psm_df = self.psm_df[self.psm_df['decoy'] == 0].copy()
        anchor_ids = self.psm_df[['elution_group_idx', 'scan_start' ,'scan_stop', 'scan_center', 'frame_start', 'frame_stop', 'frame_center','rank']]
        
        candidates_df = self.precursors_flat[(self.precursors_flat['decoy'] == 0)]
        candidates_df = candidates_df[candidates_df['elution_group_idx'].isin(anchor_ids['elution_group_idx'])]
        candidates_df = candidates_df[['precursor_idx', 'elution_group_idx', 'channel', 'decoy','flat_frag_start_idx','flat_frag_stop_idx','charge',self.mz_column]+utils.get_isotope_column_names(candidates_df.columns)]

        candidates_df = candidates_df.merge(anchor_ids, on='elution_group_idx', how='outer')
        candidates_df = candidates_df.sort_values('precursor_idx')
        validate.candidates(candidates_df)
        return candidates_df
    
def assemble_fragments(fragments_flat, fragment_mz_column='mz_calibrated'):
            
    # set cardinality to 1 if not present
    if 'cardinality' in fragments_flat.columns:
        pass
    
    else:
        logging.warning('Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1.')
        fragments_flat['cardinality'] = np.ones(len(fragments_flat), dtype=np.uint8)
    
    # validate dataframe schema and prepare jitclass compatible dtypes
    validate.fragments_flat(fragments_flat)

    return fragments.FragmentContainer(
        fragments_flat['mz_library'].values,
        fragments_flat[fragment_mz_column].values,
        fragments_flat['intensity'].values,
        fragments_flat['type'].values,
        fragments_flat['loss_type'].values,
        fragments_flat['charge'].values,
        fragments_flat['number'].values,
        fragments_flat['position'].values,
        fragments_flat['cardinality'].values
    )

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

        self.score_grouped = score_grouped
        self.max_cardinality = max_cardinality
        self.top_k_fragments = top_k_fragments
        self.top_k_isotopes = top_k_isotopes
        self.reference_channel = reference_channel

        self.precursor_mz_tolerance = precursor_mz_tolerance
        self.fragment_mz_tolerance = fragment_mz_tolerance

candidate_config_type = CandidateConfigJIT.class_type.instance_type

class CandidateConfig(config.JITConfig):

    jit_container = CandidateConfigJIT

    def __init__(self):
        self.score_grouped = True
        self.max_cardinality = 10
        self.top_k_fragments = 16
        self.top_k_isotopes = 4
        self.reference_channel = 0

        self.precursor_mz_tolerance = 10
        self.fragment_mz_tolerance = 15
    
    def validate(self):
        assert self.max_cardinality > 0, 'max_cardinality must be greater than 0'
        assert self.top_k_fragments > 0, 'top_k_fragments must be greater than 0'
        assert self.top_k_isotopes > 0, 'top_k_isotopes must be greater than 0'
        assert self.reference_channel >= -1, 'reference_channel must be greater than or equal to -1'
        assert not (self.score_grouped == True and self.reference_channel == -1), 'for grouped scoring, reference_channel must be set to a valid channel'

        assert self.precursor_mz_tolerance >= 0, 'precursor_mz_tolerance must be greater than or equal to 0'
        assert self.precursor_mz_tolerance < 200, 'precursor_mz_tolerance must be less than 200'
        assert self.fragment_mz_tolerance >= 0, 'fragment_mz_tolerance must be greater than or equal to 0'
        assert self.fragment_mz_tolerance < 200, 'fragment_mz_tolerance must be less than 200'

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
        
        self.features = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.types.float32,
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
        
        """
        Container for managing the scoring of precursors with defined boundaries.

        The `ScoreGroupContainer` contains all precursors that are to be scored.
        It consists of a list of `ScoreGroup` objects, which in turn contain a list of `Candidate` objects.

        For single channel experiments, each `ScoreGroup` contains a single `Candidate` object.
        For multi channel experiments, each `ScoreGroup` contains a `Candidate` object for each channel, including decoy channels.

        Structure:
        ```
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
        ```

        The `ScoreGroupContainer` is initialized by passing the validated columns of a candidate dataframe to the `build_from_df` method.

        
        Attributes
        ----------

        score_groups : nb.types.ListType(score_group_type)
            List of score groups.

        """
    
        score_groups: nb.types.ListType(score_group_type)
    
        def __init__(
                self,
            ) -> None:

            """
            Initialize the `ScoreGroupContainer` object without any score groups.
            """

            self.score_groups = nb.typed.List.empty_list(score_group_type)

        def __getitem__(self, idx):
            """
            Get a score group by index.
            """

            return self.score_groups[idx]

        def __len__(self):
            """
            Get the number of score groups.
            """
            return len(self.score_groups)
        
        def build_from_df(
            self,
            elution_group_idx : nb.uint32,
            score_group_idx : nb.uint32,
            precursor_idx : nb.uint32,
            channel : nb.uint8,
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
            
            """
            Build the `ScoreGroupContainer` from a candidate dataframe.
            All relevant columns of the candidate dataframe are passed to this method as numpy arrays.

            Note
            ----

            All columns of the candidate_df need to be validated for the correct type using the `extraction.validate.candidates` schema.
            columns musst be sorted by `score_group_idx` in ascending order.

            Parameters
            ----------

            """
            idx = 0
            current_score_group_idx = -1

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

                if len(precursor_isotopes[idx]) == 0:
                    raise ValueError('precursor isotopes empty')

                self.score_groups[-1].candidates.append(Candidate(
                    precursor_idx[idx],
                    channel[idx],
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

                idx += 1

        def collect_to_df(
            self,
        ):
            for score_group in self.score_groups:
                for candidate in score_group.candidates:
                    yield score_group.elution_group_idx, score_group.score_group_idx, candidate.precursor_idx, candidate.channel

        
        
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


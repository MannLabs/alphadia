from alphadia.extraction import utils
from alphadia.extraction.numba import fragments, numeric
from alphadia.extraction.utils import fourier_filter
from alphadia.extraction import validate
from alphadia.extraction.candidateselection import peak_boundaries_symmetric, GaussianFilter
import numba as nb
import numpy as np
import pandas as pd
import logging
import alphatims

import matplotlib.pyplot as plt
from matplotlib import patches

@nb.experimental.jitclass()
class Candidate:

    elution_group_idx: nb.int64
    score_group_idx: nb.int64
    precursor_idx: nb.int64
    rank: nb.int64

    score: nb.float64
    precursor_mz: nb.float64
    decoy: nb.int8
    features: nb.float32[::1]

    scan_center: nb.int64
    scan_start: nb.int64
    scan_stop: nb.int64

    frame_center: nb.int64
    frame_start: nb.int64
    frame_stop: nb.int64


    def __init__(
            self,
            elution_group_idx,
            score_group_idx,
            precursor_idx,
            rank,
            score,
            precursor_mz,
            decoy,
            features,
            scan_center,
            scan_start,
            scan_stop,
            frame_center,
            frame_start,
            frame_stop,
    ):
        self.elution_group_idx = elution_group_idx
        self.score_group_idx = score_group_idx
        self.precursor_idx = precursor_idx
        self.rank = rank
        self.score = score
        self.precursor_mz = precursor_mz
        self.decoy = decoy
        self.features = features
        self.scan_center = scan_center
        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.frame_center = frame_center
        self.frame_start = frame_start
        self.frame_stop = frame_stop

candidate_type = Candidate.class_type.instance_type

@nb.experimental.jitclass()
class HybridElutionGroup:

    # values which are shared by all precursors in the elution group
    # (1)
    score_group_idx: nb.uint32
    elution_group_idx: nb.uint32
    rt: nb.float32
    mobility: nb.float32
    charge: nb.uint8

    # values which are specific to each precursor in the elution group
    # (n_precursor)
    precursor_idx: nb.uint32[::1]
    precursor_channel: nb.uint32[::1]
    precursor_decoy: nb.uint8[::1]
    precursor_mz: nb.float32[::1]
    precursor_score_group: nb.int32[::1]
    precursor_abundance: nb.float32[::1]

    # (n_precursor, 2)
    precursor_frag_start_stop_idx: nb.uint32[:,::1]

    # (n_precursor, n_isotopes)
    precursor_isotope_intensity: nb.float32[:, :]
    precursor_isotope_mz: nb.float32[:, ::1]

    frame_limits: nb.uint64[:, ::1]
    scan_limits: nb.uint64[:, ::1]
    precursor_tof_limits: nb.uint64[:, ::1]
    fragment_tof_limits: nb.uint64[:, ::1]
    
    candidate_precursor_idx: nb.uint32[::1]
    candidate_mass_error: nb.float64[::1]
    candidate_fraction_nonzero: nb.float64[::1]
    candidate_intensity: nb.float32[::1]

    candidate_scan_limit: nb.int64[:, ::1]
    candidate_frame_limit: nb.int64[:, ::1]

    candidate_scan_center: nb.int64[::1]
    candidate_frame_center: nb.int64[::1]

    fragments: fragments.FragmentContainer.class_type.instance_type
    candidates: nb.types.ListType(candidate_type)

    #only for debugging

    dense_fragments : nb.float32[:, :, :, ::1]
    dense_precursors : nb.float32[:, :, :, ::1]

    score_group_fragment_mz: nb.float32[::1]
    score_group_precursor_mz: nb.float32[::1]

    score_group_precursor_intensity: nb.float32[::1]
    score_group_fragment_intensity: nb.float32[::1]


    def __init__(
            self, 
            score_group_idx,
            elution_group_idx,
            precursor_idx,
            channel,
            frag_start_stop_idx,
            rt,
            mobility,
            charge,
            decoy,
            mz,
            #isotope_apex_offset,
            isotope_intensity
        ) -> None:
        """
        ElutionGroup jit class which contains all information about a single elution group.

        Parameters
        ----------
        elution_group_idx : int
            index of the elution group as encoded in the precursor dataframe
        
        precursor_idx : numpy.ndarray
            indices of the precursors in the precursor dataframe

        rt : float
            retention time of the elution group in seconds, shared by all precursors

        mobility : float
            mobility of the elution group, shared by all precursors

        charge : int
            charge of the elution group, shared by all precursors

        decoy : numpy.ndarray
            array of integers indicating whether the precursor is a decoy (1) or target (0)

        mz : numpy.ndarray
            array of m/z values of the precursors

        isotope_apex_offset : numpy.ndarray
            array of integers indicating the offset of the isotope apex from the precursor m/z. 
        """
        
        self.score_group_idx = score_group_idx
        self.elution_group_idx = elution_group_idx
        self.precursor_idx = precursor_idx
        self.rt = rt
        self.mobility = mobility
        self.charge = charge

        self.precursor_decoy = decoy
        self.precursor_mz = mz
        self.precursor_channel = channel
        self.precursor_frag_start_stop_idx = frag_start_stop_idx
        self.precursor_isotope_intensity = isotope_intensity
        self.candidates = nb.typed.List.empty_list(candidate_type)

    def __str__(self):
        with nb.objmode(r='unicode_type'):
            r = f'ElutionGroup(\nelution_group_idx: {self.elution_group_idx},\nprecursor_idx: {self.precursor_idx}\n)'
        return r

    def sort_by_mz(self):
        """
        Sort all precursor arrays by m/z
        
        """
        mz_order = np.argsort(self.precursor_mz)
        self.precursor_mz = self.precursor_mz[mz_order]
        self.precursor_decoy = self.precursor_decoy[mz_order]
        self.precursor_idx = self.precursor_idx[mz_order]
        self.precursor_frag_start_stop_idx = self.precursor_frag_start_stop_idx[mz_order]

    def assemble_isotope_mz(self):
        """
        Assemble the isotope m/z values from the precursor m/z and the isotope
        offsets.
        """
        offset = np.arange(self.precursor_isotope_intensity.shape[1]) * 1.0033548350700006 / self.charge
        self.precursor_isotope_mz = np.expand_dims(self.precursor_mz, 1).astype(np.float32) + np.expand_dims(offset,0).astype(np.float32)

    def trim_isotopes(self):

        divisor = self.precursor_isotope_intensity.shape[0]

        if divisor == 0:
            raise ZeroDivisionError('Cannot divide by zero')
        
        elution_group_isotopes = np.sum(self.precursor_isotope_intensity, axis=0)/divisor
        self.precursor_isotope_intensity = self.precursor_isotope_intensity[:,elution_group_isotopes>0.1]

    def determine_frame_limits(
            self, 
            jit_data, 
            tolerance
        ):
        """
        Determine the frame limits for the elution group based on the retention time and rt tolerance.

        Parameters
        ----------
        jit_data : alphadia.extraction.data.TimsTOFJIT
            TimsTOFJIT object containing the raw data

        tolerance : float
            tolerance in seconds

        """

        rt_limits = np.array([
            self.rt-tolerance, 
            self.rt+tolerance
        ])
    
        self.frame_limits = utils.make_slice_1d(
            jit_data.return_frame_indices(
                rt_limits,
                True
            )
        )

    def determine_scan_limits(
            self, 
            jit_data, 
            tolerance
        ):
        """
        Determine the scan limits for the elution group based on the mobility and mobility tolerance.

        Parameters
        ----------
        jit_data : alphadia.extraction.data.TimsTOFJIT
            TimsTOFJIT object containing the raw data

        tolerance : float
            tolerance in inverse mobility units
        """

        mobility_limits = np.array([
            self.mobility+tolerance,
            self.mobility-tolerance
        ])

        self.scan_limits = utils.make_slice_1d(

            jit_data.return_scan_indices(
                mobility_limits
            )

        )

    def determine_precursor_tof_limits(
            self, 
            jit_data, 
            tolerance
        ):

        """
        Determine all tof limits for the elution group based on the top isotope m/z and m/z tolerance.

        Parameters
        ----------
        jit_data : alphadia.extraction.data.TimsTOFJIT
            TimsTOFJIT object containing the raw data

        tolerance : float
            tolerance in part per million (ppm)

        """
        precursor_mz_limits = utils.mass_range(self.precursor_mz, tolerance)
        self.precursor_tof_limits = utils.make_slice_2d(jit_data.return_tof_indices(
            precursor_mz_limits
        ))

    def determine_fragment_tof_limits(
            self, 
            jit_data, 
            tolerance
        ):

        """
        Determine all tof limits for the elution group based on the top isotope m/z and m/z tolerance.

        Parameters
        ----------
        jit_data : alphadia.extraction.data.TimsTOFJIT
            TimsTOFJIT object containing the raw data

        tolerance : float
            tolerance in part per million (ppm)

        """

        fragment_mz_limits = utils.mass_range(self.fragments.mz, tolerance)
        self.fragment_tof_limits = utils.make_slice_2d(
            jit_data.return_tof_indices(
                fragment_mz_limits
            )
        )

    def determine_fragment_scan_limits(
        self,
        quad_slices, 
        dia_data
        ):
        quad_mask = alphatims.bruker.calculate_dia_cycle_mask(
            dia_mz_cycle=dia_data.dia_mz_cycle,
            quad_slices=np.array([[400.,402]]),
            dia_precursor_cycle=dia_data.dia_precursor_cycle,
            precursor_slices=None
        )

        mask = quad_mask.reshape(dia_data.cycle.shape[:3])
        _, _, scans = mask.nonzero()

        return scans.min(), scans.max()
    
    def build_candidates(
        self,
        dense_fragments,
        dense_precursors,
        kernel,
        candidate_count,
        jit_data
    ):
        candidates = nb.typed.List.empty_list(candidate_type)

        # return empty list if no dense data was accumulated
        if dense_fragments.shape[2] < kernel.shape[0] or dense_fragments.shape[3] < kernel.shape[1]:
            return candidates

        smooth_precursor = numeric.fourier_a0(dense_precursors[0], kernel)
        smooth_fragment = numeric.fourier_a0(dense_fragments[0], kernel)
        
        if 1 > 2:
            candidates.append(
                Candidate(1)
            )
        
        return candidates
    
    def determine_score_groups(
        self,
        score_grouped
    ):
        """
        Determines how the different precursors are grouped for scoring.

        Parameters
        ----------

        score_grouped : bool
            If True, the precursors are grouped by their decoy status. If False, each precursor is scored individually.

        Returns
        -------
        group_ids : np.ndarray, dtype=np.uint32
            Array of group ids for each precursor (n_precursor).
        
        """

        if score_grouped:
            # The resulting score groups are expected to start with 0 and be consecutive
            # As there can be decoy only groups, we need to reindex the decoy array
            group_ids = np.unique(self.precursor_decoy)
            group_ids_reverse = np.zeros(np.max(group_ids)+1, dtype=np.int32)
            group_ids_reverse[group_ids] = np.arange(len(group_ids))
            return group_ids_reverse[self.precursor_decoy]

        else:
            return np.arange(len(self.precursor_decoy), dtype=np.int32)


    def process(
        self, 
        jit_data, 
        fragment_container,
        kernel, 
        rt_tolerance,
        mobility_tolerance,
        mz_tolerance,
        candidate_count, 
        debug,
        exclude_shared_fragments,
        top_k_fragments,
        top_k_precursors,
    ):

        """
        Process the elution group and store the candidates.

        Parameters
        ----------

        jit_data : alphadia.extraction.data.TimsTOFJIT
            TimsTOFJIT object containing the raw data

        kernel : np.ndarray
            Matrix of size (20, 20) containing the smoothing kernel

        rt_tolerance : float
            tolerance in seconds

        mobility_tolerance : float
            tolerance in inverse mobility units

        mz_tolerance : float
            tolerance in part per million (ppm)

        candidate_count : int
            number of candidates to select per precursor.

        debug : bool
            if True, self.visualize_candidates() will be called after processing the elution group.
            Make sure to use debug mode only on a small number of elution groups (10) and with a single thread. 
        """
        
        
        
        precursor_abundance = np.ones((len(self.precursor_decoy)), dtype=np.float32)
        precursor_abundance[self.precursor_channel == 0] = 10

        self.precursor_abundance = precursor_abundance

        self.sort_by_mz()
        self.trim_isotopes()
        self.assemble_isotope_mz()

        fragment_idx_slices = utils.make_slice_2d(
            self.precursor_frag_start_stop_idx
        )
        
        self.fragments = fragments.slice_manual(fragment_container,fragment_idx_slices)
        self.fragments.sort_by_mz()

        self.determine_frame_limits(jit_data, rt_tolerance)
        self.determine_scan_limits(jit_data, mobility_tolerance)
        self.determine_precursor_tof_limits(jit_data, mz_tolerance)
        self.determine_fragment_tof_limits(jit_data, mz_tolerance)
        #self.precursor_score_group = self.determine_score_groups(True)

        fragment_mz, fragment_intensity = fragments.get_ion_group_mapping(
            self.fragments.precursor_idx,
            self.fragments.mz,
            self.fragments.intensity,
            self.fragments.cardinality,
            self.precursor_abundance,
            top_k = top_k_fragments,
            max_cardinality = 10
        )
        

        # return if no valid fragments are left after grouping
        if len (fragment_mz) == 0:
            return
        
        # only for debugging
        self.score_group_fragment_mz = fragment_mz
        self.score_group_fragment_intensity = fragment_intensity

        

        isotope_mz = self.precursor_isotope_mz.flatten()
        isotope_intensity = self.precursor_isotope_intensity.flatten()
        isotope_precursor = np.repeat(np.arange(0, self.precursor_isotope_mz.shape[0]), self.precursor_isotope_mz.shape[1])

        order = np.argsort(isotope_mz)
        isotope_mz = isotope_mz[order]
        isotope_intensity = isotope_intensity[order]
        isotope_precursor = isotope_precursor[order]

        precursor_mz, precursor_intensity = fragments.get_ion_group_mapping(
            isotope_precursor, 
            isotope_mz,
            isotope_intensity,
            np.ones(len(isotope_mz), dtype=np.uint8),
            self.precursor_abundance,
            top_k = top_k_precursors
        )

        # return if no valid precursors are left after grouping
        if len(precursor_mz) == 0:
            return

        # only for debugging
        self.score_group_precursor_mz = precursor_mz
        self.score_group_precursor_intensity = precursor_intensity

        return

        quadrupole_mz = calculate_score_group_limits(
            precursor_mz, 
            precursor_intensity
        )

        precursor_tof_limits = utils.make_slice_2d( 
            jit_data.return_tof_indices( 
                utils.mass_range(
                    precursor_mz, mz_tolerance
                )
            )
        )

        fragment_tof_limits = utils.make_slice_2d(
            jit_data.return_tof_indices(
                utils.mass_range(
                    fragment_mz, mz_tolerance
                )
            )
        )

        precursor_cycle_mask = calculate_dia_cycle_mask(
            jit_data.cycle,
            np.array([[-1., -1.]])
        )

        fragment_cycle_mask = calculate_dia_cycle_mask(
            jit_data.cycle,
            quadrupole_mz
        )

        # combines different quadrupole limits into one
        # push_query : (n_pushes) 
        # push_indices : (n_pushes)
        push_query, _absolute_precursor_index = get_push_indices(
            jit_data,
            self.frame_limits,
            self.scan_limits,
            precursor_cycle_mask,
        )

        # (2, n_precursor_isotopes, n_scans, n_frames)
        dense_precursors = assemble_push(
            precursor_tof_limits,
            precursor_mz,
            jit_data.push_indices,
            jit_data.tof_indptr, 
            push_query,
            _absolute_precursor_index,
            self.frame_limits,
            self.scan_limits,
            mz_tolerance,
            jit_data
        ).sum(axis=2)

        push_query, _absolute_precursor_index = get_push_indices(
            jit_data,
            self.frame_limits,
            self.scan_limits,
            fragment_cycle_mask,
        )
        
        # (2, n_fragments, n_scans, n_frames)
        dense_fragments = assemble_push(
            fragment_tof_limits,
            fragment_mz,
            jit_data.push_indices,
            jit_data.tof_indptr, 
            push_query,
            _absolute_precursor_index,
            self.frame_limits,
            self.scan_limits,
            mz_tolerance,
            jit_data
        ).sum(axis=2)

        #self.dense_fragments = dense_fragments
        #self.dense_precursors = dense_precursors

        if dense_fragments.shape[0] == 0:
            return

        if dense_precursors.shape[0] == 0:
            return
        
        if not dense_fragments.shape[2] % 2 == 0:
            return
        
        if not dense_fragments.shape[3] % 2 == 0:
            return

        if dense_fragments.shape[2] > kernel.shape[0] and dense_fragments.shape[3] > kernel.shape[1]:

            self.candidates = build_candidates(
                dense_precursors,
                dense_fragments,
                self.score_group_precursor_intensity,
                self.score_group_fragment_intensity,
                self,
                kernel,
                jit_data,
                candidate_count = candidate_count,
                debug=debug,
                weights = np.array([1,1,1,1,1,0.5,0.5,0.5]),
            )
        return

    def visualize_candidates(
        self, 
        smooth_dense
    ):
        """
        Visualize the candidates of the elution group using numba objmode.

        Parameters
        ----------

        dense : np.ndarray
            The raw, dense intensity matrix of the elution group. 
            Shape: (2, n_precursors, n_observations ,n_scans, n_cycles)
            n_observations is indexed based on the 'precursor' index within a DIA cycle. 

        smooth_dense : np.ndarray
            Dense data of the elution group after smoothing.
            Shape: (n_precursors, n_observations, n_scans, n_cycles)

        """
        with nb.objmode():

            n_precursors = len(self.precursor_idx)

            fig, axs = plt.subplots(n_precursors,2, figsize=(10,n_precursors*3))

            if axs.shape == (2,):
                axs = axs.reshape(1,2)

            # iterate all precursors
            for j, idx in enumerate(self.precursor_idx):

                axs[j,0].set_xlabel('cycle')
                axs[j,0].set_ylabel('scan')
                axs[j,0].set_title(f'- RAW DATA - elution group: {self.elution_group_idx}, precursor: {idx}')

                axs[j,1].imshow(smooth_dense[j], aspect='auto')
                axs[j,1].set_xlabel('cycle')
                axs[j,1].set_ylabel('scan')
                axs[j,1].set_title(f'- Candidates - elution group: {self.elution_group_idx}, precursor: {idx}')

                candidate_mask = self.candidate_precursor_idx == idx
                for k, (scan_limit, scan_center, frame_limit, frame_center) in enumerate(zip(
                    self.candidate_scan_limit[candidate_mask],
                    self.candidate_scan_center[candidate_mask],
                    self.candidate_frame_limit[candidate_mask],
                    self.candidate_frame_center[candidate_mask]
                )):
                    axs[j,1].scatter(
                        frame_center, 
                        scan_center, 
                        c='r', 
                        s=10
                    )

                    axs[j,1].text(frame_limit[1], scan_limit[0], str(k), color='r')

                    axs[j,1].add_patch(patches.Rectangle(
                        (frame_limit[0], scan_limit[0]),
                        frame_limit[1]-frame_limit[0],
                        scan_limit[1]-scan_limit[0],
                        fill=False,
                        edgecolor='r'
                    ))

            fig.tight_layout()   
            plt.show()





@nb.experimental.jitclass()
class HybridElutionGroupContainer:
    
    elution_groups: nb.types.ListType(HybridElutionGroup.class_type.instance_type)

    def __init__(
            self, 
            elution_groups,
        ) -> None:
        """
        Container class which contains a list of ElutionGroup objects.

        Parameters
        ----------
        elution_groups : nb.types.ListType(ElutionGroup.class_type.instance_type)
            List of ElutionGroup objects.
        
        """

        self.elution_groups = elution_groups

    def __getitem__(self, idx):
        return self.elution_groups[idx]

    def __len__(self):
        return len(self.elution_groups)


class HybridCandidateSelection(object):

    def __init__(self, 
            dia_data,
            precursors_flat, 
            fragments_flat,
            rt_tolerance = 30,
            mobility_tolerance = 0.03,
            mz_tolerance = 120,
            candidate_count = 3,
            rt_column = 'rt_library',  
            mobility_column = 'mobility_library',
            precursor_mz_column = 'mz_library',
            fragment_mz_column = 'mz_library',
            thread_count = 10,
            debug = False,
            group_channels = False,
            exclude_shared_fragments = True,
            top_k_fragments = 12,
            top_k_precursors = 3,
        ):
        """select candidates for MS2 extraction based on MS1 features

        Parameters
        ----------

        dia_data : alphadia.extraction.data.TimsTOFDIA
            dia data object

        precursors_flat : pandas.DataFrame
            flattened precursor dataframe

        rt_tolerance : float, optional
            rt tolerance in seconds, by default 30

        mobility_tolerance : float, optional
            mobility tolerance, by default 0.03

        mz_tolerance : float, optional
            mz tolerance in ppm, by default 120

        candidate_count : int, optional
            number of candidates to extract per precursor, by default 3

        rt_column : str, optional
            name of the rt column in the precursor dataframe, by default 'rt_library'

        mobility_column : str, optional
            name of the mobility column in the precursor dataframe, by default 'mobility_library'

        precursor_mz_column : str, optional
            name of the precursor mz column in the precursor dataframe, by default 'mz_library'

        fragment_mz_column : str, optional
            name of the fragment mz column in the fragment dataframe, by default 'mz_library'

        thread_count : int, optional
            number of threads to use, by default 20

        debug : bool, optional
            if True, debug plots will be shown, by default False

        score_grouped : bool, optional
            if True, the score will be calculated based on the grouped precursors. All non-decoy precursors and decoy-precursors are grouped together.
            If False, the score will be calculated based on the individual precursors. This is the default behaviour of al
        Returns
        -------

        pandas.DataFrame
            dataframe containing the extracted candidates
        """
        self.dia_data = dia_data.jitclass(transpose=True)
        self.precursors_flat = precursors_flat.sort_values('precursor_idx').reset_index(drop=True)
        self.fragments_flat = fragments_flat

        self.debug = debug
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.mobility_tolerance = mobility_tolerance
        self.candidate_count = candidate_count

        self.thread_count = thread_count

        self.rt_column = rt_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column
        self.mobility_column = mobility_column

        gaussian_filter = GaussianFilter(
            dia_data
        )
        self.kernel = gaussian_filter.get_kernel()

        self.available_isotopes = utils.get_isotope_columns(self.precursors_flat.columns)
        self.available_isotope_columns = [f'i_{i}' for i in self.available_isotopes]

        self.group_channels = group_channels
        self.exclude_shared_fragments = exclude_shared_fragments
        self.top_k_fragments = top_k_fragments
        self.top_k_precursors = top_k_precursors

        print({
            'rt_tolerance': rt_tolerance,
            'mobility_tolerance': mobility_tolerance,
            'mz_tolerance': mz_tolerance,
            'candidate_count': candidate_count,
            'rt_column': rt_column,
            'mobility_column': mobility_column,
            'precursor_mz_column': precursor_mz_column,
            'fragment_mz_column': fragment_mz_column,
            'thread_count': thread_count,
            'debug': debug,
            'group_channels': group_channels,
            'exclude_shared_fragments': exclude_shared_fragments,
            'top_k_fragments': top_k_fragments,
            'top_k_precursors': top_k_precursors,

        })

    def __call__(self):
        """
        Perform candidate extraction workflow. 
        1. First, elution groups are assembled based on the annotation in the flattened precursor dataframe.
        Each elution group is instantiated as an ElutionGroup Numba JIT object. 
        Elution groups are stored in the ElutionGroupContainer Numba JIT object.

        2. Then, the elution groups are iterated over and the candidates are selected.
        The candidate selection is performed in parallel using the alphatims.utils.pjit function.

        3. Finally, the candidates are collected from the ElutionGroup, 
        assembled into a pandas.DataFrame and precursor information is appended.
        """

        if self.debug:
            logging.info('starting candidate selection')

        # initialize input container
        elution_group_container = self.assemble_score_groups(self.precursors_flat, group_channels = self.group_channels)
        fragment_container = self.assemble_fragments()

        # if debug mode, only iterate over 10 elution groups
        iterator_len = min(10,len(elution_group_container)) if self.debug else len(elution_group_container)
        thread_count = 1 if self.debug else self.thread_count

        alphatims.utils.set_threads(thread_count)

        _executor(
            range(iterator_len), 
            elution_group_container,
            self.dia_data, 
            fragment_container, 
            self.kernel, 
            self.rt_tolerance,
            self.mobility_tolerance,
            self.mz_tolerance,
            self.candidate_count,
            self.debug,
            self.exclude_shared_fragments,
            self.top_k_fragments,
            self.top_k_precursors
        )

        if self.debug: 
            return elution_group_container
   
        df = self.assemble_candidates(elution_group_container)
        df = self.append_precursor_information(df)
        #self.log_stats(df)
        return df
    
    def assemble_fragments(self):
            
            # set cardinality to 1 if not present
            if 'cardinality' in self.fragments_flat.columns:
                self.fragments_flat['cardinality'] = self.fragments_flat['cardinality'].values
            
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

    def assemble_score_groups(
            self,
            precursors_flat,
            group_channels=False,
        ):
    
        """
        Assemble elution groups from precursor library.

        Parameters
        ----------

        precursors_flat : pandas.DataFrame
            Precursor library.

        Returns
        -------
        HybridElutionGroupContainer
            Numba jitclass with list of elution groups.
        """
        
        if len(precursors_flat) == 0:
            return

        available_isotopes = utils.get_isotope_columns(precursors_flat.columns)
        available_isotope_columns = [f'i_{i}' for i in available_isotopes]

        precursors_sorted = utils.calculate_score_groups(precursors_flat, group_channels).copy()

        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.precursors_flat(precursors_sorted)

        @nb.njit(debug=True)
        def assemble_njit(
            score_group_idx,
            elution_group_idx,
            precursor_idx,
            channel,
            flat_frag_start_stop_idx,
            rt_values,
            mobility_values,
            charge,
            decoy,
            precursor_mz,
            isotope_intensity
        ):
            score_group = score_group_idx[0]
            score_group_start = 0
            score_group_stop = 0

            eg_list = []
            
            while score_group_stop < len(score_group_idx)-1:
                
                score_group_stop += 1

                if score_group_idx[score_group_stop] != score_group:
                        
                    eg_list.append(HybridElutionGroup(    
                        score_group,
                        elution_group_idx[score_group_start],
                        precursor_idx[score_group_start:score_group_stop],
                        channel[score_group_start:score_group_stop],
                        flat_frag_start_stop_idx[score_group_start:score_group_stop],
                        rt_values[score_group_start],
                        mobility_values[score_group_start],
                        charge[score_group_start],
                        decoy[score_group_start:score_group_stop],
                        precursor_mz[score_group_start:score_group_stop],
                        isotope_intensity[score_group_start:score_group_stop]
                    ))

                    score_group_start = score_group_stop
                    score_group = score_group_idx[score_group_start]
                    
            egs = nb.typed.List(eg_list)
            return HybridElutionGroupContainer(egs)

        return assemble_njit(
            precursors_sorted['score_group_idx'].values,
            precursors_sorted['elution_group_idx'].values,
            precursors_sorted['precursor_idx'].values,
            precursors_sorted['channel'].values,
            precursors_sorted[['flat_frag_start_idx','flat_frag_stop_idx']].values.copy(),
            precursors_sorted[self.rt_column].values,
            precursors_sorted[self.mobility_column].values,
            precursors_sorted['charge'].values,
            precursors_sorted['decoy'].values,
            precursors_sorted[self.precursor_mz_column].values,
            precursors_sorted[available_isotope_columns].values.copy(),
        )
    
    def assemble_candidates(
            self, 
            elution_group_container
        ):

        """
        Candidates are collected from the ElutionGroup objects and assembled into a pandas.DataFrame.

        Parameters
        ----------
        elution_group_container : ElutionGroupContainer
            container object containing a list of ElutionGroup objects

        Returns
        -------
        pandas.DataFrame
            dataframe containing the extracted candidates
        
        """
       
        candidates = []
        for i in range(len(elution_group_container)):
            for j in range(len(elution_group_container[i].candidates)):
                candidates.append(elution_group_container[i].candidates[j])
    
        attributes = ['elution_group_idx', 'score_group_idx', 'precursor_idx', 'rank','score','precursor_mz', 'decoy', 'scan_center', 'scan_start', 'scan_stop', 'frame_center', 'frame_start', 'frame_stop']
        candidates = pd.DataFrame({attr: [getattr(c, attr) for c in candidates] for attr in attributes})

        return candidates
    
    def append_precursor_information(
            self, 
            df
        ):
        """
        Append relevant precursor information to the candidates dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe containing the extracted candidates

        Returns
        -------
        pandas.DataFrame
            dataframe containing the extracted candidates with precursor information appended
        """

        # precursor_flat_lookup has an element for every candidate and contains the index of the respective precursor
        precursor_pidx = self.precursors_flat['precursor_idx'].values
        candidate_pidx = df['precursor_idx'].values
        precursor_flat_lookup = np.searchsorted(precursor_pidx, candidate_pidx, side='left')

        df['decoy'] = self.precursors_flat['decoy'].values[precursor_flat_lookup]

        df['rt_library'] = self.precursors_flat['rt_library'].values[precursor_flat_lookup]
        if self.rt_column == 'rt_calibrated':
            df['rt_calibrated'] = self.precursors_flat['rt_calibrated'].values[precursor_flat_lookup]
        
        df['mobility_library'] = self.precursors_flat['mobility_library'].values[precursor_flat_lookup]
        if self.mobility_column == 'mobility_calibrated':
            df['mobility_calibrated'] = self.precursors_flat['mobility_calibrated'].values[precursor_flat_lookup]

        df['flat_frag_start_idx'] = self.precursors_flat['flat_frag_start_idx'].values[precursor_flat_lookup]
        df['flat_frag_stop_idx'] = self.precursors_flat['flat_frag_stop_idx'].values[precursor_flat_lookup]

        df['charge'] = self.precursors_flat['charge'].values[precursor_flat_lookup]

        available_isotopes = utils.get_isotope_columns(self.precursors_flat.columns)
        available_isotope_columns = [f'i_{i}' for i in available_isotopes]

        for col in available_isotope_columns:
            df[col] = self.precursors_flat[col].values[precursor_flat_lookup]

        return df
    
from tqdm import tqdm
    
@alphatims.utils.pjit()
def _executor(
        i,
        eg_container,
        jit_data, 
        fragment_container,
        kernel, 
        rt_tolerance,
        mobility_tolerance,
        mz_tolerance,
        candidate_count, 
        debug,
        exclude_shared_fragments,
        top_k_fragments,
        top_k_precursors
    ):
    """
    Helper function.
    Is decorated with alphatims.utils.pjit to enable parallel execution.
    """

    eg_container[i].process(
        jit_data, 
        fragment_container,
        kernel, 
        rt_tolerance,
        mobility_tolerance,
        mz_tolerance,
        candidate_count, 
        debug,
        exclude_shared_fragments,
        top_k_fragments,
        top_k_precursors
    )

@nb.njit
def calculate_dia_cycle_mask(
        cycle : np.ndarray,
        quad_slices : np.ndarray
    ):

    """ Calculate the DIA cycle quadrupole mask for each score group.

    Parameters
    ----------

    cycle : np.ndarray
        The DIA mz cycle as part of the bruker.TimsTOF object. (n_frames * n_scans)

    quad_slices : np.ndarray
        The quadrupole slices for each score group. (n_score_groups, 2)

    Returns
    -------

    np.ndarray
        The DIA cycle quadrupole mask for each score group. (n_score_groups, n_frames * n_scans)
    """

    n_score_groups = quad_slices.shape[0]

    dia_mz_cycle = cycle.reshape(-1, 2)

    mz_mask = np.zeros((n_score_groups, len(dia_mz_cycle)), dtype=np.bool_)
    for i, (mz_start, mz_stop) in enumerate(dia_mz_cycle):
        for j, (quad_mz_start, quad_mz_stop) in enumerate(quad_slices):
            if (quad_mz_start <= mz_stop) and (quad_mz_stop >= mz_start):
                mz_mask[j, i] = True

    return mz_mask



@nb.njit
def calculate_score_group_limits(
        precursor_mz, 
        precursor_intensity
    ):

    quadrupole_mz = np.zeros((1, 2))

    mask = precursor_intensity > 0

    quadrupole_mz[0,0] = precursor_mz[mask].min()
    quadrupole_mz[0,1] = precursor_mz[mask].max()

    return quadrupole_mz

@nb.njit
def get_push_indices(
        jit_data,
        frame_limits,
        scan_limits,
        cycle_mask,
    ):

    n_score_groups = cycle_mask.shape[0]

    push_indices = []
    #score_group = []
    absolute_precursor_cycle = []
    len_dia_mz_cycle = len(jit_data.dia_mz_cycle)
    

    frame_start, frame_stop, frame_step = frame_limits[0]
    scan_start, scan_stop, scan_step = scan_limits[0]
    for frame_index in range(frame_start, frame_stop, frame_step):

        for scan_index in range(scan_start, scan_stop, scan_step):

            push_index = frame_index * jit_data.scan_max_index + scan_index
            # subtract a whole frame if the first frame is zero
            if jit_data.zeroth_frame:
                cyclic_push_index = push_index - jit_data.scan_max_index
            else:
                cyclic_push_index = push_index

            # gives the scan index in the dia mz cycle
            scan_in_dia_mz_cycle = cyclic_push_index % len_dia_mz_cycle

            # check fragment push indices
            for i in range(n_score_groups):
                if cycle_mask[i,scan_in_dia_mz_cycle]:
                    precursor_cycle = jit_data.dia_precursor_cycle[scan_in_dia_mz_cycle]
                    absolute_precursor_cycle.append(precursor_cycle)
                    push_indices.append(push_index)
                    #score_group.append(i+1)

    return np.array(push_indices), np.array(absolute_precursor_cycle)

# deprecated
@nb.njit()
def get_dense_hybrid(
        jit_data,
        frame_limits,
        scan_limits,
        precursor_mz,
        precursor_ppm,
        fragment_mz,
        fragment_ppm,
        quadrupole_mz
):
    precursor_cycle_mask = calculate_dia_cycle_mask(
        jit_data.cycle,
        np.array([[-1., -1.]])
    )

    fragment_cycle_mask = calculate_dia_cycle_mask(
        jit_data.cycle,
        quadrupole_mz
    )

    push_indices, source_indices, absolute_precursor_index = get_push_indices(
        jit_data,
        frame_limits,
        scan_limits,
        fragment_cycle_mask,
        precursor_cycle_mask,
    )

    precursor_tof_limits = utils.make_slice_2d( 
        jit_data.return_tof_indices( 
            utils.mass_range(
                precursor_mz, precursor_ppm
            )
        )
    )

    fragment_tof_limits = utils.make_slice_2d(
        jit_data.return_tof_indices(
            utils.mass_range(
                fragment_mz, fragment_ppm
            )
        )
    )


    precursor_index = np.unique(absolute_precursor_index)
    n_precursor_indices = len(precursor_index)
    precursor_index_reverse = np.zeros(np.max(precursor_index)+1, dtype=np.int64)
    precursor_index_reverse[precursor_index] = np.arange(len(precursor_index))

    mobility_len = scan_limits[0,1] - scan_limits[0,0]

    cycle_start = int(frame_limits[0,0]//jit_data.cycle.shape[1])
    cycle_stop = int(frame_limits[0,1]//jit_data.cycle.shape[1])
    cycle_len =  cycle_stop - cycle_start
    n_precursors = len(precursor_tof_limits)
    n_fragments = len(fragment_tof_limits)

    precursor_dense = np.zeros(
        (
            2, 
            n_precursors,
            1,
            mobility_len,
            cycle_len
        ),
        dtype=np.float32
    )

    precursor_dense[1] = precursor_ppm

    fragments_dense = np.zeros(
        (
            2,
            n_fragments,
            n_precursor_indices,
            mobility_len,
            cycle_len
        ),
        dtype=np.float32
    )

    fragments_dense[1] = fragment_ppm
    
    for push_index, source_index, absolute_precursor_index in zip(push_indices, source_indices, absolute_precursor_index):
        
        start = jit_data.push_indptr[push_index]
        end = jit_data.push_indptr[push_index + 1]
        idx = start

        if jit_data.zeroth_frame:
            cycle_index = push_index - jit_data.scan_max_index
        else:
            cycle_index = push_index

        absolute_cycle_index = cycle_index // (jit_data.scan_max_index * jit_data.cycle.shape[1])
        relative_cycle_index = absolute_cycle_index - cycle_start

        absolute_scan_index = push_index % jit_data.scan_max_index
        relative_scan_index = absolute_scan_index - scan_limits[0,0]

        relative_precursor_index = precursor_index_reverse[absolute_precursor_index]

        if source_index == 0:
            tof_limits = precursor_tof_limits[:3]
            output_matrix = precursor_dense
        else:
            tof_limits = fragment_tof_limits[:12]
            output_matrix = fragments_dense

        # precursor
        
        for i, (tof_start, tof_stop, tof_step) in enumerate(tof_limits):
            # Instead of leaving it at the end of the first tof slice it's reset to the start to allow for 
            # Overlap of tof slices
            start_idx = np.searchsorted(jit_data.tof_indices[idx: end], tof_start)
            idx += start_idx
            tof_value = jit_data.tof_indices[idx]

            intensity = 0
            weighted_error = 0

            while (tof_value < tof_stop) and (idx < end):
                if tof_value in range(tof_start, tof_stop, tof_step):

                    intensity += jit_data.intensity_values[idx]
                    mz = jit_data.mz_values[tof_value]
                    weighted_error += (mz - precursor_mz[i])/mz * 1e6 * jit_data.intensity_values[idx]

                idx += 1
                tof_value = jit_data.tof_indices[idx]

            if intensity > 0:
                pass
                output_matrix[0,i,relative_precursor_index,relative_scan_index, relative_cycle_index] = intensity
                output_matrix[1,i,relative_precursor_index,relative_scan_index, relative_cycle_index] = weighted_error / intensity
    
        idx = start + start_idx

    return precursor_dense, fragments_dense

# deprecated
@nb.njit
def filter_push(
    tof_limits, 
    push_index,
    tof_indptr, 
    push_query,
    precursor_index
    
):
    
    tof_slice = []
    values = []
    push_indices = []
    precursor_indices = []
    tof_indices= []

    for j, (tof_start, tof_stop, tof_step) in enumerate(tof_limits):
        
        for tof_index in range(tof_start, tof_stop, tof_step):
            
            start = tof_indptr[tof_index]
            stop = tof_indptr[tof_index + 1]

            i = 0
            idx = int(start)

            while (idx < stop) and (i < len(push_query)):
                    
                    if push_query[i] < push_index[idx]:
                        i += 1

                    else:
                        if push_query[i] == push_index[idx]:
                            tof_slice.append(j)
                            values.append(idx)
                            push_indices.append(push_index[idx])
                            precursor_indices.append(precursor_index[i])
                            tof_indices.append(tof_index)
                        
                        idx = idx + 1


        
        

    return np.array(tof_slice), np.array(values), np.array(push_indices), np.array(precursor_indices), np.array(tof_indices)

@nb.njit
def assemble_push(
    tof_limits, 
    mz_values,
    push_index,
    tof_indptr, 
    push_query,
    precursor_index,
    frame_limits,
    scan_limits,
    ppm_background,
    jit_data
):  
    if len(precursor_index) == 0:
         return np.empty((0,0,0,0,0),dtype=np.float32)
    
    unique_precursor_index = np.unique(precursor_index)
    precursor_index_reverse = np.zeros(np.max(unique_precursor_index)+1, dtype=np.int64)
    precursor_index_reverse[unique_precursor_index] = np.arange(len(unique_precursor_index))

    relative_precursor_index = precursor_index_reverse[precursor_index]

    n_precursor_indices = len(unique_precursor_index)
    n_tof_slices = len(tof_limits)

    # scan valuesa
    mobility_start = int(scan_limits[0,0])
    mobility_stop = int(scan_limits[0,1])
    mobility_len = mobility_stop - mobility_start

    # cycle values
    precursor_cycle_start = int(frame_limits[0,0]-jit_data.zeroth_frame)//jit_data.cycle.shape[1]
    precursor_cycle_stop = int(frame_limits[0,1]-jit_data.zeroth_frame)//jit_data.cycle.shape[1]
    precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

    dense_output = np.zeros(
        (
            2, 
            n_tof_slices,
            n_precursor_indices,
            mobility_len,
            precursor_cycle_len
        ), 
        dtype=np.float32
    )

    dense_output[1,:,:,:,:] = ppm_background

    for j, (tof_start, tof_stop, tof_step) in enumerate(tof_limits):

        library_mz_value = mz_values[j]
        
        for tof_index in range(tof_start, tof_stop, tof_step):

            measured_mz_value = jit_data.mz_values[tof_index]
            
            start = tof_indptr[tof_index]
            stop = tof_indptr[tof_index + 1]

            i = 0
            idx = int(start)

            while (idx < stop) and (i < len(push_query)):
                    
                    if push_query[i] < push_index[idx]:
                        i += 1

                    else:
                        if push_query[i] == push_index[idx]:
                            
                            frame_index = push_index[idx] // jit_data.scan_max_index
                            scan_index = push_index[idx] % jit_data.scan_max_index
                            precursor_cycle_index = (frame_index-jit_data.zeroth_frame)//jit_data.cycle.shape[1]

                            relative_scan = scan_index - mobility_start
                            relative_precursor = precursor_cycle_index - precursor_cycle_start

                            accumulated_intensity = dense_output[0,j,relative_precursor_index[i],relative_scan,relative_precursor]
                            accumulated_error = dense_output[1,j,relative_precursor_index[i],relative_scan,relative_precursor]

                            new_intensity = jit_data.intensity_values_t[idx]
                            new_error = (measured_mz_value - library_mz_value) / library_mz_value * 10**6

                            weighted_error = (accumulated_error * accumulated_intensity + new_error * new_intensity)/(accumulated_intensity + new_intensity)
                            
                            dense_output[0,j,relative_precursor_index[i],relative_scan,relative_precursor] = accumulated_intensity + new_intensity
                            dense_output[1,j,relative_precursor_index[i],relative_scan,relative_precursor] = weighted_error
                        
                        idx = idx + 1

    return dense_output

@nb.njit
def build_features(
    smooth_precursor,
    smooth_fragment,
    precursor_intensity,
    fragment_intensity
):
    
    n_features = 8

    features = np.zeros(
        (
            n_features,
            smooth_precursor.shape[2],
            smooth_fragment.shape[3],
            
        ),
        dtype=np.float32
    )

    precursor_kernel = precursor_intensity.reshape(-1, 1, 1)
    fragment_kernel = fragment_intensity.reshape(-1, 1, 1)

    fragment_binary = smooth_fragment[0] > 2
    #fragment_binary_sum = np.sum(fragment_binary, axis=0)
    fragment_binary_weighted = np.sum(fragment_binary * fragment_kernel, axis=0)

    precursor_binary = smooth_precursor[0] > 2
    #precursor_binary_sum = np.sum(precursor_binary, axis=0)
    precursor_binary_weighted = np.sum(precursor_binary * precursor_kernel, axis=0)

    precursor_dot = np.sum(smooth_precursor[0] * precursor_kernel, axis=0)
    precursor_dot_mean = np.mean(precursor_dot)
    precursor_norm = precursor_dot/(precursor_dot_mean+0.0001)

    fragment_dot = np.sum(smooth_fragment[0] * fragment_kernel, axis=0)
    fragment_dot_mean = np.mean(fragment_dot)
    fragment_norm = fragment_dot/(fragment_dot_mean+0.001)

    fragment_mass_error = np.sum(np.abs(smooth_fragment[1]), axis=0)
    fragment_mass_error_max = np.max(fragment_mass_error)
    fragment_mass_error_norm = 1-(fragment_mass_error/fragment_mass_error_max)

    precursor_mass_error = np.sum(np.abs(smooth_precursor[1]), axis=0)
    precursor_mass_error_max = np.max(precursor_mass_error)
    precursor_mass_error_norm = 1-(precursor_mass_error/precursor_mass_error_max)

    # top fragment
    frag_order = np.argsort(fragment_intensity)

    features[0] = fragment_binary_weighted
    features[1] = fragment_norm
    features[2] = fragment_mass_error_norm
    features[3] = smooth_fragment[0, frag_order[0]]
    features[4] = smooth_fragment[0][frag_order[:3]].sum(axis=0)
    features[5] = precursor_binary_weighted
    features[6] = precursor_norm
    features[7] = precursor_mass_error_norm

    return features

@nb.njit
def join_close_peaks(
    peak_scan_list,
    peak_cycle_list,
    peak_score_list,
    scan_tolerance,
    cycle_tolerance
    ):
    """
    find peaks that are close in scan and cycle and return
    """
    n_peaks = peak_scan_list.shape[0]
    peak_mask = np.ones(n_peaks, dtype=np.bool_)
    for peak_idx in range(n_peaks):
        if not peak_mask[peak_idx]:
            continue
        scan = peak_scan_list[peak_idx]
        cycle = peak_cycle_list[peak_idx]
        score = peak_score_list[peak_idx]
        for other_peak_idx in range(peak_idx+1, n_peaks):
            if not peak_mask[other_peak_idx]:
                continue
            other_scan = peak_scan_list[other_peak_idx]
            other_cycle = peak_cycle_list[other_peak_idx]
            other_score = peak_score_list[other_peak_idx]
            if abs(scan-other_scan) <= scan_tolerance and abs(cycle-other_cycle) <= cycle_tolerance:
                if score > other_score:
                    peak_mask[other_peak_idx] = False
                else:
                    peak_mask[peak_idx] = False
                    break

    return peak_mask



def plot_candidates(score, candidates, jit_data, scan_limits, frame_limits):
    plt.imshow(score)
    
    absolute_scan = np.array([c.scan_center for c in candidates])
    absolute_frame = np.array([c.frame_center for c in candidates])

    relative_scan = absolute_scan - scan_limits[0,0]
    relative_frame = (absolute_frame - frame_limits[0,0])//jit_data.cycle.shape[1]

    plt.scatter(relative_frame, relative_scan, c='red', s=1)

    absolute_scan_start = np.array([c.scan_start for c in candidates])
    absolute_scan_stop = np.array([c.scan_stop for c in candidates])
    absolute_frame_start = np.array([c.frame_start for c in candidates])
    absolute_frame_stop = np.array([c.frame_stop for c in candidates])

    relative_scan_start = absolute_scan_start - scan_limits[0,0]
    relative_scan_stop = absolute_scan_stop - scan_limits[0,0]
    relative_frame_start = (absolute_frame_start - frame_limits[0,0])//jit_data.cycle.shape[1]
    relative_frame_stop = (absolute_frame_stop - frame_limits[0,0])//jit_data.cycle.shape[1]

    
    ax = plt.gca()
    for i in range(len(candidates)):
        rect = patches.Rectangle(
            (relative_frame_start[i], relative_scan_start[i]),
            relative_frame_stop[i]-relative_frame_start[i],
            relative_scan_stop[i]-relative_scan_start[i],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    plt.show()

@nb.njit
def build_candidates(
    dense_precursors,
    dense_fragments,
    precursor_intensity,
    fragment_intensity,
    eg,
    kernel,
    jit_data,
    candidate_count = 3,
    debug = False,
    weights = None,
    mean = None,
    std = None,
    ):

    cycle_length = jit_data.cycle.shape[1]

    candidates = nb.typed.List.empty_list(candidate_type)

    if weights is None:
        feature_weights = np.ones(8)
    else:
        feature_weights = weights

    feature_weights = feature_weights.reshape(-1,1,1)


    smooth_precursor = numeric.convolve_fourier_a1(dense_precursors, kernel)
    smooth_fragment = numeric.convolve_fourier_a1(dense_fragments, kernel)

    if not smooth_precursor.shape == dense_precursors.shape:
        print(smooth_precursor.shape, dense_precursors.shape)
        print('smooth_precursor shape does not match dense_precursors shape')
    if not smooth_fragment.shape == dense_fragments.shape:
        print(smooth_fragment.shape, dense_fragments.shape)
        print('smooth_fragment shape does not match dense_fragments shape')

    # works until here
    

    #with nb.objmode:
    #    for i in range(smooth_fragment.shape[1]):
    #        plt.imshow(smooth_fragment[0,i])
    #        plt.show()

    #    for i in range(smooth_precursor.shape[1]):
    #        plt.imshow(smooth_precursor[0,i])
    #        plt.show()
    

    feature_matrix = build_features(
        smooth_precursor,
        smooth_fragment,
        precursor_intensity,
        fragment_intensity,
    ).astype('float32')

    # get mean and std to normalize features
    # if trained, use the mean and std from training
    # otherwise calculate the mean and std from the current data
    if mean is None:
        feature_mean = utils.amean1(feature_matrix)
    else:
        feature_mean = mean
    feature_mean = feature_mean.reshape(-1,1,1)

    if std is None:
        feature_std = utils.astd1(feature_matrix).reshape(-1,1,1)
    else:
        feature_std = std
    feature_std = feature_std.reshape(-1,1,1)

    # make sure that mean, std and weights have the same shape
    if not (feature_std.shape == feature_mean.shape == feature_weights.shape):
        raise ValueError("feature_mean, feature_std and feature_weights must have the same shape")
    

    feature_matrix_norm = feature_weights * (feature_matrix - feature_mean)/(feature_std+1e-6)
    score = np.sum(feature_matrix_norm, axis=0)

    peak_scan_list, peak_cycle_list, peak_score_list = utils.find_peaks(
        score, top_n=candidate_count
    )


    peak_mask = join_close_peaks(peak_scan_list, peak_cycle_list, peak_score_list, 3, 3)
    
    peak_scan_list = peak_scan_list[peak_mask]
    peak_cycle_list = peak_cycle_list[peak_mask]
    peak_score_list = peak_score_list[peak_mask]

    # works until here

    

    for candidate_rank, (scan_relative, cycle_relative, candidate_score) in enumerate(zip(peak_scan_list, peak_cycle_list, peak_score_list)):

        
        scan_limits_relative, cycle_limits_relative = numeric.symetric_limits_2d(
            score, 
            scan_relative, 
            cycle_relative, 
            f_mobility = 0.99,
            f_rt = 0.99,
            center_fraction = 0.05,
            min_size_mobility = 10,
            min_size_rt = 5,
            max_size_mobility = 40,
            max_size_rt = 30,
        )

    # does not work anymore

        scan_limits_absolute = numeric.wrap1(scan_limits_relative + eg.scan_limits[0,0], jit_data.scan_max_index)
        frame_limits_absolute = numeric.wrap1(cycle_limits_relative * cycle_length + eg.frame_limits[0,0], jit_data.frame_max_index)

        scan_absolute = numeric.wrap0(scan_relative + eg.scan_limits[0,0], jit_data.scan_max_index)
        frame_absolute = numeric.wrap0(cycle_relative * cycle_length + eg.frame_limits[0,0], jit_data.frame_max_index)

        features = np.zeros(feature_matrix.shape[0], dtype='float32')
        for j in range(feature_matrix.shape[0]):
            features[j] = numeric.get_mean0(feature_matrix[j], scan_relative, cycle_relative)

        mass_error = np.zeros(smooth_precursor.shape[1], dtype='float32')
        for j in range(smooth_precursor.shape[0]):
            mass_error[j] = numeric.get_mean_sparse0(smooth_precursor[1,j], scan_relative, cycle_relative, 110)

        # iterate all precursors within this score group
        for precursor_idx in eg.precursor_idx:

            candidates.append(
                Candidate(
                    eg.elution_group_idx,
                    eg.score_group_idx,
                    precursor_idx,
                    candidate_rank,
                    candidate_score,
                    eg.precursor_mz[0],
                    eg.precursor_decoy[0],
                    features,
                    scan_absolute,
                    scan_limits_absolute[0],
                    scan_limits_absolute[1],
                    frame_absolute,
                    frame_limits_absolute[0],
                    frame_limits_absolute[1]
                )
            )

    if debug:
        with nb.objmode():
            plot_candidates(score, candidates, jit_data, eg.scan_limits, eg.frame_limits)

    return candidates
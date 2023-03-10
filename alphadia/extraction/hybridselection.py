from alphadia.extraction import utils
from alphadia.extraction.numba import fragments, numeric
from alphadia.extraction.utils import fourier_filter
from alphadia.extraction.candidateselection import peak_boundaries_symmetric, GaussianFilter
import numba as nb
import numpy as np
import pandas as pd
import logging
import alphatims

import matplotlib.pyplot as plt



@nb.njit()
def calculate_score(dense_precursors, dense_fragments, expected_intensity, kernel):

    precursor_intensity = numeric.fourier_a1(dense_precursors[0], kernel)
    #print(precursor_i.shape)

    fragment_intensity = numeric.fourier_a1(dense_fragments[0], kernel)

    fragment_kernel = expected_intensity.reshape(-1, 1, 1, 1)

    fragment_dot = np.sum(fragment_intensity * fragment_kernel, axis=0)

    s = fragment_dot * precursor_intensity[0] * np.sum(fragment_intensity, axis=0)

    return s[0]

@nb.experimental.jitclass()
class HybridElutionGroup:

    elution_group_idx: nb.uint32
    precursor_idx: nb.uint32[::1]
    channel: nb.uint32[::1]
    frag_start_stop_idx: nb.uint32[:,::1]
    
    rt: nb.float64
    mobility: nb.float64
    charge: nb.uint8

    decoy: nb.uint8[::1]
    mz: nb.float64[::1]
    
    #isotope_apex_offset: nb.int8[::1]
    #top_isotope_mz: nb.float64[::1]

    isotope_intensity: nb.float32[:, :]
    isotope_mz: nb.float32[:, ::1]

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
    dense_fragments : nb.float64[:, :, :, :, ::1]
    dense_precursors : nb.float64[:, :, :, :, ::1]

    def __init__(
            self, 
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

        self.elution_group_idx = elution_group_idx
        self.precursor_idx = precursor_idx
        self.rt = rt
        self.mobility = mobility
        self.charge = charge
        self.decoy = decoy
        self.mz = mz
        self.channel = channel
        #self.isotope_apex_offset = isotope_apex_offset
        #self.top_isotope_mz = mz + isotope_apex_offset * 1.0033548350700006 / charge

        self.frag_start_stop_idx = frag_start_stop_idx
        self.isotope_intensity = isotope_intensity

        

    def __str__(self):
        with nb.objmode(r='unicode_type'):
            r = f'ElutionGroup(\nelution_group_idx: {self.elution_group_idx},\nprecursor_idx: {self.precursor_idx}\n)'
        return r

    def sort_by_mz(self):
        """
        Sort all precursor arrays by m/z
        
        """
        mz_order = np.argsort(self.mz)
        self.mz = self.mz[mz_order]
        self.decoy = self.decoy[mz_order]
        #self.isotope_apex_offset = self.isotope_apex_offset[mz_order]
        #self.top_isotope_mz = self.top_isotope_mz[mz_order]
        self.precursor_idx = self.precursor_idx[mz_order]
        self.frag_start_stop_idx = self.frag_start_stop_idx[mz_order]

    def assemble_isotope_mz(self):
        """
        Assemble the isotope m/z values from the precursor m/z and the isotope
        offsets.
        """
        offset = np.arange(self.isotope_intensity.shape[1]) * 1.0033548350700006 / self.charge
        self.isotope_mz = np.expand_dims(self.mz, 1).astype(np.float32) + np.expand_dims(offset,0).astype(np.float32)

    def trim_isotopes(self):

        elution_group_isotopes = np.sum(self.isotope_intensity, axis=0)/self.isotope_intensity.shape[0]
        self.isotope_intensity = self.isotope_intensity[:,elution_group_isotopes>0.1]

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
        precursor_mz_limits = utils.mass_range(self.mz, tolerance)
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


    def process(
        self, 
        jit_data, 
        fragment_container,
        kernel, 
        rt_tolerance,
        mobility_tolerance,
        mz_tolerance,
        candidate_count, 
        debug
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
        

        self.sort_by_mz()
        self.trim_isotopes()
        self.assemble_isotope_mz()

        fragment_idx_slices = utils.make_slice_2d(
            self.frag_start_stop_idx
        )
        self.fragments = fragment_container.slice(fragment_idx_slices)
        self.fragments.sort_by_mz()


        self.determine_frame_limits(jit_data, rt_tolerance)
        self.determine_scan_limits(jit_data, mobility_tolerance)
        self.determine_precursor_tof_limits(jit_data, mz_tolerance)
        self.determine_fragment_tof_limits(jit_data, mz_tolerance)

        #return
        
        precursor_quad_limits = utils.mass_range(self.mz, 500)
        
        dense_fragments, _ = jit_data.get_dense_fragments(
            self.frame_limits,
            self.scan_limits,
            precursor_quad_limits,
            self.fragments.precursor_idx,
            self.fragment_tof_limits
        )

        return

        # (2, n_isotopes, n_observations, n_scans, n_frames)
        dense_precursors, _ = jit_data.get_dense(
            self.frame_limits,
            self.scan_limits,
            self.precursor_tof_limits,
            np.array([[-1.,-1.]]),
            False
        )

        candidate_precursor_idx = []
        candidate_mass_error = []
        candidate_fraction_nonzero = []
        candidate_intensity = []
        candidate_scan_center = []
        candidate_frame_center = []

        # This is the theoretical maximum number of candidates
        # As we don't know how many candidates we will have, we will
        # have to resize the arrays later
        n_candidates = len(self.precursor_idx) * candidate_count
        self.candidate_scan_limit = np.zeros((n_candidates,2), dtype=nb.int64)
        self.candidate_frame_limit = np.zeros((n_candidates,2), dtype=nb.int64)

        candidate_idx = 0

        if dense_fragments.shape[3] > kernel.shape[0] and dense_fragments.shape[4] > kernel.shape[1]:

            score = calculate_score(
                dense_precursors,
                dense_fragments,
                self.fragments.intensity,
                kernel
            )

            for i, idx in enumerate(self.precursor_idx):

                peak_scan_list, peak_cycle_list, peak_intensity_list = utils.find_peaks(
                    score, top_n=candidate_count
                )

                for j, (scan, cycle, intensity) in enumerate(
                    zip(
                        peak_scan_list, 
                        peak_cycle_list, 
                        peak_intensity_list
                        )
                    ):

                    limit_scan, limit_cycle = peak_boundaries_symmetric(
                        score, 
                        scan, 
                        cycle, 
                        f_mobility=0.95,
                        f_rt=0.99,
                        center_fraction=0.05,
                        min_size_mobility=6,
                        min_size_rt=3,
                        max_size_mobility = 40,
                        max_size_rt = 30,
                    )

                    fraction_nonzero, mz = utils.get_precursor_mz(
                        dense_precursors[0,i,0],
                        dense_precursors[1,i,0],
                        scan, 
                        cycle
                    )
                    mass_error = (mz - self.mz[i])/mz*10**6

                    if intensity > 1:

                        candidate_precursor_idx.append(idx)
                        candidate_mass_error.append(mass_error)
                        candidate_fraction_nonzero.append(fraction_nonzero)
                        candidate_intensity.append(intensity)
                        candidate_scan_center.append(scan)
                        candidate_frame_center.append(cycle)

                        self.candidate_scan_limit[candidate_idx] = limit_scan
                        self.candidate_frame_limit[candidate_idx] = limit_cycle

                        candidate_idx += 1

        self.candidate_precursor_idx = np.array(candidate_precursor_idx)
        self.candidate_mass_error = np.array(candidate_mass_error)
        self.candidate_fraction_nonzero = np.array(candidate_fraction_nonzero)
        self.candidate_intensity = np.array(candidate_intensity)

        self.candidate_scan_center = np.array(candidate_scan_center)
        self.candidate_frame_center = np.array(candidate_frame_center)

        # resize the arrays to the actual number of candidates
        self.candidate_scan_limit = self.candidate_scan_limit[:candidate_idx]
        self.candidate_frame_limit = self.candidate_frame_limit[:candidate_idx]






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
            thread_count = 20,
            kernel_sigma_rt = 5,
            kernel_sigma_mobility = 12,
            debug = False
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

        Returns
        -------

        pandas.DataFrame
            dataframe containing the extracted candidates
        """
        self.dia_data = dia_data
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
        elution_group_container = self.assemble_elution_groups(self.precursors_flat)
        fragment_container = self.assemble_fragments()

        # if debug mode, only iterate over 10 elution groups
        iterator_len = min(10,len(elution_group_container)) if self.debug else len(elution_group_container)
        thread_count = 1 if self.debug else self.thread_count

        alphatims.utils.set_threads(thread_count)

        _executor(
            range(iterator_len), 
            elution_group_container,
            self.dia_data.jitclass(), 
            fragment_container, 
            self.kernel, 
            self.rt_tolerance,
            self.mobility_tolerance,
            self.mz_tolerance,
            self.candidate_count,
            self.debug
        )
   
        #df = self.assemble_candidates(elution_group_container)

        return elution_group_container

        #return df
        df = self.append_precursor_information(df)
        #self.log_stats(df)
        return df
    
    def assemble_fragments(self):
            
            if 'cardinality' in self.fragments_flat.columns:
                cardinality = self.fragments_flat['cardinality'].values.astype(np.uint8)
            
            else:
                logging.warning('Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1.')
                cardinality = np.ones(len(self.fragments_flat), dtype=np.uint8)

            return fragments.FragmentContainer(
                self.fragments_flat['mz_library'].values.astype(np.float32),
                self.fragments_flat[self.fragment_mz_column].values.astype(np.float32),
                self.fragments_flat['intensity'].values.astype(np.float32),
                self.fragments_flat['type'].values.astype(np.uint8),
                self.fragments_flat['loss_type'].values.astype(np.uint8),
                self.fragments_flat['charge'].values.astype(np.uint8),
                self.fragments_flat['number'].values.astype(np.uint8),
                self.fragments_flat['position'].values.astype(np.uint8),
                cardinality
            )

    def assemble_elution_groups(
            self,
            precursors_flat,
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

        precursors_sorted = precursors_flat.sort_values('elution_group_idx').copy()

        @nb.njit(debug=True)
        def assemble_njit(
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
            elution_group = elution_group_idx[0]
            elution_group_start = 0
            elution_group_stop = 0

            eg_list = []
            
            while elution_group_stop < len(elution_group_idx)-1:
                
                elution_group_stop += 1

                if elution_group_idx[elution_group_stop] != elution_group:
                        
                    eg_list.append(HybridElutionGroup(    
                        elution_group,
                        precursor_idx[elution_group_start:elution_group_stop],
                        channel[elution_group_start:elution_group_stop],
                        flat_frag_start_stop_idx[elution_group_start:elution_group_stop],
                        rt_values[elution_group_start],
                        mobility_values[elution_group_start],
                        charge[elution_group_start],
                        decoy[elution_group_start:elution_group_stop],
                        precursor_mz[elution_group_start:elution_group_stop],
                        isotope_intensity[elution_group_start:elution_group_stop]
                    ))

                    elution_group_start = elution_group_stop
                    elution_group = elution_group_idx[elution_group_start]
                    
            egs = nb.typed.List(eg_list)
            return HybridElutionGroupContainer(egs)

        return assemble_njit(
            precursors_sorted['elution_group_idx'].values.astype(np.uint32),
            precursors_sorted['precursor_idx'].values.astype(np.uint32),
            precursors_sorted['channel'].values.astype(np.uint32),
            precursors_sorted[['flat_frag_start_idx','flat_frag_stop_idx']].values.copy().astype(np.uint32),
            precursors_sorted[self.rt_column].values.astype(np.float64),
            precursors_sorted[self.mobility_column].values.astype(np.float64),
            precursors_sorted['charge'].values.astype(np.uint8),
            precursors_sorted['decoy'].values.astype(np.uint8),
            precursors_sorted[self.precursor_mz_column].values.astype(np.float64),
            precursors_sorted[available_isotope_columns].values.copy().astype(np.float32),
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
       
        precursor_idx = []
        elution_group_idx = []
        mass_error = []
        fraction_nonzero = []
        intensity = []

        rt = []
        mobility = []

        candidate_scan_center = []
        candidate_scan_start = []
        candidate_scan_stop = []
        candidate_frame_center = []
        candidate_frame_start = []
        candidate_frame_stop = []

        duty_cycle_length = self.dia_data.cycle.shape[1]

        for i, eg in enumerate(elution_group_container):
            # make sure that the elution group has been processed
            # in debug mode, all elution groups are instantiated but not all are processed
            if len(eg.scan_limits) == 0:
                continue

            n_candidates = len(eg.candidate_precursor_idx)
            elution_group_idx += [eg.elution_group_idx] * n_candidates

            precursor_idx.append(eg.candidate_precursor_idx)
            mass_error.append(eg.candidate_mass_error)
            fraction_nonzero.append(eg.candidate_fraction_nonzero)
            intensity.append(eg.candidate_intensity)

            rt += [eg.rt] * n_candidates
            mobility += [eg.mobility] * n_candidates

            candidate_scan_center.append(eg.candidate_scan_center + eg.scan_limits[0,0])
            candidate_scan_start.append(eg.candidate_scan_limit[:,0]+ eg.scan_limits[0,0])
            candidate_scan_stop.append(eg.candidate_scan_limit[:,1]+ eg.scan_limits[0,0])
            candidate_frame_center.append(eg.candidate_frame_center*duty_cycle_length + eg.frame_limits[0,0])
            candidate_frame_start.append(eg.candidate_frame_limit[:,0]*duty_cycle_length + eg.frame_limits[0,0])
            candidate_frame_stop.append(eg.candidate_frame_limit[:,1]*duty_cycle_length + eg.frame_limits[0,0])


        elution_group_idx = np.array(elution_group_idx)

        precursor_idx = np.concatenate(precursor_idx)
        mass_error = np.concatenate(mass_error)
        fraction_nonzero = np.concatenate(fraction_nonzero)
        intensity = np.concatenate(intensity)

        rt = np.array(rt)
        mobility = np.array(mobility)

        candidate_scan_center = np.concatenate(candidate_scan_center)
        candidate_scan_start = np.concatenate(candidate_scan_start)
        candidate_scan_stop = np.concatenate(candidate_scan_stop)
        candidate_frame_center = np.concatenate(candidate_frame_center)
        candidate_frame_start = np.concatenate(candidate_frame_start)
        candidate_frame_stop = np.concatenate(candidate_frame_stop)

        return pd.DataFrame({
            'precursor_idx': precursor_idx,
            'elution_group_idx': elution_group_idx,
            'mass_error': mass_error,
            'fraction_nonzero': fraction_nonzero,
            'intensity': intensity,
            'scan_center': candidate_scan_center,
            'scan_start': candidate_scan_start,
            'scan_stop': candidate_scan_stop,
            'frame_center': candidate_frame_center,
            'frame_start': candidate_frame_start,
            'frame_stop': candidate_frame_stop,
            self.rt_column: rt,
            self.mobility_column: mobility,
        })
    
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

        if self.rt_column == 'rt_calibrated':
            df['rt_library'] = self.precursors_flat['rt_library'].values[precursor_flat_lookup]

        if self.mobility_column == 'mobility_calibrated':
            df['mobility_library'] = self.precursors_flat['mobility_library'].values[precursor_flat_lookup]

        df['charge'] = self.precursors_flat['charge'].values[precursor_flat_lookup]

        return df
    
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
        debug
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
        debug
    )


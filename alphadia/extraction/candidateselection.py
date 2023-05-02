# native imports
import logging

# alphadia imports
from alphadia.extraction import utils
from alphadia.extraction.utils import fourier_filter

# alpha family imports
import alphatims

# third party imports
import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import patches

#@alphatims.utils.njit()
def symetric_limits(
        array_1d, 
        center, 
        f = 0.95,
        center_fraction = 0.01,
        min_size = 1, 
        max_size = 10,
    ):

    center_intensity = array_1d[center]
    trailing_intensity = center_intensity
    max_len = min(array_1d.shape[0], array_1d.shape[0]-center)
    max_len = int(min(max_len, max_size))

    limit = min_size

    for s in range(min_size,max_len):
        intensity = (array_1d[center-s]+array_1d[center+s])/2
        if intensity < f * trailing_intensity:
            if intensity > center_intensity * center_fraction:
                limit = s
                trailing_intensity = intensity
        else: break

    return np.array([center-limit, center+limit], dtype='int32')


#@alphatims.utils.njit()
def peak_boundaries_symmetric(
        a, 
        scan_center, 
        dia_cycle_center,
        f_mobility = 0.95,
        f_rt = 0.95,
        center_fraction = 0.01,
        min_size_mobility = 3,
        max_size_mobility = 20,
        min_size_rt = 1,
        max_size_rt = 10,
        refine = True
    ):


    mobility_limits = symetric_limits(
        a[:,dia_cycle_center],
        scan_center,
        f = f_mobility,
        center_fraction = center_fraction,
        min_size = min_size_mobility,
        max_size = max_size_mobility,

    )

    dia_cycle_limits = symetric_limits(
        a[scan_center,:],
        dia_cycle_center,
        f = f_rt,
        center_fraction = center_fraction,
        min_size = min_size_rt,
        max_size = max_size_rt
    )

    if refine:

        window = a[mobility_limits[0]:mobility_limits[1],dia_cycle_limits[0]:dia_cycle_limits[1]]
        window_scan_center = scan_center - mobility_limits[0]
        window_dia_cycle_center = dia_cycle_center - dia_cycle_limits[0]

        mobility_limits = symetric_limits(
            np.sum(window, axis=1),
            window_scan_center,
            f = f_mobility,
            center_fraction = center_fraction,
            min_size = min_size_mobility,
            max_size = max_size_mobility,
        ) + mobility_limits[0]

        dia_cycle_limits = symetric_limits(
            np.sum(window, axis=0),
            window_dia_cycle_center,
            f = f_rt,
            center_fraction = center_fraction,
            min_size = min_size_rt,
            max_size = max_size_rt
        ) + dia_cycle_limits[0]

    return mobility_limits, dia_cycle_limits






class MS1CentricCandidateSelection(object):

    def __init__(self, 
            dia_data,
            precursors_flat, 
            rt_tolerance = 30,
            mobility_tolerance = 0.03,
            mz_tolerance = 120,
            candidate_count = 3,
            rt_column = 'rt_library',  
            precursor_mz_column = 'mz_library',
            mobility_column = 'mobility_library',
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
        self.precursors_flat = precursors_flat

        self.debug = debug
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.mobility_tolerance = mobility_tolerance
        self.candidate_count = candidate_count

        self.thread_count = thread_count

        self.rt_column = rt_column
        self.precursor_mz_column = precursor_mz_column
        self.mobility_column = mobility_column

        gaussian_filter = GaussianFilter(
            dia_data
        )
        self.kernel = gaussian_filter.get_kernel()

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
        elution_group_container = self.assemble_elution_groups()

        # if debug mode, only iterate over 10 elution groups
        iterator_len = min(10,len(elution_group_container)) if self.debug else len(elution_group_container)
        thread_count = 1 if self.debug else self.thread_count

        alphatims.utils.set_threads(thread_count)

        _executor(
            range(iterator_len), 
            self.dia_data.jitclass(), 
            elution_group_container, 
            self.kernel, 
            self.rt_tolerance,
            self.mobility_tolerance,
            self.mz_tolerance,
            self.candidate_count,
            self.debug
        )
   
        df = self.assemble_candidates(elution_group_container)
        df = self.append_precursor_information(df)
        self.log_stats(df)
        return df

    def assemble_elution_groups(self):
        """
        Create an ElutionGroup object for every elution group in the precursor dataframe.
        The list of ElutionGroup objects is stored in an ElutionGroupContainer object and returned.

        Returns
        -------
        ElutionGroupContainer
            container object containing a list of ElutionGroup objects
        """
        self.precursors_flat = self.precursors_flat.sort_values('precursor_idx').reset_index(drop=True)
        eg_grouped = self.precursors_flat.groupby('elution_group_idx')

        egs = []
        for i, (name, grouped) in enumerate(eg_grouped):

            if 'isotope_apex_offset' in self.precursors_flat.columns:
                isotope_apex_offset = self.precursors_flat['isotope_apex_offset'].values.astype(np.int8)
            else:
                isotope_apex_offset = np.zeros_like(grouped['precursor_idx'].values).astype(np.int8)

            egs.append(ElutionGroup(
                int(name), 
                grouped['precursor_idx'].values.astype(np.uint32),
                grouped[self.rt_column].values.astype(np.float64)[0],
                grouped[self.mobility_column].values.astype(np.float64)[0],
                grouped['charge'].values.astype(np.uint8)[0],
                grouped['decoy'].values.astype(np.uint8),
                grouped[self.precursor_mz_column].values.astype(np.float64),
                isotope_apex_offset,
                ))

        egs = nb.typed.List(egs)
        return ElutionGroupContainer(egs)
    
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

    def log_stats(
            self,
            df
        ):
        """
        Log statistics about the extracted candidates.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe containing the extracted candidates
        
        """

        # log information
        number_of_precursors = len(self.precursors_flat)
        number_of_decoy_precursors = self.precursors_flat['decoy'].sum()
        number_of_target_precursors = number_of_precursors - number_of_decoy_precursors
        number_of_target_extractions = df[df['decoy'] == False]['precursor_idx'].nunique()
        number_of_decoy_extractions = df[df['decoy'] == True]['precursor_idx'].nunique()

        target_percentage = number_of_target_extractions / number_of_target_precursors * 100
        decoy_percentage = number_of_decoy_extractions / number_of_decoy_precursors * 100

        logging.info(f'Extracted candidates for {number_of_target_extractions} target precursors ({target_percentage:.2f}%)')
        logging.info(f'Extracted candidates for {number_of_decoy_extractions} decoy precursors  ({decoy_percentage:.2f}%)')

@nb.experimental.jitclass()
class ElutionGroup:

    elution_group_idx: nb.uint32
    precursor_idx: nb.uint32[::1]
    
    rt: nb.float64
    mobility: nb.float64
    charge: nb.uint8

    decoy: nb.uint8[::1]
    mz: nb.float64[::1]
    isotope_apex_offset: nb.int8[::1]
    top_isotope_mz: nb.float64[::1]

    frame_limits: nb.uint64[:, ::1]
    scan_limits: nb.uint64[:, ::1]
    tof_limits: nb.uint64[:, ::1]
    
    candidate_precursor_idx: nb.uint32[::1]
    candidate_mass_error: nb.float64[::1]
    candidate_fraction_nonzero: nb.float64[::1]
    candidate_intensity: nb.float32[::1]

    candidate_scan_limit: nb.int64[:, ::1]
    candidate_frame_limit: nb.int64[:, ::1]

    candidate_scan_center: nb.int64[::1]
    candidate_frame_center: nb.int64[::1]

    def __init__(
            self, 
            elution_group_idx,
            precursor_idx,
            rt,
            mobility,
            charge,
            decoy,
            mz,
            isotope_apex_offset
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
        self.isotope_apex_offset = isotope_apex_offset
        self.top_isotope_mz = mz + isotope_apex_offset * 1.0033548350700006 / charge

        self.sort_by_mz()

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
        self.isotope_apex_offset = self.isotope_apex_offset[mz_order]
        self.top_isotope_mz = self.top_isotope_mz[mz_order]
        self.precursor_idx = self.precursor_idx[mz_order]

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
            utils.expand_if_odd(
                jit_data.return_frame_indices(
                    rt_limits,
                    True
                )
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
            utils.expand_if_odd(
                jit_data.return_scan_indices(
                    mobility_limits
                )
            )
        )

    def determine_tof_limits(
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
        mz_limits = utils.mass_range(self.top_isotope_mz, tolerance)
        self.tof_limits = utils.make_slice_2d(jit_data.return_tof_indices(
            mz_limits
        ))

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

        self.determine_frame_limits(jit_data, rt_tolerance)
        self.determine_scan_limits(jit_data, mobility_tolerance)
        self.determine_tof_limits(jit_data, mz_tolerance)

        # (2, n_isotopes, n_observations, n_scans, n_frames)
        dense, precursor_index = jit_data.get_dense(
            self.frame_limits,
            self.scan_limits,
            self.tof_limits,
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

        if dense.shape[3] > kernel.shape[0] and dense.shape[4] > kernel.shape[1]:

            for i, idx in enumerate(self.precursor_idx):
                
                lower_scan_limit, upper_scan_limit = self.determine_fragment_scan_limits(np.array([[self.top_isotope_mz[i]-0.1, self.top_isotope_mz[i]+0.1]]), jit_data)

                lower_scan_limit -= self.scan_limits[0,0]
                lower_scan_limit = max(lower_scan_limit, 0)

                upper_scan_limit -= self.scan_limits[0,0]
                upper_scan_limit = min(upper_scan_limit, dense.shape[3])

                #dense[0,i,0,:lower_scan_limit] = 0
                #dense[0,i,0, upper_scan_limit:] = 0

            
            smooth_dense = fourier_filter(dense, kernel)

            for i, idx in enumerate(self.precursor_idx):

                smooth_precursor = smooth_dense[i]
                smooth_precursor = np.sum(smooth_precursor, axis=0)

                peak_scan_list, peak_cycle_list, peak_intensity_list = utils.find_peaks(
                    smooth_precursor, top_n=candidate_count
                )

                for j, (scan, cycle, intensity) in enumerate(
                    zip(
                        peak_scan_list, 
                        peak_cycle_list, 
                        peak_intensity_list
                        )
                    ):

                    limit_scan, limit_cycle = peak_boundaries_symmetric(
                        smooth_precursor, 
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
                        dense[0,i,0],
                        dense[1,i,0],
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
        
        if debug:
            if dense.shape[3] >= kernel.shape[0] or dense.shape[4] >= kernel.shape[1]:
                self.visualize_candidates(dense, smooth_dense)

    def visualize_candidates(
        self, 
        dense, 
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
                axs[j,0].imshow(
                dense[0,j,0], 
                aspect='auto'
                )
                axs[j,0].set_xlabel('cycle')
                axs[j,0].set_ylabel('scan')
                axs[j,0].set_title(f'- RAW DATA - elution group: {self.elution_group_idx}, precursor: {idx}')

                axs[j,1].imshow(smooth_dense[j,0], aspect='auto')
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
class ElutionGroupContainer:
    
        elution_groups: nb.types.ListType(ElutionGroup.class_type.instance_type)
    
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

@alphatims.utils.pjit()
def _executor(
        i,
        jit_data, 
        eg_container, 
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
        kernel, 
        rt_tolerance,
        mobility_tolerance,
        mz_tolerance,
        candidate_count, 
        debug
    )




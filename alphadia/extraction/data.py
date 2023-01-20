import alphatims.utils
import alphatims.bruker
import numpy as np
import logging
from numba.core import types
from numba.typed import Dict
from numba.experimental import jitclass


class TimsTOFDIA_(alphatims.bruker.TimsTOF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def jitclass(self):
        return TimsTOFJIT(
            self._accumulation_times,
            self._cycle,
            self._dia_mz_cycle,
            self._dia_precursor_cycle,
            self._frame_max_index,
            self._intensity_corrections,
            self._intensity_max_value,
            self._intensity_min_value,
            self._intensity_values,
            self._max_accumulation_time,
            self._mobility_max_value,
            self._mobility_min_value,
            self._mobility_values,
            self._mz_values,
            self._precursor_indices,
            self._precursor_max_index,
            self._push_indptr,
            self._quad_indptr,
            self._quad_max_mz_value,
            self._quad_min_mz_value,
            self._quad_mz_values,
            self._raw_quad_indptr,
            self._rt_values,
            self._scan_max_index,
            self._tof_indices,
            self._tof_max_index,
            self._use_calibrated_mz_values_as_default,
            self._zeroth_frame
        )

@jitclass([('accumulation_times', types.float64[:]),
            ('cycle', types.float64[:, :, :, ::1]),
            ('dia_mz_cycle', types.float64[:, ::1]),
            ('dia_precursor_cycle', types.int64[::1]),
            ('frame_max_index', types.int64),
            ('intensity_corrections', types.float64[::1]),
            ('intensity_max_value', types.int64),
            ('intensity_min_value', types.int64),
            ('intensity_values', types.uint16[::1]),
            ('max_accumulation_time', types.float64),
            ('mobility_max_value', types.float64),
            ('mobility_min_value', types.float64),
            ('mobility_values', types.float64[::1]),
            ('mz_values', types.float64[::1]),
            ('precursor_indices', types.int64[::1]),
            ('precursor_max_index', types.int64),
            ('push_indptr', types.int64[::1]),
            ('quad_indptr', types.int64[::1]),
            ('quad_max_mz_value', types.float64),
            ('quad_min_mz_value', types.float64),
            ('quad_min_mz_value', types.float64),
            ('quad_mz_values', types.float64[::1,:]),
            ('raw_quad_indptr', types.int64[::1]),
            ('rt_values', types.float64[::1]),
            ('scan_max_index', types.int64),
            ('tof_indices', types.uint32[::1]),
            ('tof_max_index', types.int64),
            ('use_calibrated_mz_values_as_default', types.int64),
            ('zeroth_frame', types.boolean)
        ])
class TimsTOFJIT(object):
    def __init__(
            self, 
            accumulation_times: types.float64[::1],
            cycle:types.float64[:, :, :, ::1],
            dia_mz_cycle:types.float64[:, ::1],
            dia_precursor_cycle:types.int64[::1],
            frame_max_index: types.int64,
            intensity_corrections: types.float64[::1],
            intensity_max_value: types.int64,
            intensity_min_value: types.int64,
            intensity_values: types.uint16[::1],
            max_accumulation_time: types.float64,
            mobility_max_value: types.float64,
            mobility_min_value: types.float64,
            mobility_values: types.float64[::1],
            mz_values: types.float64[::1],
            precursor_indices: types.int64[::1],
            precursor_max_index: types.int64,
            push_indptr: types.int64[::1],
            quad_indptr: types.int64[::1],
            quad_max_mz_value: types.float64,
            quad_min_mz_value: types.float64,
            quad_mz_values: types.float64[::1,:],
            raw_quad_indptr: types.int64[::1],
            rt_values: types.float64[::1],
            scan_max_index: types.int64,
            tof_indices: types.uint32[::1],
            tof_max_index: types.int64,
            use_calibrated_mz_values_as_default: types.int64,
            zeroth_frame: types.boolean
        ):

        self.accumulation_times = accumulation_times
        self.cycle = cycle
        self.dia_mz_cycle = dia_mz_cycle
        self.dia_precursor_cycle = dia_precursor_cycle
        self.frame_max_index = frame_max_index
        self.intensity_corrections = intensity_corrections
        self.intensity_max_value = intensity_max_value
        self.intensity_min_value = intensity_min_value
        self.intensity_values = intensity_values
        self.max_accumulation_time = max_accumulation_time
        self.mobility_max_value = mobility_max_value
        self.mobility_min_value = mobility_min_value
        self.mobility_values = mobility_values
        self.mz_values = mz_values
        self.precursor_indices = precursor_indices
        self.precursor_max_index = precursor_max_index
        self.push_indptr = push_indptr
        self.quad_indptr = quad_indptr
        self.quad_max_mz_value = quad_max_mz_value
        self.quad_min_mz_value = quad_min_mz_value
        self.quad_mz_values = quad_mz_values
        self.raw_quad_indptr = raw_quad_indptr
        self.rt_values = rt_values
        self.scan_max_index = scan_max_index
        self.tof_indices = tof_indices
        self.tof_max_index = tof_max_index
        self.use_calibrated_mz_values_as_default = use_calibrated_mz_values_as_default
        self.zeroth_frame = zeroth_frame

    def return_frame_indices(
            self,
            rt_values, 
            full_precursor_cycle
        ):
        """convert array of rt values into frame indices, precursor cycle aware, njit compatible
        
        """
        frame_index = np.searchsorted(self.rt_values, rt_values, 'left')

        if not full_precursor_cycle:
            return frame_index
        precursor_cycle_limits = (frame_index+self.zeroth_frame)//self.cycle.shape[1]
        
        # second element is the index of the first whole cycle which should not be used
        #precursor_cycle_limits[1] += 1
        # convert back to frame indices
        frame_limits = precursor_cycle_limits*self.cycle.shape[1]+self.zeroth_frame
        return frame_limits

    def return_scan_indices(
            self,
            mobility_values
        ):
        """convert array of mobility values into scan indices, njit compatible"""
        scan_index = self.scan_max_index - np.searchsorted(
                    self.mobility_values[::-1],
                    mobility_values,"right"
                )
        
        scan_index[1] += 1
        
        return scan_index

    def return_tof_indices(
            self,
            mz_values, 
        ):
        """convert array of mobility values into scan indices, njit compatible"""
        return np.searchsorted(self.mz_values, mz_values, 'left')

    def filter_tof_to_csr(
        self,
        tof_slices: np.ndarray,
        push_indices: np.ndarray,
    ) -> tuple:
        """Get a CSR-matrix with raw indices satisfying push indices and tof slices. 
        In contrast to the alphatims.bruker.filter_tof_to_csr implementation, this function will return all hits.

        Parameters
        ----------
        tof_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted, disjunct and strictly increasing
            (i.e. np.all(np.diff(tof_slices[:, :2].ravel()) >= 0) = True).
        push_indices : np.int64[:]
            The push indices from where to retrieve the TOF slices.

        Returns
        -------
        (np.int64[:], np.int64[:], np.int64[:],)
            An (indptr, values, columns) tuple, where indptr are push indices,
            values raw indices, and columns the tof_slices.
        """

        indptr = [0]
        values = []
        columns = []
        for push_index in push_indices:
            
            start = self.push_indptr[push_index]
            end = self.push_indptr[push_index + 1]
            idx = start
            for i, (tof_start, tof_stop, tof_step) in enumerate(tof_slices):
                # Instead of leaving it at the end of the first tof slice it's reset to the start to allow for 
                # Overlap of tof slices
                start_idx = np.searchsorted(self.tof_indices[idx: end], tof_start)
                idx += start_idx
                tof_value = self.tof_indices[idx]
                while (tof_value < tof_stop) and (idx < end):
                    if tof_value in range(tof_start, tof_stop, tof_step):
                        values.append(idx)
                        columns.append(i)
                        # don't break on first hit
                        #break  # TODO what if multiple hits?
                    idx += 1
                    tof_value = self.tof_indices[idx]
                idx = start + start_idx
            indptr.append(len(values))
        return np.array(indptr), np.array(values), np.array(columns)


    def get_dia_push_indices(
        self,
        frame_slices: np.ndarray,
        scan_slices: np.ndarray,
        quad_slices: np.ndarray,
        precursor_slices: np.ndarray = None,
        dia_mz_cycle = None
    ):
        """Get the push indices and precursor indices for a given frame, scan and quad slice.
        On top of the quad slices, the precursor slices can be used to filter the push indices.

        This method differs from the alphatims.bruker.get_dia_push_indices in the follwoing ways:
        - TimsTOF attributes are used from the jitclass without passing them as arguments
        - The dia_mz_cycle can be specified on top as an argument
        - Both the push indices as well as the precursor indices are returned

        Parameters
        ----------
        frame_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted, disjunct and strictly increasing
            (i.e. np.all(np.diff(frame_slices[:, :2].ravel()) >= 0) = True).

        scan_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted, disjunct and strictly increasing
            (i.e. np.all(np.diff(scan_slices[:, :2].ravel()) >= 0) = True).

        quad_slices : np.float64[:, 2]
            Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
            This array is assumed to be sorted, disjunct and strictly increasing
            (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).
            To select only precursor ions, use np.array([[-1.,-1.]]).
            To select all ions, use np.array([[-np.inf,np.inf]]).

        precursor_slices : np.int64[:, 3]
            Can be used as an additional filter on top of the quad slices.
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted, disjunct and strictly increasing
            (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).

        dia_mz_cycle : np.float64[:, 2], optional
            An array with (upper, lower) mz values of a DIA cycle (per push).
            If None, dia_mz_cycle from TimsTOF object is used. Can be replaced if
            quadrupole calibration should be enabled at some point.

        Returns
        -------

        : np.int64[:]
            The raw push indices that satisfy all the slices.

        : np.int64[:]
            Array with the sake length as the push indices array.
            Each element is the precursor index of the push index.

        """

        if dia_mz_cycle is None:
            dia_mz_cycle = self.dia_mz_cycle

        # if quadrupole calibration should be enabled at some point
        # dia_mz_cycle can be replaced by an updated version with wider windows
        quad_mask = alphatims.bruker.calculate_dia_cycle_mask(
            dia_mz_cycle=dia_mz_cycle,
            quad_slices=quad_slices,
            dia_precursor_cycle=self.dia_precursor_cycle,
            precursor_slices=precursor_slices
        )

        # quad mask is of length frames * 928

        l_dia_mz_cycle = len(dia_mz_cycle)

        push_indices = [] 
        precursor_indices = []

        for frame_start, frame_stop, frame_step in frame_slices:
            for frame_index in range(frame_start, frame_stop, frame_step):
                for scan_start, scan_stop, scan_step in scan_slices:
                    for scan_index in range(scan_start, scan_stop, scan_step):
                        # first push of the frame: frame_index * scan_max_index 
                        # nth push of the frame: frame_index * scan_max_index + n
                        push_index = frame_index * self.scan_max_index + scan_index
                        # subtract a whole frame if the first frame is zero
                        if self.zeroth_frame:
                            cyclic_push_index = push_index - self.scan_max_index
                        else:
                            cyclic_push_index = push_index

                        # gives the scan index in the dia mz cycle
                        scan_in_dia_mz_cycle = cyclic_push_index % l_dia_mz_cycle
                        # make sure the scan is in the quad mask
                        if quad_mask[scan_in_dia_mz_cycle]:
                            # add the precursor index to the list
                            precursor_indices.append(self.dia_precursor_cycle[scan_in_dia_mz_cycle])
                            # add the absolute push index to the list
                            push_indices.append(push_index)

        return np.array(push_indices), np.array(precursor_indices)


    def get_dense(self,
            frame_limits,
            scan_limits,
            tof_limits,
            quadrupole_limits,
            skip_mz,
        ):

        push_indices, absolute_precursor_index  = self.get_dia_push_indices(
            frame_limits,
            scan_limits,
            quadrupole_limits,
        )

        # The precursor cycle is the 3rd dimension of the dense precursor representation
        # The 3rd axis could theoretically have the length of all existing precursor cycles
        # However, we only need to store the precursor cycles that are actually present in the data
        #   
        # indexed_precursor_cycles is returned with the dense object and contains the original, absolute precursor cycles
        # indexed_precursor_cycles_reverse is a lookup table that maps the absolute precursor cycles to the indices in the dense object
        # 
        # e.g. indexed_precursor_cycles = [0 3 4 5]
        #      indexed_precursor_cycles_reverse = [0 0 0 1 2 3]
        # note that non exising precursor cycles are mapped to 0!

        precursor_index = np.unique(absolute_precursor_index)
        n_precursor_indices = len(precursor_index)
        precursor_index_reverse = np.zeros(np.max(precursor_index)+1, dtype=np.int64)
        precursor_index_reverse[precursor_index] = np.arange(len(precursor_index))
        
        push_ptr, raw_indices, tof_slice_ptr = self.filter_tof_to_csr(
            tof_limits, 
            push_indices, 
        )

        # But, not all pushes have ions in the right TOF range
        # So next, we have to get the matching TOF slices
    
        # Get sparse representation of indices
        # push_ptr has P + 1 elements
        #   push_ptr maps all given pushes to the raw indices identified within the tof limits
        #   push_ptr[i:i+1] is the start and stop index of the raw indices for the push
        #   np.diff(push_ptr) is therefore number of raw indices for every push
        #       e.g. push_ptr = [0, 0, 1, 1, 2, 3]
        #       np.diff(push_ptr) = [0, 1, 0, 1, 1]
        #       push_ptr[-1] therefore is the number of raw indices, N
        #
        # raw_indices has N raw indices
        # 
        # tof_slice_ptr has N elements

        push_len = np.diff(push_ptr)

        # get push index for every raw index
        raw_push_indices = np.repeat(push_indices, push_len)

        # get relative precursor cycle for every raw index
        raw_absolute_precursor_index = np.repeat(absolute_precursor_index, push_len)
        raw_relative_precursor_index  = precursor_index_reverse[raw_absolute_precursor_index]

        frame_indices = raw_push_indices // self.scan_max_index
        scan_indices = raw_push_indices % self.scan_max_index
        precursor_cycle_indices = (frame_indices-self.zeroth_frame)//self.cycle.shape[1]

        # cycle values
        precursor_cycle_start = (frame_limits[0,0]-self.zeroth_frame)//self.cycle.shape[1]
        precursor_cycle_stop = (frame_limits[0,1]-self.zeroth_frame)//self.cycle.shape[1]
        precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

        # scan valuesa
        mobility_start = scan_limits[0,0]
        mobility_stop = scan_limits[0,1]
        mobility_len = mobility_stop - mobility_start

        tof_indices = self.tof_indices[raw_indices]
        mz_values = self.mz_values[tof_indices]

        intensities = self.intensity_values[raw_indices]

        # number of channels: intensity, mz
        if skip_mz:
            n_channels = 1
        else:
            n_channels = 2
        
        

        n_tof_slices = len(tof_limits)

        dense_output = np.zeros(
            (
                n_channels, 
                n_tof_slices,
                n_precursor_indices,
                mobility_len,
                precursor_cycle_len
            ), 
            dtype=np.float64
        )

        # create dense intensities
        for i, (tof_slice, intensity, mz) in enumerate(zip(tof_slice_ptr, intensities, mz_values)):
            mobility = scan_indices[i]-mobility_start
            precursor_cycle = precursor_cycle_indices[i]-precursor_cycle_start
            p_slice = raw_relative_precursor_index[i]
            dense_output[0, tof_slice, p_slice ,mobility, precursor_cycle] += intensity

        if not skip_mz:
            # create dense weighted mz
            for i, (tof_slice, intensity, mz) in enumerate(zip(tof_slice_ptr, intensities, mz_values)):
                mobility = scan_indices[i]-mobility_start
                precursor_cycle = precursor_cycle_indices[i]-precursor_cycle_start
                p_slice = raw_relative_precursor_index[i]
                dense_output[1,tof_slice, p_slice, mobility, precursor_cycle] += mz_values[i] * (intensities[i]/dense_output[0,tof_slice, p_slice, mobility, precursor_cycle])

        return dense_output, precursor_index

class TimsTOFDIA(alphatims.bruker.TimsTOF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logging.info('Initializing class njit functions')
        try:
            self.filter_tof_to_csr(
                np.array([[1,2,1]]), 
                np.array([1,2]), 
            )
        except:
            logging.info('initialization of filter_tof_to_csr failed')


    @alphatims.utils.class_njit()
    def return_frame_indices(
            self,
            values, 
            full_precursor_cycle=True
        ):
        """convert array of rt values into frame indices, precursor cycle aware, njit compatible
        
        """
        frame_index = np.searchsorted(self._rt_values, values, 'left')

        if not full_precursor_cycle:
            return frame_index
        precursor_cycle_limits = (frame_index+self._zeroth_frame)//self._cycle.shape[1]
        
        # second element is the index of the first whole cycle which should not be used
        #precursor_cycle_limits[1] += 1
        # convert back to frame indices
        frame_limits = precursor_cycle_limits*self._cycle.shape[1]+self._zeroth_frame
        return frame_limits

    
    @alphatims.utils.class_njit()
    def return_scan_indices(
            self,
            mobility_values, 
        ):
        """convert array of mobility values into scan indices, njit compatible"""
        scan_index = self._scan_max_index - np.searchsorted(
                    self._mobility_values[::-1],
                    mobility_values,"right"
                )
        
        scan_index[1] += 1
        
        return scan_index

    @alphatims.utils.class_njit()
    def return_tof_indices(
            self,
            values, 
        ):
        """convert array of mobility values into scan indices, njit compatible"""
        return np.searchsorted(self._mz_values, values, 'left')

    @alphatims.utils.class_njit()
    def filter_tof_to_csr(
        self,
        tof_slices: np.ndarray,
        push_indices: np.ndarray,
    ) -> tuple:
        """Get a CSR-matrix with raw indices satisfying push indices and tof slices. 
        In contrast to the alphatims.bruker.filter_tof_to_csr implementation, this function will return all hits.

        Parameters
        ----------
        tof_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted, disjunct and strictly increasing
            (i.e. np.all(np.diff(tof_slices[:, :2].ravel()) >= 0) = True).
        push_indices : np.int64[:]
            The push indices from where to retrieve the TOF slices.

        Returns
        -------
        (np.int64[:], np.int64[:], np.int64[:],)
            An (indptr, values, columns) tuple, where indptr are push indices,
            values raw indices, and columns the tof_slices.
        """

        indptr = [0]
        values = []
        columns = []
        for push_index in push_indices:
            
            start = self._push_indptr[push_index]
            end = self._push_indptr[push_index + 1]
            idx = start
            for i, (tof_start, tof_stop, tof_step) in enumerate(tof_slices):
                idx += np.searchsorted(self._tof_indices[idx: end], tof_start)
                tof_value = self._tof_indices[idx]
                while (tof_value < tof_stop) and (idx < end):
                    if tof_value in range(tof_start, tof_stop, tof_step):
                        values.append(idx)
                        columns.append(i)
                        # don't break on first hit
                        #break  # TODO what if multiple hits?
                    idx += 1
                    tof_value = self._tof_indices[idx]
            indptr.append(len(values))
        return np.array(indptr), np.array(values), np.array(columns)

    @alphatims.utils.class_njit()
    def get_dense(self,
            frame_limits,
            scan_limits,
            tof_limits,
            quadrupole_limits
        ):

        # push indices contains the indices for all pushes within the frame, scan limits
        push_indices = alphatims.bruker.get_dia_push_indices(
            frame_limits,
            scan_limits,
            quadrupole_limits,
            self._scan_max_index,
            self._dia_mz_cycle
        )
        
        # csr of indices
        # push_ptr has len of push_indices + 1 

        push_ptr, raw_indices, tof_slice_ptr = self.filter_tof_to_csr(
            tof_limits, 
            push_indices, 
        )


        push_len = np.diff(push_ptr)

        raw_push_indices = np.repeat(push_indices, push_len)

        frame_indices = raw_push_indices // self._scan_max_index
        scan_indices = raw_push_indices % self._scan_max_index
        precursor_cycle_indices = (frame_indices-self._zeroth_frame)//self._cycle.shape[1]

        # cycle values
        precursor_cycle_start = (frame_limits[0,0]-self._zeroth_frame)//self._cycle.shape[1]
        precursor_cycle_stop = (frame_limits[0,1]-self._zeroth_frame)//self._cycle.shape[1]
        precursor_cycle_len = precursor_cycle_stop - precursor_cycle_start

        # scan valuesa
        mobility_start = scan_limits[0,0]
        mobility_stop = scan_limits[0,1]
        mobility_len = mobility_stop - mobility_start

        tof_indices = self._tof_indices[raw_indices]
        mz_values = self._mz_values[tof_indices]

        intensities = self._intensity_values[raw_indices]

        # number of channels: intensity, mz
        num_channels = 2

        num_isotopes = len(tof_limits)

        dense_output = np.zeros(
            (
                num_channels, 
                num_isotopes,
                mobility_len,
                precursor_cycle_len
            ), 
            dtype=np.float64
        )

        for i, (isotope, intensity, mz) in enumerate(zip(tof_slice_ptr, intensities, mz_values)):
            mobility = scan_indices[i]-mobility_start
            precursor_cycle = precursor_cycle_indices[i]-precursor_cycle_start
            dense_output[0,isotope, mobility, precursor_cycle] += intensity

        for i, (isotope, intensity, mz) in enumerate(zip(tof_slice_ptr, intensities, mz_values)):
            mobility = scan_indices[i]-mobility_start
            precursor_cycle = precursor_cycle_indices[i]-precursor_cycle_start
            dense_output[1,isotope, mobility, precursor_cycle] += mz_values[i] * (intensities[i]/dense_output[0,isotope, mobility, precursor_cycle])


        return dense_output
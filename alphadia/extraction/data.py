import alphatims.utils
import alphatims.bruker
import numpy as np
import logging

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
        precursor_cycle_limits = (frame_index-self._zeroth_frame)//self._cycle.shape[1]
        
        # second element is the index of the first whole cycle which should not be used
        precursor_cycle_limits[1] += 1
        # convert back to frame indices
        frame_limits = precursor_cycle_limits*self._cycle.shape[1]+self._zeroth_frame
        return frame_limits

    
    @alphatims.utils.class_njit()
    def return_scan_indices(
            self,
            values, 
        ):
        """convert array of mobility values into scan indices, njit compatible"""
        scan_index = self._scan_max_index - np.searchsorted(
                    self._mobility_values[::-1],
                    values,"right"
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
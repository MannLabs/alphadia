import alphatims.utils
import alphatims.bruker
import numpy as np

class TimsTOFDIA(alphatims.bruker.TimsTOF):

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
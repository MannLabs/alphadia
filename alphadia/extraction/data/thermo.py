import alpharaw.thermo
import numpy as np
import numba as nb
import pandas as pd
import math

from alphadia.extraction import utils

def normed_auto_correlation(x):
    x = x-x.mean()
    result = np.correlate(x, x, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result

def calculate_cycle(spectrum_df):

    # the cycle length is calculated by using the auto correlation of the isolation window m/z values
    x = spectrum_df.isolation_lower_mz.values[:10000] + spectrum_df.isolation_upper_mz.values[:10000]
    corr = normed_auto_correlation(x)
    corr[0] = 0
    cycle_length = np.argmax(corr)

    # check that the cycles really match
    first_cycle = spectrum_df.isolation_lower_mz.values[:cycle_length] + spectrum_df.isolation_upper_mz.values[:cycle_length]
    second_cycle = spectrum_df.isolation_lower_mz.values[cycle_length:2*cycle_length] + spectrum_df.isolation_upper_mz.values[cycle_length:2*cycle_length]
    if not np.allclose(first_cycle, second_cycle):
        raise ValueError('No DIA cycle pattern found in the data.')

    cycle = np.zeros((1,cycle_length,1,2), dtype=np.float64)
    cycle[0,:,0,0] = spectrum_df.isolation_lower_mz.values[:cycle_length]
    cycle[0,:,0,1] = spectrum_df.isolation_upper_mz.values[:cycle_length]

    return cycle

class Thermo(alpharaw.thermo.ThermoRawData):

    def __init__(
        self, 
        path, 
        astral_ms1=False
        ):
        super().__init__()
        self.load_raw(path)
        
        self.astral_ms1 = astral_ms1
        self.filter_spectra()

        self.cycle = calculate_cycle(self.spectrum_df)
        self.rt_values = self.spectrum_df.rt.values * 60
        self.zeroth_frame = 0
        self.precursor_cycle_max_index = len(self.rt_values)//self.cycle.shape[1]
        self.mobility_values = np.array([-1, 1], dtype=np.float64)

        self.max_mz_value = self.spectrum_df.precursor_mz.max()
        self.min_mz_value = self.spectrum_df.precursor_mz.min()
        self.quad_max_mz_value = self.spectrum_df[self.spectrum_df['ms_level'] == 2].isolation_upper_mz.max()
        self.quad_min_mz_value = self.spectrum_df[self.spectrum_df['ms_level'] == 2].isolation_lower_mz.min()
        

    def filter_spectra(self):
        if self.astral_ms1:
            self.spectrum_df = self.spectrum_df[self.spectrum_df['nce'] > 0.1]
            self.spectrum_df.loc[self.spectrum_df['nce'] < 1.1, 'ms_level'] = 1
            self.spectrum_df.loc[self.spectrum_df['nce'] < 1.1, 'precursor_mz'] = -1.0
            self.spectrum_df.loc[self.spectrum_df['nce'] < 1.1, 'isolation_lower_mz'] = -1.0
            self.spectrum_df.loc[self.spectrum_df['nce'] < 1.1, 'isolation_upper_mz'] = -1.0
            self.spectrum_df['spec_idx'] = np.arange(len(self.spectrum_df))
        else:
            self.spectrum_df = self.spectrum_df[(self.spectrum_df['nce'] < 0.1) | (self.spectrum_df['nce'] > 1.1)]
            self.spectrum_df['spec_idx'] = np.arange(len(self.spectrum_df))

    def jitclass(self):
        return ThermoJIT(
            self.cycle,
            self.rt_values,
            self.mobility_values,
            self.zeroth_frame,
            self.max_mz_value,
            self.min_mz_value,
            self.quad_max_mz_value,
            self.quad_min_mz_value,
            self.precursor_cycle_max_index,
        )
    

@nb.experimental.jitclass([
            ('cycle', nb.core.types.float64[:, :, :, ::1]),
            ('rt_values', nb.core.types.float64[::1]),
            ('mobility_values', nb.core.types.float64[::1]),
            ('zeroth_frame', nb.core.types.boolean),
            ('max_mz_value', nb.core.types.float64),
            ('min_mz_value', nb.core.types.float64),
            ('quad_max_mz_value', nb.core.types.float64),
            ('quad_min_mz_value', nb.core.types.float64),
            ('precursor_cycle_max_index', nb.core.types.int64),
        ])

class ThermoJIT(object):
    """Numba compatible transposed TimsTOF data structure."""
    def __init__(
            self, 
            cycle: nb.core.types.float64[:, :, :, ::1],
            rt_values: nb.core.types.float64[::1],
            mobility_values: nb.core.types.float64[::1],
            zeroth_frame: nb.core.types.boolean,
            max_mz_value: nb.core.types.float64,
            min_mz_value: nb.core.types.float64,
            quad_max_mz_value: nb.core.types.float64,
            quad_min_mz_value: nb.core.types.float64,
            precursor_cycle_max_index: nb.core.types.int64,
        ):

        """Numba compatible transposed TimsTOF data structure.

        Parameters
        ----------

        accumulation_times : np.ndarray, shape = (n_frames,), dtype = float64
            array of accumulation times
        
        """
        
        self.cycle = cycle
        self.rt_values = rt_values
        self.mobility_values = mobility_values
        self.zeroth_frame = zeroth_frame
        self.max_mz_value = max_mz_value
        self.min_mz_value = min_mz_value
        self.quad_max_mz_value = quad_max_mz_value
        self.quad_min_mz_value = quad_min_mz_value
        self.precursor_cycle_max_index = precursor_cycle_max_index

    def get_frame_indices(
        self,
        rt_values : np.array,
        optimize_size : int = 16
    ):
        """
        
        Convert an interval of two rt values to a frame index interval.
        The length of the interval is rounded up so that a multiple of 16 cycles are included.

        Parameters
        ----------
        rt_values : np.ndarray, shape = (2,), dtype = float32
            array of rt values

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the precursor cycle to a multiple of 16

        Returns
        -------
        np.ndarray, shape = (2,), dtype = int64
            array of frame indices
        
        """

        if rt_values.shape != (2,):
            raise ValueError('rt_values must be a numpy array of shape (2,)')
        
        frame_index = np.searchsorted(self.rt_values, rt_values, 'left')

        precursor_cycle_limits = (frame_index+self.zeroth_frame)//self.cycle.shape[1]
        precursor_cycle_len = precursor_cycle_limits[1]-precursor_cycle_limits[0]

        # round up to the next multiple of 16
        optimal_len = int(optimize_size * math.ceil( precursor_cycle_len / optimize_size))

        # by default, we extend the precursor cycle to the right
        optimal_cycle_limits = np.array([precursor_cycle_limits[0], precursor_cycle_limits[0]+optimal_len], dtype=np.int64)

        # if the cycle is too long, we extend it to the left
        if optimal_cycle_limits[1] > self.precursor_cycle_max_index:
            optimal_cycle_limits[1] = self.precursor_cycle_max_index
            optimal_cycle_limits[0] = self.precursor_cycle_max_index-optimal_len

            if optimal_cycle_limits[0] < 0:
                optimal_cycle_limits[0] = 0 if self.precursor_cycle_max_index % 2 == 0 else 1

        # second element is the index of the first whole cycle which should not be used
        #precursor_cycle_limits[1] += 1
        # convert back to frame indices
        frame_limits = optimal_cycle_limits*self.cycle.shape[1]+self.zeroth_frame
        return utils.make_slice_1d(
            frame_limits
        )
    
    def get_frame_indices_tolerance(
            self,
            rt : float,
            tolerance : float,
            optimize_size : int = 16
        ):
        """
        Determine the frame indices for a given retention time and tolerance. 
        The frame indices will make sure to include full precursor cycles and will be optimized for fft.

        Parameters
        ----------
        rt : float
            retention time in seconds

        tolerance : float
            tolerance in seconds

        optimize_size : int, default = 16
            To optimize for fft efficiency, we want to extend the precursor cycle to a multiple of 16

        Returns
        -------
        np.ndarray, shape = (1, 3,), dtype = int64
            array which contains a slice object for the frame indices [[start, stop step]]

        """

        rt_limits = np.array([
            rt-tolerance, 
            rt+tolerance
        ], dtype=np.float32)
    
        return self.get_frame_indices(
            rt_limits,
            optimize_size = optimize_size
        )
    
    def get_scan_indices(
            self,
            mobility_values : np.array,
            optimize_size : int = 16
        ):

        return np.array([[0,2,1]], dtype=np.int64)
    
    def get_scan_indices_tolerance(
            self, 
            mobility,
            tolerance,
            optimize_size=16

        ):
        return np.array([[0,2,1]], dtype=np.int64)
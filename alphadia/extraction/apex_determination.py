# internal imports
from .utils import indices_to_slices

# alpha imports
import alphatims.bruker
import alphatims.utils
import alphatims.utils
import alphatims.tempmmap

# external imports
import logging

from typing import Tuple, Union, List

import pandas as pd
import numpy as np

def assemble_fragment_df(fragment_index_tuple: Tuple[int,int], lib):

    """ Collects all fragments encoded for an precursor and places them in a dataframe
    
    """
    # FUTURE, reduce minimum fragment intensity treshold in library generation

    # get fragments from df based on precursor fragemnt slicing
    fragment_slice = slice(*fragment_index_tuple)

    fragment_mz_slice = lib.fragment_mz_df.iloc[fragment_slice]
    fragment_intensity_slice = lib.fragment_intensity_df.iloc[fragment_slice]

    num_fragments = fragment_index_tuple[1]-fragment_index_tuple[0]
    # FUTURE, same for whole lib
    num_cols = len(fragment_mz_slice.columns)
    start_index = 1

    #fragment series number
    fragment_series = np.tile(np.arange(start_index,start_index + num_fragments),num_cols)

    #fragment mzs
    fragment_mz_flat = fragment_mz_slice.values.flatten(order='F')

    #fragment intensity
    fragment_intensity_flat = fragment_intensity_slice.values.flatten(order='F')

    # fragment series and charge
    fragment_ion_type = []
    for column in fragment_mz_slice.columns:
        fragment_ion_type += [column]*num_fragments

    # assemble dataframe
    df = pd.DataFrame({'fragment_index': fragment_series,
                        'fragment_type': fragment_ion_type,
                        'fragment_mz': fragment_mz_flat,
                        'fragment_intensity': fragment_intensity_flat,
                        })

    df = df[df['fragment_intensity'] > 0]

    return df

def dense(dia_data: alphatims.bruker.TimsTOF, 
        frame_index_tuple: Tuple[int], 
        scan_index_tuple: Tuple[int], 
        quad_tuple: Tuple[Union[float, int]],
        mz_tuple_list: List[Tuple[float]], 
        background: str = 'ones'):
    """Retrive a list of mass slices with common frame index, scan index and quadrupole values

    Returns
    -------
    numpy.ndarray: Numpy array of shape (N, S, F) with N: number of mass slices, S: number of scans, F: number of frames

    """

    cycle_length = dia_data.cycle.shape[1]

    number_of_ions = len(mz_tuple_list)
    scan_size = scan_index_tuple[1]-scan_index_tuple[0] + 1
    frame_size = (frame_index_tuple[1]-frame_index_tuple[0])//cycle_length + 1

    dense_mat = np.ones([number_of_ions, scan_size, frame_size])

    for i, mz_tuple in enumerate(mz_tuple_list):
        
        data_df = dia_data[slice(*frame_index_tuple),slice(*scan_index_tuple),quad_tuple,slice(*mz_tuple)]
      
        cycle_index = data_df['frame_indices']//cycle_length
        cycle_index_local = cycle_index - frame_index_tuple[0] // cycle_length

        scan_index = data_df['scan_indices'].values
        scan_index_local = scan_index-scan_index_tuple[0]

        intensity_arr = data_df['intensity_values'].values

        for j, (cycle, scan, intensity) in enumerate(zip(cycle_index_local, scan_index_local, intensity_arr)):
            dense_mat[i,scan,cycle]+=intensity

    return dense_mat
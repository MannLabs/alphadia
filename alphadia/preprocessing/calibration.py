"""Calibrating quadrupole settings"""


import logging

import numpy as np

import alphatims.utils

class QuadCalibrator:

    def __init__(
        self,
        dia_data,
    ):
        self.dia_data = dia_data

    def calculate_calibrated_cycle(self,):
        logging.info("Calibrating quadrupole")
        cycle = np.copy(self.dia_data.cycle).reshape(-1, 2)
        summed_intensity = np.zeros(len(cycle))
        estimate_isolation_window(
            range(len(cycle)),
            self.dia_data.intensity_values,
            self.dia_data.tof_indices,
            self.dia_data.push_indptr,
            self.dia_data.zeroth_frame,
            len(cycle),
            self.dia_data.tof_max_index,
            self.dia_data.scan_max_index,
            self.dia_data.mz_values,
            cycle,
            summed_intensity,
        )
        cycle = cycle.reshape(self.dia_data.cycle.shape)
        summed_intensity = summed_intensity.reshape(self.dia_data.cycle.shape[:-1])
        self.summed_intensity = summed_intensity
        self.cycle = cycle
        self.predict_cycle()

    def predict_cycle(self):
        import sklearn.linear_model
        predicted_cycle = np.copy(self.cycle)
        frame_length = self.cycle.shape[2]
        for subcycle_i, subcycle in enumerate(self.cycle):
            for frame_i, frame in enumerate(subcycle):
                model_lower = sklearn.linear_model.LinearRegression().fit(
                    np.arange(frame_length).reshape(-1, 1),
                    frame[:, 0].reshape(-1, 1),
                    self.summed_intensity[subcycle_i, frame_i] + 1
                )
                model_upper = sklearn.linear_model.LinearRegression().fit(
                    np.arange(frame_length).reshape(-1, 1),
                    frame[:, 1].reshape(-1, 1),
                    self.summed_intensity[subcycle_i, frame_i] + 1
                )
                predicted_frame_lower = model_lower.predict(
                    np.arange(frame_length).reshape(-1, 1)
                )
                predicted_frame_upper = model_upper.predict(
                    np.arange(frame_length).reshape(-1, 1)
                )
                predicted_cycle[subcycle_i, frame_i, :, 0] = predicted_frame_lower.ravel()
                predicted_cycle[subcycle_i, frame_i, :, 1] = predicted_frame_upper.ravel()
        self.predicted_cycle = predicted_cycle


@alphatims.utils.pjit
def estimate_isolation_window(
    cyclic_push_index,
    intensity_values,
    tof_indices,
    push_indptr,
    zeroth_frame,
    cycle_length,
    tof_max_index,
    scan_max_index,
    mz_values,
    cycle,
    summed_intensity
):
    if cycle[cyclic_push_index, 0] <= 0:
        return
    intensity_buffer = merge_cyclic_pushes(
        cyclic_push_index,
        intensity_values,
        tof_indices,
        push_indptr,
        zeroth_frame,
        cycle_length,
        tof_max_index,
        scan_max_index
    )
    vals = np.cumsum(intensity_buffer)
    total_intensity = vals[-1]
    vals /= total_intensity
    low_tof = np.searchsorted(vals, 0.25, "left")
    high_tof = np.searchsorted(vals, 0.75, "right")
    low_mz = mz_values[low_tof]
    high_mz = mz_values[high_tof]
    mz_width = high_mz - low_mz
    mz_mean = (high_mz + low_mz) / 2
    cycle[cyclic_push_index] = (mz_mean - mz_width, mz_mean + mz_width)
    summed_intensity[cyclic_push_index] = total_intensity


@alphatims.utils.njit
def merge_cyclic_pushes(
    cyclic_push_index,
    intensity_values,
    tof_indices,
    push_indptr,
    zeroth_frame,
    cycle_length,
    tof_max_index,
    scan_max_index,
):
    offset = scan_max_index * zeroth_frame + cyclic_push_index
    intensity_buffer = np.zeros(tof_max_index)
    for push_index in range(offset, len(push_indptr) - 1, cycle_length):
        start = push_indptr[push_index]
        end = push_indptr[push_index + 1]
        for index in range(start, end):
            tof = tof_indices[index]
            intensity = intensity_values[index]
            intensity_buffer[tof] += intensity
    return intensity_buffer

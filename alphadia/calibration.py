"""Calibrate quad"""

import alphatims.bruker
import alphatims.utils
import numpy as np
import pandas as pd
import alphatims.plotting


@alphatims.utils.njit(nogil=True, cache=False)
def merge_cyclic_pushes(
    cyclic_push_index,
    intensity_values,
    tof_indices,
    push_indptr,
    zeroth_frame,
    cycle_length,
    tof_max_index,
    scan_max_index,
    return_sparse=False,
):
    offset = scan_max_index * zeroth_frame + cyclic_push_index
    intensity_buffer = np.zeros(tof_max_index)
    tofs = []
    for push_index in range(offset, len(push_indptr) - 1, cycle_length):
        start = push_indptr[push_index]
        end = push_indptr[push_index + 1]
        for index in range(start, end):
            tof = tof_indices[index]
            intensity = intensity_values[index]
            if intensity_buffer[tof] == 0:
                tofs.append(tof)
            intensity_buffer[tof] += intensity
    tofs = np.array(tofs, dtype=tof_indices.dtype)
    if return_sparse:
        tofs = np.sort(tofs)
        intensity_buffer = intensity_buffer[tofs]
    return tofs, intensity_buffer


def guesstimate_quad_settings(
    dia_data,
    smooth_window=100,
    gaussian_blur=5,
    percentile=50,
    regresion_mz_lower_cutoff=400,
    regresion_mz_upper_cutoff=1000,
):
    dia_mz_cycle = np.empty_like(dia_data.dia_mz_cycle)
    weights = np.zeros(len(dia_mz_cycle))
    for cyclic_push_index, (low_quad, high_quad) in alphatims.utils.progress_callback(
        enumerate(dia_data.dia_mz_cycle),
        total=len(dia_data.dia_mz_cycle)
    ):
        if (low_quad == -1) and (high_quad == -1):
            dia_mz_cycle[cyclic_push_index] = (low_quad, high_quad)
            continue
        tofs, intensity_buffer = merge_cyclic_pushes(
            cyclic_push_index=cyclic_push_index,
            intensity_values=dia_data.intensity_values,
            tof_indices=dia_data.tof_indices,
            push_indptr=dia_data.push_indptr,
            zeroth_frame=dia_data.zeroth_frame,
            cycle_length=len(dia_data.dia_mz_cycle),
            tof_max_index=dia_data.tof_max_index,
            scan_max_index=dia_data.scan_max_index,
            return_sparse=True,
        )
        if len(tofs) > 0:
            cum_int = np.cumsum(intensity_buffer)
            low_threshold = cum_int[-1] * percentile / 100 / 2
            high_threshold = cum_int[-1] * (1 - (percentile / 100 / 2))
            low_index = np.searchsorted(cum_int, low_threshold)
            high_index = np.searchsorted(cum_int, high_threshold, "right")
            low_quad_estimate = dia_data.mz_values[tofs[low_index]]
            high_quad_estimate = dia_data.mz_values[tofs[high_index]]
        else:
            low_quad_estimate, high_quad_estimate = -1, -1
        dia_mz_cycle[cyclic_push_index] = (
            low_quad_estimate,
            high_quad_estimate
        )
        weights[cyclic_push_index] = np.sum(intensity_buffer)
    predicted_dia_mz_cycle = predict_dia_mz_cycle(
        dia_mz_cycle,
        dia_data,
        weights,
    )
    return dia_mz_cycle, predicted_dia_mz_cycle



def predict_dia_mz_cycle(
    dia_mz_cycle,
    dia_data,
    weights,
):
    import sklearn.linear_model
    df = pd.DataFrame(
        {
            "detected_lower": dia_mz_cycle[:, 0],
            "detected_upper": dia_mz_cycle[:, 1],
            "frame": np.arange(len(dia_mz_cycle)) // dia_data.scan_max_index,
            "scan": np.arange(len(dia_mz_cycle)) % dia_data.scan_max_index,
            "weights": weights,
        }
    )
    frame_reg_lower = {}
    frame_reg_upper = {}
    model = sklearn.linear_model.HuberRegressor
    for frame in np.unique(df.frame):
        if np.all(dia_data.dia_mz_cycle[df.frame == frame] == -1):
            continue
        selection = df[df.frame == frame]
        frame_reg_lower[frame] = model().fit(
            selection.scan.values.reshape(-1, 1),
            selection.detected_lower.values.reshape(-1, 1),
            selection.weights.values,
        )
        frame_reg_upper[frame] = model().fit(
            selection.scan.values.reshape(-1, 1),
            selection.detected_upper.values.reshape(-1, 1),
            selection.weights.values,
        )
    predicted_upper = []
    predicted_lower = []
    for index, frame in enumerate(df.frame.values):
        if frame not in frame_reg_upper:
            predicted_upper.append(-1)
            predicted_lower.append(-1)
            continue
        predicted_lower_ = frame_reg_lower[frame].predict(
            df.scan.values[index: index + 1].reshape(-1, 1)
        )
        predicted_upper_ = frame_reg_upper[frame].predict(
            df.scan.values[index: index + 1].reshape(-1, 1)
        )
        predicted_lower.append(predicted_lower_[0])
        predicted_upper.append(predicted_upper_[0])
    predicted_dia_mz_cycle = np.vstack(
        [predicted_lower, predicted_upper]
    ).T
    return predicted_dia_mz_cycle

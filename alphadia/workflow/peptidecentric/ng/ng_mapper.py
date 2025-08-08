"""Conversion of AlphaDIA to NG data structure and back.

TODO: This module is a temporary solution, the mapping should be moved to the NG module.
"""

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_ng import DIADataNextGen as DiaDataNG
from alphadia_ng import SpecLibFlat as SpecLibFlatNG

from alphadia.raw_data import DiaData


def dia_data_to_ng(dia_data: DiaData) -> "DiaDataNG":  # noqa: F821
    """Convert DIA data from classic to ng format."""

    spectrum_df = dia_data.spectrum_df
    peak_df = dia_data.peak_df

    cycle_len = dia_data.cycle.shape[1]
    delta_scan_idx = np.tile(
        np.arange(cycle_len), int(len(dia_data.spectrum_df) / cycle_len + 1)
    )
    cycle_idx = np.repeat(
        np.arange(int(len(dia_data.spectrum_df) / cycle_len + 1)), cycle_len
    )

    dia_data.spectrum_df["delta_scan_idx"] = delta_scan_idx[: len(dia_data.spectrum_df)]
    dia_data.spectrum_df["cycle_idx"] = cycle_idx[: len(dia_data.spectrum_df)]

    return DiaDataNG.from_arrays(
        spectrum_df["delta_scan_idx"].values,
        spectrum_df["isolation_lower_mz"].values.astype(np.float32),
        spectrum_df["isolation_upper_mz"].values.astype(np.float32),
        spectrum_df["peak_start_idx"].values,
        spectrum_df["peak_stop_idx"].values,
        spectrum_df["cycle_idx"].values,
        spectrum_df["rt"].values.astype(np.float32) * 60,  # TODO check factor
        peak_df["mz"].values.astype(np.float32),
        peak_df["intensity"].values.astype(np.float32),
    )


def speclib_to_ng(
    speclib: SpecLibFlat,
    *,
    rt_column: str,
    precursor_mz_column: str,
    fragment_mz_column: str,
    mobility_column: str,
) -> "SpecLibFlatNG":  # noqa: F821
    """Convert speclib from classic to ng format."""

    precursor_df = speclib.precursor_df
    fragment_df = speclib.fragment_df
    speclib_ng = SpecLibFlatNG.from_arrays(
        precursor_df["precursor_idx"].values.astype(np.uint64),
        precursor_df[precursor_mz_column].values.astype(np.float32),  # 'precursor_mz'
        precursor_df[rt_column].values.astype(np.float32),  # rt_pred
        precursor_df["nAA"].values.astype(np.uint8),  # added in e5f3e32d
        precursor_df["flat_frag_start_idx"].values.astype(np.uint64),
        precursor_df["flat_frag_stop_idx"].values.astype(np.uint64),
        fragment_df[fragment_mz_column].values.astype(np.float32),  # mz
        fragment_df["intensity"].values.astype(np.float32),
        # added in 802c323
        fragment_df["cardinality"].values.astype(np.uint8),
        fragment_df["charge"].values.astype(np.uint8),
        fragment_df["loss_type"].values.astype(np.uint8),
        fragment_df["number"].values.astype(np.uint8),
        fragment_df["position"].values.astype(np.uint8),
        fragment_df["type"].values.astype(np.uint8),
    )

    return speclib_ng


def parse_candidates(
    dia_data: DiaData, candidates, precursor_df: pd.DataFrame
) -> pd.DataFrame:
    """Parse candidates from NG to classic format."""
    result = candidates.to_arrays()

    precursor_idx = result[0]
    rank = result[1]
    score = result[2]
    scan_center = result[3]
    scan_start = result[4]
    scan_stop = result[5]
    frame_center = result[6]
    frame_start = result[7]
    frame_stop = result[8]

    candidates_df = pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "rank": rank,
            "score": score,
            "scan_center": scan_center,
            "scan_start": scan_start,
            "scan_stop": scan_stop,
            "frame_center": frame_center,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )

    candidates_df = candidates_df.merge(
        precursor_df[["precursor_idx", "elution_group_idx", "decoy"]],
        on="precursor_idx",
        how="left",
    )

    cycle_len = dia_data.cycle.shape[1]
    candidates_df["frame_start"] = candidates_df["frame_start"] * cycle_len
    candidates_df["frame_stop"] = candidates_df["frame_stop"] * cycle_len
    candidates_df["frame_center"] = candidates_df["frame_center"] * cycle_len

    candidates_df["scan_start"] = 0
    candidates_df["scan_stop"] = 1
    candidates_df["scan_center"] = 0

    return candidates_df

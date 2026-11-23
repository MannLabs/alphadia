"""Conversion between alphaDIA and alphadia-search-rs data structures."""

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import (
    CandidateCollection,
    CandidateFeatureCollection,
    FeatureSet,
    set_num_threads,
)
from alphadia_search_rs import (
    DIAData as DiaDataRust,
)
from alphadia_search_rs import SpecLibFlat as SpecLibFlatRust

from alphadia.constants.keys import CalibCols
from alphadia.raw_data import DiaData


def set_thread_count(thread_count: int) -> None:
    """Set the number of threads used by alphadia-search-rs."""
    set_num_threads(thread_count)


def dia_data_to_rust(dia_data: DiaData) -> "DiaDataRust":  # noqa: F821
    """Convert DIA data to the alphadia-search-rs format."""

    spectrum_df = dia_data.spectrum_df
    peak_df = dia_data.peak_df

    cycle_len = dia_data.cycle.shape[1]
    spectrum_df_len = len(dia_data.spectrum_df)

    delta_scan_idx = np.tile(
        np.arange(cycle_len), int(spectrum_df_len / cycle_len + 1)
    )[:spectrum_df_len]
    cycle_idx = np.repeat(np.arange(int(spectrum_df_len / cycle_len + 1)), cycle_len)[
        :spectrum_df_len
    ]

    return DiaDataRust.from_arrays(
        delta_scan_idx.astype(np.int64),
        spectrum_df["isolation_lower_mz"].values.astype(np.float32),
        spectrum_df["isolation_upper_mz"].values.astype(np.float32),
        spectrum_df["peak_start_idx"].values.astype(np.int64),
        spectrum_df["peak_stop_idx"].values.astype(np.int64),
        cycle_idx.astype(np.int64),
        spectrum_df["rt"].values.astype(np.float32) * 60,
        peak_df["mz"].values.astype(np.float32),
        peak_df["intensity"].values.astype(np.float32),
        dia_data.cycle.astype(np.float32),
    )


def speclib_to_rust(
    speclib: SpecLibFlat,
    *,
    rt_column: str,
    precursor_mz_column: str,
    fragment_mz_column: str,
) -> "SpecLibFlatRust":  # noqa: F821
    """Convert a spectral library to the alphadia-search-rs format."""

    precursor_df = speclib.precursor_df
    fragment_df = speclib.fragment_df

    return SpecLibFlatRust.from_arrays(
        precursor_df["precursor_idx"].values.astype(np.uint64),
        precursor_df["mz_library"].values.astype(np.float32),
        precursor_df[precursor_mz_column].values.astype(np.float32),
        precursor_df["rt_library"].values.astype(np.float32),
        precursor_df[rt_column].values.astype(np.float32),
        precursor_df["nAA"].values.astype(np.uint8),
        precursor_df["flat_frag_start_idx"].values.astype(np.uint64),
        precursor_df["flat_frag_stop_idx"].values.astype(np.uint64),
        fragment_df["mz_library"].values.astype(np.float32),
        fragment_df[fragment_mz_column].values.astype(np.float32),
        fragment_df["intensity"].values.astype(np.float32),
        fragment_df["cardinality"].values.astype(np.uint8),
        fragment_df["charge"].values.astype(np.uint8),
        fragment_df["loss_type"].values.astype(np.uint8),
        fragment_df["number"].values.astype(np.uint8),
        fragment_df["position"].values.astype(np.uint8),
        fragment_df["type"].values.astype(np.uint8),
    )


def get_feature_names(feature_set: FeatureSet) -> list[str]:
    """Get the names of the features alphadia-search-rs computes with `feature_set`."""
    return list(CandidateFeatureCollection.get_feature_names(feature_set))


# The final-round classifier leaves fwhm_rt out: permuting it in offline replays of the final round moved the real
# identifications by less than 1 %. It is still scored, because the precursor output reports it as the peak width.
FINAL_ROUND_EXCLUDED_FEATURES = ("fwhm_rt",)


def parse_candidates(
    candidates: CandidateCollection,
    spectral_library: SpecLibFlat,
    dia_data: DiaDataRust,
) -> pd.DataFrame:
    """Parse alphadia-search-rs candidates into a DataFrame."""

    cycle_len = dia_data.cycle.shape[1]

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
        spectral_library.precursor_df[["precursor_idx", "elution_group_idx", "decoy"]],
        on="precursor_idx",
        how="left",
    )

    candidates_df["frame_start"] = candidates_df["frame_start"] * cycle_len
    candidates_df["frame_stop"] = candidates_df["frame_stop"] * cycle_len
    candidates_df["frame_center"] = candidates_df["frame_center"] * cycle_len

    candidates_df["scan_start"] = 0
    candidates_df["scan_stop"] = 1
    candidates_df["scan_center"] = 0

    return candidates_df


def candidates_to_rust(
    candidates_df: pd.DataFrame, dia_data: DiaDataRust
) -> CandidateCollection:
    """Convert a candidates DataFrame to the alphadia-search-rs format."""

    cycle_len = dia_data.cycle.shape[1]

    candidates = CandidateCollection.from_arrays(
        candidates_df["precursor_idx"].values.astype(np.uint64),
        candidates_df["rank"].values.astype(np.uint64),
        candidates_df["score"].values.astype(np.float32),
        candidates_df["scan_center"].values.astype(np.uint64),
        candidates_df["scan_start"].values.astype(np.uint64),
        candidates_df["scan_stop"].values.astype(np.uint64),
        candidates_df["frame_center"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_start"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_stop"].values.astype(np.uint64) // cycle_len,
    )
    return candidates


def to_features_df(
    candidate_features: CandidateFeatureCollection, spectral_library: SpecLibFlat
) -> pd.DataFrame:
    """Convert alphadia-search-rs candidate features into a DataFrame."""

    features_dict = candidate_features.to_dict_arrays()

    features_df = pd.DataFrame(features_dict)

    features_df = features_df.merge(
        spectral_library.precursor_df[
            [
                "precursor_idx",
                "decoy",
                "elution_group_idx",
                "channel",
                "proteins",
                CalibCols.MZ_LIBRARY,
            ]
        ],
        on="precursor_idx",
        how="left",
    )
    # As in the quantified precursors: fragment competition assigns each PSM to its
    # isolation window by this m/z, so scored and quantified PSMs must use the same one.
    # TODO change logic together with RustExtractionHandler.add_columns_from_library:
    # https://github.com/MannLabs/alphadia/blob/d12dd6b918f450e6e55447e1dd40335aedd31d6b/alphadia/workflow/peptidecentric/extraction_handler.py#L669-L671
    features_df[CalibCols.MZ_OBSERVED] = features_df[CalibCols.MZ_LIBRARY]

    return features_df


def parse_quantification(
    quantified_speclib: "SpecLibFlatQuantified",  # noqa: F821
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the alphadia-search-rs quantified spectral library into precursor and fragments DataFrames."""

    precursor_dict, fragment_dict = quantified_speclib.to_dict_arrays()

    precursor_df = pd.DataFrame(precursor_dict).rename(
        columns={"idx": "precursor_idx"}
    )  # TODO: remove when #96 is merged

    fragments_df = pd.DataFrame(fragment_dict).rename(
        columns={
            "correlation_observed": "correlation",
            "mass_error_observed": "mass_error",
        }
    )

    return precursor_df, fragments_df

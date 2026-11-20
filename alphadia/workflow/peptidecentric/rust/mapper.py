"""Conversion between alphaDIA and alphadia-search-rs data structures."""

from collections.abc import Collection

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

from alphadia.calibration.ms1_calibration import Ms1Calibration
from alphadia.constants.keys import CalibCols
from alphadia.raw_data import DiaData

# library isotope abundances (alphabase IsotopeGenerator) scored against the MS1 isotope XICs
MS1_ISOTOPE_COLUMNS = ["i_0", "i_1", "i_2", "i_3"]


def set_thread_count(thread_count: int) -> None:
    """Set the number of threads used by alphadia-search-rs."""
    set_num_threads(thread_count)


def dia_data_to_rust(dia_data: DiaData) -> "DiaDataRust":  # noqa: F821
    """Convert DIA data to the alphadia-search-rs format.

    For MS1 scans aligned to the MS2 cycle (`ms1_alignment.py`), the acquisition time of each spectrum within the cycle
    (`cycle_positions`) is passed on, so that the MS1 features that score the MS1 scans at their own times place them
    there.
    """

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

    dia_data_rust = DiaDataRust.from_arrays(
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
    if dia_data.cycle_positions is not None:
        dia_data_rust.set_cycle_positions(
            np.asarray(dia_data.cycle_positions, np.float32).tolist()
        )
    return dia_data_rust


def speclib_to_rust(
    speclib: SpecLibFlat,
    *,
    rt_column: str,
    precursor_mz_column: str,
    fragment_mz_column: str,
    mobility_column: str | None = None,
) -> "SpecLibFlatRust":  # noqa: F821
    """Convert a spectral library to the alphadia-search-rs format.

    With `mobility_column`, each precursor's XICs are restricted to the scans within the mobility tolerance of that
    column's 1/K0 (data with an ion mobility dimension).
    """

    precursor_df = speclib.precursor_df
    fragment_df = speclib.fragment_df

    speclib_rust = SpecLibFlatRust.from_arrays(
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
    if mobility_column is not None:
        speclib_rust.set_precursor_mobility(
            precursor_df["precursor_idx"].values.astype(np.uint64),
            precursor_df[mobility_column].values.astype(np.float32),
        )
    return speclib_rust


def set_ms1_isotopes(
    speclib_rust: "SpecLibFlatRust",  # noqa: F821
    precursor_df: pd.DataFrame,
    precursor_mz_column: str,
    rt_column: str,
    calibration: Ms1Calibration,
) -> None:
    """Give the rust library the MS1 isotopes of every precursor at its own m/z, shifted by the calibrated MS1 error.

    Decoys too are scored at their own m/z and envelope: scoring them at their target's would let a classifier
    recognise them. The expected error is evaluated at the precursor's m/z and RT. The isotopes are the leading
    `MS1_ISOTOPE_COLUMNS` the library has.
    """
    isotope_columns = []
    for column in MS1_ISOTOPE_COLUMNS:
        if column not in precursor_df.columns:
            break
        isotope_columns.append(column)
    if not isotope_columns:
        raise ValueError(
            f"search.ms1_features needs the library isotope abundances {MS1_ISOTOPE_COLUMNS} in the precursor table"
        )
    mz = precursor_df[precursor_mz_column].to_numpy(np.float64)
    rt = precursor_df[rt_column].to_numpy(np.float64)
    speclib_rust.set_isotopes(
        precursor_df["precursor_idx"].to_numpy().astype(np.uint64),
        precursor_df["charge"].to_numpy().astype(np.uint8),
        (mz * (1 + calibration.offset_ppm(mz, rt) * 1e-6)).astype(np.float32),
        np.ascontiguousarray(
            precursor_df[isotope_columns].to_numpy(), dtype=np.float32
        ),
    )


def get_feature_names(feature_set: FeatureSet) -> list[str]:
    """Get the names of the features alphadia-search-rs computes with `feature_set`."""
    return list(CandidateFeatureCollection.get_feature_names(feature_set))


# The final-round classifier leaves fwhm_rt out: permuting it in offline replays of the final round moved the real
# identifications by less than 1 %. It is still scored, because the precursor output reports it as the peak width.
FINAL_ROUND_EXCLUDED_FEATURES = ("fwhm_rt",)

# The optional features by the `search.<flag>` that adds them to the classifier. They are in FeatureSet.Full only,
# so they only exist in the final round and never enter the calibration fits.
OPTIONAL_FEATURE_GROUPS = {
    # |fitted - library| of the unit-sum fragment intensities, each fragment XIC fitted to the median elution profile
    "fitted_spread_features": (
        "fit_intensity_abs_diff_mean",
        "fit_intensity_abs_diff_std",
        "fit_intensity_abs_diff_max",
        "fit_y_intensity_abs_diff_std",
    ),
    # the candidate's share of its isolation window's MS2 signal at the apex, raw and against its neighbours
    "window_signal_share_features": (
        "apex_window_signal_frac",
        "apex_window_signal_frac_rank",
        "apex_window_signal_frac_zscore",
    ),
    # the MS1 isotope XICs of the precursor against the fragment elution profile and the library isotope abundances;
    # scored only if the flag is set and the raw file and the library allow it (RustExtractionHandler)
    "ms1_features": (
        "ms1_signal_at_apex",
        "ms1_isotope_correlation",
        "ms1_isotope_cosine",
        "ms1_m_minus_1_frac",
        "ms1_log_intensity",
        "ms1_iso_m1_err",
        "ms1_m0_accurate_share",
        "ms1_m1_accurate_share",
        "ms1_fine_shape",
        "ms1_fine_iso_corr",
    ),
    # the RT x mobility peak of the fragments (ion mobility data only): each fragment's patch of apex cycles x scans
    # against the consensus of the others, mobilogram apex agreement, signal share inside the joint peak, peak width
    "mobility_profile_features": (
        "mob2d_corr_mean",
        "mob2d_coherent_share",
        "mob2d_apex_sd",
        "mob2d_peak_share",
        "mob2d_fwhm",
    ),
    # fragment detection at the apex scan, for one-scan peaks at low input; scored only if the flag is set
    "detection_features": (
        "shadow_lib_share",
        "det_faint_hits",
        "det_faint_miss",
    ),
}


# The mobility block, scored in every round; its values are missing on data without an ion mobility dimension, so its
# columns are dropped there (`to_features_df`).
MOBILITY_FEATURES = (
    "mobility_observed",
    "mobility_error",
    "mobility_fragment_sd",
    "mobility_peak_sd",
    "mobility_outer_frac",
)

# Scored, but not classifier features: the observed mobility is an absolute position and feeds the mobility
# calibration only.
NON_CLASSIFIER_FEATURES = ("mobility_observed",)

# A Gaussian peak's full width at half maximum per standard deviation
FWHM_PER_SD = 2.3548


def get_enabled_feature_groups(search_config: dict) -> list[str]:
    """Get the optional feature groups whose `search.<group>` flag is set."""
    return [
        group for group in OPTIONAL_FEATURE_GROUPS if search_config.get(group, False)
    ]


def get_feature_group_columns(feature_groups: Collection[str]) -> list[str]:
    """Get the feature columns of the optional `feature_groups`."""
    unknown_groups = set(feature_groups) - set(OPTIONAL_FEATURE_GROUPS)
    if unknown_groups:
        raise ValueError(f"Unknown optional feature groups: {unknown_groups}")
    return [
        column
        for group in OPTIONAL_FEATURE_GROUPS
        if group in feature_groups
        for column in OPTIONAL_FEATURE_GROUPS[group]
    ]


def get_classifier_feature_columns(
    feature_groups: Collection[str] = (),
) -> list[str]:
    """Get the feature columns the FDR classifier is trained on.

    The full feature set; the FDR manager uses only the columns present, so calibration rounds use the core subset.
    The optional features are left out unless their group is in `feature_groups`.
    """
    excluded = (
        set(get_feature_group_columns(OPTIONAL_FEATURE_GROUPS))
        - set(get_feature_group_columns(feature_groups))
    ) | set(NON_CLASSIFIER_FEATURES)
    return [
        column
        for column in get_feature_names(FeatureSet.Full)
        if column not in excluded
    ]


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
    candidate_features: CandidateFeatureCollection,
    spectral_library: SpecLibFlat,
    *,
    unscored_feature_groups: Collection[str] = (),
    has_mobility: bool = False,
) -> pd.DataFrame:
    """Convert alphadia-search-rs candidate features into a DataFrame.

    The columns of the `unscored_feature_groups` (`OPTIONAL_FEATURE_GROUPS`) hold the missing values and are dropped,
    so that the classifier never sees them, also when their flag is set but the file cannot score them. So are the
    `MOBILITY_FEATURES` without `has_mobility`; with it, the mobility peak width is added as `fwhm_mobility`.
    """
    dropped = set(get_feature_group_columns(unscored_feature_groups))
    if not has_mobility:
        dropped |= set(MOBILITY_FEATURES) | set(
            OPTIONAL_FEATURE_GROUPS["mobility_profile_features"]
        )
    features_dict = {
        name: values
        for name, values in candidate_features.to_dict_arrays().items()
        if name not in dropped
    }

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
    if has_mobility:
        features_df["fwhm_mobility"] = FWHM_PER_SD * features_df["mobility_peak_sd"]

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

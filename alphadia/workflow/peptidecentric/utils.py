from functools import wraps

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.constants.keys import CalibCols, FragmentCols
from alphadia.outputtransform.quantification.quant_builder import ION_HASH_COLUMNS
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    get_feature_names,
)

feature_columns = [
    "reference_intensity_correlation",
    "mean_reference_scan_cosine",
    "top3_reference_scan_cosine",
    "mean_reference_frame_cosine",
    "top3_reference_frame_cosine",
    "mean_reference_template_scan_cosine",
    "mean_reference_template_frame_cosine",
    "mean_reference_template_frame_cosine_rank",
    "mean_reference_template_scan_cosine_rank",
    "mean_reference_frame_cosine_rank",
    "mean_reference_scan_cosine_rank",
    "reference_intensity_correlation_rank",
    "top3_b_ion_correlation_rank",
    "top3_y_ion_correlation_rank",
    "top3_frame_correlation_rank",
    "fragment_frame_correlation_rank",
    "weighted_ms1_intensity_rank",
    "isotope_intensity_correlation_rank",
    "isotope_pattern_correlation_rank",
    "mono_ms1_intensity_rank",
    "weighted_mass_error_rank",
    "base_width_mobility",
    "base_width_rt",
    CalibCols.RT_OBSERVED,
    "delta_rt",
    CalibCols.MOBILITY_OBSERVED,
    "mono_ms1_intensity",
    "top_ms1_intensity",
    "sum_ms1_intensity",
    "weighted_ms1_intensity",
    "weighted_mass_deviation",
    "weighted_mass_error",
    CalibCols.MZ_LIBRARY,
    CalibCols.MZ_OBSERVED,
    "mono_ms1_height",
    "top_ms1_height",
    "sum_ms1_height",
    "weighted_ms1_height",
    "isotope_intensity_correlation",
    "isotope_height_correlation",
    "n_observations",
    "intensity_correlation",
    "height_correlation",
    "intensity_fraction",
    "height_fraction",
    "intensity_fraction_weighted",
    "height_fraction_weighted",
    "mean_observation_score",
    "sum_b_ion_intensity",
    "sum_y_ion_intensity",
    "diff_b_y_ion_intensity",
    "fragment_scan_correlation",
    "top3_scan_correlation",
    "fragment_frame_correlation",
    "top3_frame_correlation",
    "template_scan_correlation",
    "template_frame_correlation",
    "top3_b_ion_correlation",
    "top3_y_ion_correlation",
    "n_b_ions",
    "n_y_ions",
    "f_masked",
    "fwhm_rt",
    "fwhm_mobility",
    "top_3_ms2_mass_error",
    "mean_ms2_mass_error",
    "n_overlapping",
    "mean_overlapping_intensity",
    "mean_overlapping_mass_error",
]


IDF_FEATURE_COLUMNS = (
    "idf_hyperscore",
    "idf_xic_dot_product",
    "idf_intensity_dot_product",
    "num_over_0_top6_idf",
    "num_over_50_top6_idf",
)


def get_classifier_feature_columns(
    extraction_backend: str, use_idf_features: bool
) -> list[str]:
    """Get the feature columns the FDR classifier is trained on.

    The candidate context features are always listed for the rust backend: the FDR manager
    only uses the listed columns that the features actually carry, so the config flag that
    computes them does not need to be repeated here.
    The IDF features are left out unless `use_idf_features` is set.
    """
    if extraction_backend != "rust":
        return feature_columns

    columns = get_feature_names() + get_context_feature_names()
    if use_idf_features:
        return columns
    return [column for column in columns if column not in IDF_FEATURE_COLUMNS]


def add_library_fragment_intensity(
    fragments_df: pd.DataFrame, spectral_library: SpecLibFlat
) -> pd.DataFrame:
    """Add the library intensity of every quantified fragment as `FragmentCols.INTENSITY_LIBRARY`.

    The library-anchored precursor rollup compares the observed fragment intensities with the
    library pattern, so the pattern is stored next to the observed intensities in frag.parquet.
    Fragments not found in the library get 0.

    Parameters
    ----------
    fragments_df : pd.DataFrame
        Quantified fragments with the `ION_HASH_COLUMNS`, which identify a fragment ion
    spectral_library : SpecLibFlat
        The library the fragments were quantified from

    Returns
    -------
    pd.DataFrame
        The fragments in the same order with the library intensity column added
    """
    if FragmentCols.INTENSITY_LIBRARY in fragments_df.columns:
        return fragments_df

    precursor_df = spectral_library.precursor_df
    precursor_df = precursor_df[
        precursor_df["precursor_idx"].isin(fragments_df["precursor_idx"].unique())
    ]
    start = precursor_df["flat_frag_start_idx"].to_numpy(dtype=np.int64)
    n_fragments = precursor_df["flat_frag_stop_idx"].to_numpy(dtype=np.int64) - start
    first_row = np.cumsum(n_fragments) - n_fragments
    flat_idx = np.repeat(start - first_row, n_fragments) + np.arange(n_fragments.sum())

    library_df = spectral_library.fragment_df.iloc[flat_idx][
        ION_HASH_COLUMNS[1:] + ["intensity"]
    ].rename(columns={"intensity": FragmentCols.INTENSITY_LIBRARY})
    library_df.insert(
        0,
        "precursor_idx",
        np.repeat(precursor_df["precursor_idx"].to_numpy(), n_fragments),
    )
    library_df = library_df.astype(
        {c: fragments_df[c].dtype for c in ION_HASH_COLUMNS}
    ).drop_duplicates(subset=ION_HASH_COLUMNS)

    fragments_df = fragments_df.merge(library_df, on=ION_HASH_COLUMNS, how="left")
    fragments_df[FragmentCols.INTENSITY_LIBRARY] = (
        fragments_df[FragmentCols.INTENSITY_LIBRARY].fillna(0).astype(np.float32)
    )
    return fragments_df


def log_precursor_df(reporter: Pipeline, precursor_df: pd.DataFrame) -> None:
    total_precursors = len(precursor_df)

    total_precursors_denom = max(
        float(total_precursors), 1e-6
    )  # avoid division by zero

    target_precursors = len(precursor_df[precursor_df["decoy"] == 0])
    target_precursors_percentages = target_precursors / total_precursors_denom * 100
    decoy_precursors = len(precursor_df[precursor_df["decoy"] == 1])
    decoy_precursors_percentages = decoy_precursors / total_precursors_denom * 100

    reporter.log_string(
        "============================= Precursor FDR =============================",
        verbosity="progress",
    )
    reporter.log_string(
        f"Total precursors accumulated: {total_precursors:,}", verbosity="progress"
    )
    reporter.log_string(
        f"Target precursors: {target_precursors:,} ({target_precursors_percentages:.2f}%)",
        verbosity="progress",
    )
    reporter.log_string(
        f"Decoy precursors: {decoy_precursors:,} ({decoy_precursors_percentages:.2f}%)",
        verbosity="progress",
    )

    reporter.log_string("", verbosity="progress")
    reporter.log_string("Precursor Summary:", verbosity="progress")

    for channel in precursor_df["channel"].unique():
        fdr_counts = {
            threshold: len(
                precursor_df[
                    (precursor_df["qval"] < threshold)
                    & (precursor_df["decoy"] == 0)
                    & (precursor_df["channel"] == channel)
                ]
            )
            for threshold in [0.05, 0.01, 0.001]
        }
        reporter.log_string(
            f"Channel {channel:>3}:\t "
            + "; ".join(
                f"{threshold:.3f} FDR: {fdr_counts[threshold]:>5,}"
                for threshold, fdr_count in fdr_counts.items()
            ),
            verbosity="progress",
        )

    reporter.log_string("", verbosity="progress")
    reporter.log_string("Protein Summary:", verbosity="progress")

    for channel in precursor_df["channel"].unique():
        fdr_counts = {
            threshold: precursor_df[
                (precursor_df["qval"] < threshold)
                & (precursor_df["decoy"] == 0)
                & (precursor_df["channel"] == channel)
            ]["proteins"].nunique()
            for threshold in [0.05, 0.01, 0.001]
        }
        reporter.log_string(
            f"Channel {channel:>3}:\t "
            + "; ".join(
                f"{threshold:.3f} FDR: {fdr_counts[threshold]:>5,}"
                for threshold, fdr_count in fdr_counts.items()
            ),
            verbosity="progress",
        )

    reporter.log_string(
        "=========================================================================",
        verbosity="progress",
    )


def use_timing_manager(phase_name: str):
    """Decorator to record timing in TimingManager for a specific phase.

    Works only if the first argument of the decorated function is a workflow instance with `timing_manager` attribute,
    will do nothing otherwise.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            workflow_instance = args[0] if args else None
            is_timing_supported = workflow_instance and hasattr(
                workflow_instance, "timing_manager"
            )

            if is_timing_supported:
                workflow_instance.timing_manager.set_start_time(phase_name)

            result = func(*args, **kwargs)

            if is_timing_supported:
                workflow_instance.timing_manager.set_end_time(phase_name)

            return result

        return wrapper

    return decorator

from functools import wraps

import pandas as pd

from alphadia.constants.keys import CalibCols
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


# The rust backend's IDF is computed over every library fragment, decoys included. The DIA-NN decoy
# mutation maps residues 2 and n-1 onto a few amino acids, so decoy fragments crowd their own m/z bins
# and get a low IDF, while the fragments of any target, false ones included, do not. These features
# let the classifier recognise decoys from the library alone, so false targets pass as targets.
DECOY_SCHEME_FEATURES = (
    "idf_hyperscore",
    "idf_xic_dot_product",
    "idf_intensity_dot_product",
    "num_over_0_top6_idf",
    "num_over_50_top6_idf",
)


# Context features left out of the classifier. Candidate density, competitor count and the fraction of fragments shared
# with anyone measure how crowded a candidate's spot is, whatever the competitors score; the decoy mutation moves decoy
# fragments onto masses of their own, so decoys see a different crowd than false targets, and on 5 ng paired entrapment
# each raised the paired FDP at 1 % q by 0.2-0.35 pp. The count and rank of higher-scoring competitors cost
# identifications at equal true FDP. The features weighing the fragments shared with higher-scoring competitors add
# identifications without shifting the FDP.
CROWDING_CONTEXT_FEATURES = (
    "ctx_candidate_density",
    "ctx_n_competitors",
    "ctx_n_competitors_higher",
    "ctx_claimant_rank",
    "ctx_shared_frac_any",
)


def get_classifier_feature_columns(extraction_backend: str) -> list[str]:
    """Get the feature columns the FDR classifier is trained on.

    The candidate context features are always listed for the rust backend: the FDR manager
    only uses the listed columns that the features actually carry, so the config flag that
    computes them does not need to be repeated here.
    """
    if extraction_backend != "rust":
        return feature_columns

    return [
        name
        for name in get_feature_names() + get_context_feature_names()
        if name not in DECOY_SCHEME_FEATURES and name not in CROWDING_CONTEXT_FEATURES
    ]


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

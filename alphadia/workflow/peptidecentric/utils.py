from functools import wraps

import pandas as pd

from alphadia.constants.keys import CalibCols
from alphadia.reporting.reporting import Pipeline

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
    "cycle_fwhm",
    "mobility_fwhm",
    "top_3_ms2_mass_error",
    "mean_ms2_mass_error",
    "n_overlapping",
    "mean_overlapping_intensity",
    "mean_overlapping_mass_error",
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

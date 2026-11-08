from functools import wraps

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.outputtransform.quantification.quant_builder import (
    INTENSITY_LIBRARY_COLUMN,
    ION_HASH_COLUMNS,
)
from alphadia.reporting.reporting import Pipeline


def add_library_fragment_intensity(
    fragments_df: pd.DataFrame, spectral_library: SpecLibFlat
) -> pd.DataFrame:
    """Add the library intensity of every quantified fragment as `INTENSITY_LIBRARY_COLUMN`.

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
    precursor_df = spectral_library.precursor_df
    precursor_df = precursor_df[
        precursor_df["precursor_idx"].isin(fragments_df["precursor_idx"].unique())
    ]
    start = precursor_df["flat_frag_start_idx"].to_numpy(dtype=np.int64)
    n_fragments = precursor_df["flat_frag_stop_idx"].to_numpy(dtype=np.int64) - start
    first_row = np.cumsum(n_fragments) - n_fragments
    flat_idx = np.repeat(start - first_row, n_fragments) + np.arange(n_fragments.sum())

    intensity_library_df = spectral_library.fragment_df.iloc[flat_idx][
        ION_HASH_COLUMNS[1:] + ["intensity"]
    ].rename(columns={"intensity": INTENSITY_LIBRARY_COLUMN})
    intensity_library_df.insert(
        0,
        "precursor_idx",
        np.repeat(precursor_df["precursor_idx"].to_numpy(), n_fragments),
    )
    intensity_library_df = intensity_library_df.astype(
        {c: fragments_df[c].dtype for c in ION_HASH_COLUMNS}
    ).drop_duplicates(subset=ION_HASH_COLUMNS)

    fragments_df = fragments_df.merge(
        intensity_library_df, on=ION_HASH_COLUMNS, how="left"
    )
    fragments_df[INTENSITY_LIBRARY_COLUMN] = (
        fragments_df[INTENSITY_LIBRARY_COLUMN].fillna(0).astype(np.float32)
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

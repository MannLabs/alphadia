import pandas as pd
from reporting.reporting import Pipeline

from alphadia.data.alpharaw_wrapper import AlphaRawJIT
from alphadia.data.bruker import TimsTOFTranspose
from alphadia.workflow.config import Config
from alphadia.workflow.managers.fdr_manager import FDRManager


def fdr_correction(
    fdr_manager: FDRManager,
    config: Config,
    dia_data: TimsTOFTranspose | AlphaRawJIT,
    features_df: pd.DataFrame,
    df_fragments: pd.DataFrame,
    version: int = -1,
) -> pd.DataFrame:
    """Peptide-centric specific FDR correction."""
    return fdr_manager.fit_predict(
        features_df,
        decoy_strategy="precursor_channel_wise"
        if config["fdr"]["channel_wise_fdr"]
        else "precursor",
        competetive=config["fdr"]["competetive_scoring"],
        df_fragments=df_fragments
        if config["search"]["compete_for_fragments"]
        else None,
        dia_cycle=dia_data.dia_cycle,
        version=version,
    )


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
        precursor_05fdr = len(
            precursor_df[
                (precursor_df["qval"] < 0.05)
                & (precursor_df["decoy"] == 0)
                & (precursor_df["channel"] == channel)
            ]
        )
        precursor_01fdr = len(
            precursor_df[
                (precursor_df["qval"] < 0.01)
                & (precursor_df["decoy"] == 0)
                & (precursor_df["channel"] == channel)
            ]
        )
        precursor_001fdr = len(
            precursor_df[
                (precursor_df["qval"] < 0.001)
                & (precursor_df["decoy"] == 0)
                & (precursor_df["channel"] == channel)
            ]
        )
        reporter.log_string(
            f"Channel {channel:>3}:\t 0.05 FDR: {precursor_05fdr:>5,}; 0.01 FDR: {precursor_01fdr:>5,}; 0.001 FDR: {precursor_001fdr:>5,}",
            verbosity="progress",
        )

    reporter.log_string("", verbosity="progress")
    reporter.log_string("Protein Summary:", verbosity="progress")

    for channel in precursor_df["channel"].unique():
        proteins_05fdr = precursor_df[
            (precursor_df["qval"] < 0.05)
            & (precursor_df["decoy"] == 0)
            & (precursor_df["channel"] == channel)
        ]["proteins"].nunique()
        proteins_01fdr = precursor_df[
            (precursor_df["qval"] < 0.01)
            & (precursor_df["decoy"] == 0)
            & (precursor_df["channel"] == channel)
        ]["proteins"].nunique()
        proteins_001fdr = precursor_df[
            (precursor_df["qval"] < 0.001)
            & (precursor_df["decoy"] == 0)
            & (precursor_df["channel"] == channel)
        ]["proteins"].nunique()
        reporter.log_string(
            f"Channel {channel:>3}:\t 0.05 FDR: {proteins_05fdr:>5,}; 0.01 FDR: {proteins_01fdr:>5,}; 0.001 FDR: {proteins_001fdr:>5,}",
            verbosity="progress",
        )

    reporter.log_string(
        "=========================================================================",
        verbosity="progress",
    )

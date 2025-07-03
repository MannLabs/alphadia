import pandas as pd

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

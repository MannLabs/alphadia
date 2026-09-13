import logging

import pandas as pd

from alphadia.fdr import fdr

logger = logging.getLogger()


def _build_protein_features(psm_df: pd.DataFrame) -> pd.DataFrame:
    return (
        psm_df.groupby(["pg", "decoy"])
        .agg(
            genes=("genes", "first"),
            proteins=("proteins", "first"),
            count=("proba", "size"),
            n_precursor=("precursor_idx", "nunique"),
            n_peptides=("sequence", "nunique"),
            n_runs=("run", "nunique"),
            mean_score=("proba", "mean"),
            proba=("proba", "min"),
            worst_score=("proba", "max"),
        )
        .reset_index()
    )


def perform_protein_fdr(psm_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate a q-value for every protein group in the PSM dataframe.

    Parameters
    ----------
    psm_df : pd.DataFrame
        Precursors carrying a protein group assignment.

    Returns
    -------
    pd.DataFrame
        One row per protein group, with its `pg_qval`.

    """
    protein_features = _build_protein_features(psm_df)

    n_targets = (protein_features["decoy"] == 0).sum()
    n_decoys = (protein_features["decoy"] == 1).sum()
    logger.info(f"Protein FDR over {n_targets:,} target and {n_decoys:,} decoy groups")

    return fdr.get_q_values(
        protein_features,
        score_column="proba",
        decoy_column="decoy",
        qval_column="pg_qval",
        extra_sort_columns=["pg"],
    )

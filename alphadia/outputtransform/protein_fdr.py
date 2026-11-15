import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alphadia.exceptions import TooFewProteinsError
from alphadia.fdr import fdr
from alphadia.fdr.plotting import plot_fdr

logger = logging.getLogger()

# Only precursors at or below this q-value contribute to the score of their protein group, and a group without any of
# them gets NO_EVIDENCE_QVAL. Precursors above it but within fdr.fdr stay in the output if their group passes.
# The cutoff is fixed instead of following fdr.fdr: it only has an effect when fdr.fdr is set higher (e.g. 10 %),
# and the precursors this adds are then mostly false at low input.
MAX_PRECURSOR_QVAL_FOR_PROTEIN_SCORING = 0.01
NO_EVIDENCE_QVAL = 1.0

# count features (precursors, peptides, runs per group) reward the false targets that pile up in large groups.
FEATURE_COLUMNS = ["mean_score", "best_score", "worst_score"]
MAX_ITER = 1000
# The features are taken on the logit of the precursor probability.
MIN_PROBA = 1e-300
MAX_PROBA = 1 - 1e-16


def perform_protein_fdr(psm_df: pd.DataFrame, figure_path: str) -> pd.DataFrame:
    """Perform protein FDR on PSM dataframe"""

    confident = psm_df.loc[
        psm_df["qval"] <= MAX_PRECURSOR_QVAL_FOR_PROTEIN_SCORING,
        ["pg", "decoy", "proba"],
    ]
    proba = confident["proba"].clip(MIN_PROBA, MAX_PROBA)
    confident = confident.assign(logit=np.log(proba) - np.log1p(-proba))
    protein_features = (
        confident.groupby(["pg", "decoy"])["logit"]
        .agg(mean_score="mean", best_score="min", worst_score="max")
        .reset_index()
    )

    y = protein_features["decoy"].values
    if len(set(y)) < 2:
        raise TooFewProteinsError("protein FDR needs target and decoy groups")

    X_scaled = StandardScaler().fit_transform(protein_features[FEATURE_COLUMNS].values)
    classifier = LogisticRegression(max_iter=MAX_ITER).fit(X_scaled, y)
    protein_features["proba"] = classifier.predict_proba(X_scaled)[:, 1]

    protein_features = fdr.get_q_values(
        protein_features,
        score_column="proba",
        decoy_column="decoy",
        qval_column="pg_qval",
        extra_sort_columns=["pg"],
    )

    n_targets = (protein_features["decoy"] == 0).sum()
    n_decoys = (protein_features["decoy"] == 1).sum()

    logger.info(f"Protein FDR over {n_targets:,} target and {n_decoys:,} decoy groups")

    if figure_path is not None:
        # get_q_values sorts the groups, so the labels are taken from the sorted frame, not from `y`
        sorted_y = protein_features["decoy"].to_numpy()
        plot_fdr(
            sorted_y,
            sorted_y,
            protein_features["proba"].values,
            protein_features["proba"].values,
            protein_features["pg_qval"],
            figure_path,
        )

    scored = psm_df.merge(
        protein_features[["pg", "decoy", "pg_qval"]], on=["pg", "decoy"], how="left"
    )
    scored["pg_qval"] = scored["pg_qval"].fillna(NO_EVIDENCE_QVAL)
    return scored

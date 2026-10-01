import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alphadia.exceptions import TooFewProteinsError
from alphadia.fdr import fdr
from alphadia.fdr.plotting import plot_fdr

logger = logging.getLogger()

# Make the protein FDR more conservative
DECOY_COUNT_OFFSET = 1

# Only confident precursors score a protein group. Weak ones (the output keeps precursors up to fdr.fdr, e.g. 10 %)
# are mostly false at low input, and their excess over the decoys gets summed into every group they hit.
PROTEIN_FEATURE_QVAL = 0.01
NO_EVIDENCE_QVAL = 1.0

# A linear model on the precursor scores of a group, fit on all groups: plasma brings only ~30 decoy groups, on which
# an MLP fit on a split varies from useful to worse than random between seeds and then accepts no group at all, and
# count features (precursors, peptides, runs per group) reward the false targets that pile up in large groups.
FEATURE_COLUMNS = ["mean_score", "best_score", "worst_score"]
MAX_ITER = 1000


def perform_protein_fdr(psm_df: pd.DataFrame, figure_path: str) -> pd.DataFrame:
    """Perform protein FDR on PSM dataframe"""

    confident = psm_df[psm_df["qval"] <= PROTEIN_FEATURE_QVAL]
    protein_features = (
        confident.groupby(["pg", "decoy"])["proba"]
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
        decoy_offset=DECOY_COUNT_OFFSET,
    )

    n_targets = (protein_features["decoy"] == 0).sum()
    n_decoys = (protein_features["decoy"] == 1).sum()

    logger.info(f"Protein FDR over {n_targets:,} target and {n_decoys:,} decoy groups")

    if figure_path is not None:
        plot_fdr(
            y,
            y,
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

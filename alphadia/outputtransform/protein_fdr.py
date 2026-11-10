import logging

import numpy as np
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
# The features are taken on the logit of the precursor probability. The final-round probabilities of the
# confident precursors spread over many decades (5 ng: targets 1e-7 to 1e-2, median 1e-4; decoys around 1e-2), so on
# the probability scale every confident target sits at 0, the group means follow the worst precursor and the model
# weights the best precursor negatively; on the logit scale the best precursor carries the ranking (paired entrapment,
# real groups at matched FDP: 200 ng +10 %, 5 ng +4-7 %, 1 ft unchanged).
MIN_PROBA = 1e-300
MAX_PROBA = 1 - 1e-16


def perform_protein_fdr(psm_df: pd.DataFrame, figure_path: str) -> pd.DataFrame:
    """Perform protein FDR on PSM dataframe"""

    confident = psm_df.loc[
        psm_df["qval"] <= PROTEIN_FEATURE_QVAL, ["pg", "decoy", "proba"]
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
        decoy_offset=DECOY_COUNT_OFFSET,
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


def transfer_library_pg_qval(
    psm_df: pd.DataFrame, library_pg_qval: pd.DataFrame
) -> pd.DataFrame:
    """Give each target protein group with a confident precursor its q-value from the search that built the MBR
    library, and every other group NO_EVIDENCE_QVAL.

    The MBR step only searches the groups that passed the first search, and the false groups among them were counted by
    that search's protein FDR. A protein FDR of its own would count decoys of groups that are already accepted, so a few
    of its decoys scoring into the targets cut accepted groups (paired entrapment, 5 ng: protein FDP 0.26-0.49 % at
    q 1 % with an own estimate, 0.58-1.21 % with the library q-value).
    """
    is_target = (psm_df["decoy"] == 0).to_numpy()
    confident_groups = psm_df.loc[
        is_target & (psm_df["qval"] <= PROTEIN_FEATURE_QVAL).to_numpy(), "pg"
    ].unique()
    library_qval = (
        psm_df["pg"].map(library_pg_qval.set_index("pg")["pg_qval"]).to_numpy()
    )
    keep = (
        is_target
        & psm_df["pg"].isin(confident_groups).to_numpy()
        & ~np.isnan(library_qval)
    )
    return psm_df.assign(pg_qval=np.where(keep, library_qval, NO_EVIDENCE_QVAL))

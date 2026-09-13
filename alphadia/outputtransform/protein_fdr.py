import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from alphadia.exceptions import TooFewProteinsError
from alphadia.fdr import fdr
from alphadia.fdr.plotting import plot_fdr

logger = logging.getLogger()

FEATURE_COLUMNS = [
    "count",
    "mean_score",
    "n_peptides",
    "n_precursor",
    "n_runs",
    "best_score",
    "worst_score",
]
N_FOLDS = 5
RANDOM_STATE = 0


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
            best_score=("proba", "min"),
            worst_score=("proba", "max"),
        )
        .reset_index()
    )


def _cross_fitted_decoy_probability(
    protein_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Probability of being a decoy group, out of fold and in fold.

    Scoring a group with a model that was trained on it lets the classifier memorise the few
    thousand rows it is given, which moves groups across the q-value threshold for no reason a
    rerun reproduces. The in-fold probabilities are only there to show how much was memorised.
    """
    x = protein_features[FEATURE_COLUMNS].to_numpy()
    y = protein_features["decoy"].to_numpy()

    n_decoys = int(y.sum())
    if min(n_decoys, len(y) - n_decoys) < N_FOLDS:
        raise TooFewProteinsError()

    out_of_fold = np.zeros(len(protein_features))
    in_fold = np.zeros(len(protein_features))
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in folds.split(x, y):
        scaler = StandardScaler().fit(x[train_idx])
        classifier = MLPClassifier(random_state=RANDOM_STATE).fit(
            scaler.transform(x[train_idx]), y[train_idx]
        )
        out_of_fold[test_idx] = classifier.predict_proba(scaler.transform(x[test_idx]))[
            :, 1
        ]
        in_fold[train_idx] += classifier.predict_proba(scaler.transform(x[train_idx]))[
            :, 1
        ] / (N_FOLDS - 1)
    return out_of_fold, in_fold


def perform_protein_fdr(psm_df: pd.DataFrame, figure_path: str | None) -> pd.DataFrame:
    """Estimate a q-value for every protein group in the PSM dataframe.

    Parameters
    ----------
    psm_df : pd.DataFrame
        Precursors carrying a protein group assignment.

    figure_path : str | None
        Directory the FDR figure is written to.

    Returns
    -------
    pd.DataFrame
        One row per protein group, with its `pg_qval`.

    """
    protein_features = _build_protein_features(psm_df)
    protein_features["proba"], in_fold_proba = _cross_fitted_decoy_probability(
        protein_features
    )
    protein_features["in_fold_proba"] = in_fold_proba

    n_targets = (protein_features["decoy"] == 0).sum()
    n_decoys = (protein_features["decoy"] == 1).sum()
    logger.info(f"Protein FDR over {n_targets:,} target and {n_decoys:,} decoy groups")

    protein_features = fdr.get_q_values(
        protein_features,
        score_column="proba",
        decoy_column="decoy",
        qval_column="pg_qval",
        extra_sort_columns=["pg"],
    )

    if figure_path is not None:
        is_decoy = protein_features["decoy"].to_numpy()
        plot_fdr(
            is_decoy,
            is_decoy,
            protein_features["in_fold_proba"].to_numpy(),
            protein_features["proba"].to_numpy(),
            protein_features["pg_qval"],
            figure_path,
        )

    return protein_features.drop(columns=["in_fold_proba"])

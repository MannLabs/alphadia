import logging

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from alphadia.exceptions import TooFewProteinsError
from alphadia.fdr import fdr
from alphadia.fdr.plotting import plot_fdr
from alphadia.fdr.utils import train_test_split_

logger = logging.getLogger()


def perform_protein_fdr(psm_df: pd.DataFrame, figure_path: str) -> pd.DataFrame:
    """Perform protein FDR on PSM dataframe"""

    protein_features = []
    for _, group in psm_df.groupby(["pg", "decoy"]):
        protein_features.append(
            {
                "pg": group["pg"].iloc[0],
                "genes": group["genes"].iloc[0],
                "proteins": group["proteins"].iloc[0],
                "decoy": group["decoy"].iloc[0],
                "count": len(group),
                "n_precursor": len(group["precursor_idx"].unique()),
                "n_peptides": len(group["sequence"].unique()),
                "n_runs": len(group["run"].unique()),
                "mean_score": group["proba"].mean(),
                "best_score": group["proba"].min(),
                "worst_score": group["proba"].max(),
            }
        )

    feature_columns = [
        "count",
        "mean_score",
        "n_peptides",
        "n_precursor",
        "n_runs",
        "best_score",
        "worst_score",
    ]

    protein_features = pd.DataFrame(protein_features)

    X = protein_features[feature_columns].values
    y = protein_features["decoy"].values

    X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split_(
        X,
        y,
        test_size=0.2,
        random_state=42,  # we do this only once so a fixed random state is fine
        exception=TooFewProteinsError,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)

    classifier = MLPClassifier(
        random_state=0  # we do this only once so a fixed random state is fine
    ).fit(X_train_scaled, y_train)

    predicted_proba = classifier.predict_proba(X_scaled)[:, 1]

    protein_features["proba"] = predicted_proba
    protein_features = pd.DataFrame(protein_features)

    protein_features = fdr.get_q_values(
        protein_features,
        score_column="proba",
        decoy_column="decoy",
        qval_column="pg_qval",
        extra_sort_columns=["pg"],
    )

    n_targets = (protein_features["decoy"] == 0).sum()
    n_decoys = (protein_features["decoy"] == 1).sum()

    logger.info(
        f"Normalizing q-values using {n_targets:,} targets and {n_decoys:,} decoys"
    )

    protein_features["pg_qval"] = protein_features["pg_qval"] * n_targets / n_decoys

    if figure_path is not None:
        plot_fdr(
            y_train,
            y_test,
            predicted_proba[idxs_train],
            predicted_proba[idxs_test],
            protein_features["pg_qval"],
            figure_path,
        )

    return pd.concat(
        [
            psm_df[psm_df["decoy"] == 0].merge(
                protein_features[protein_features["decoy"] == 0][["pg", "pg_qval"]],
                on="pg",
                how="left",
            ),
            psm_df[psm_df["decoy"] == 1].merge(
                protein_features[protein_features["decoy"] == 1][["pg", "pg_qval"]],
                on="pg",
                how="left",
            ),
        ]
    )

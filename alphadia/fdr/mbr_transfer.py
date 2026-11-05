"""MBR transfers rescored by one classifier fitted on the transfer rows of all runs.

A transfer is an MBR-step PSM of an elution group the first search did not identify in that run. Each run's MBR step
fits its classifier on all of its rows, where the transfers are a minority next to the re-found groups and their
decoys number a few hundred, and writes the final-round features of its transfer rows. Here one classifier, cross-fitted
over elution groups so that no row is scored by a fit that saw its group, scores the transfers of all runs, and their
q-values are recomputed within run and peptide length stratum.
"""

import logging
import os

import numpy as np
import pandas as pd

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching

logger = logging.getLogger()

TRANSFER_FEATURES_FILE_NAME = "transfer_features.parquet"
N_FOLDS = 4
ROW_KEY = ["run", "precursor_idx", "rank"]
NON_FEATURE_COLUMNS = {*fdr.DUMP_ID_COLUMNS, "fold", "proba", "run"}
# the per-run classifier's settings (peptidecentric._get_classifier_base)
CLASSIFIER_PARAMETERS = {
    "test_size": 0.001,
    "batch_size": 5000,
    "learning_rate": 0.001,
    "epochs": 10,
}


def load_transfer_features(folder_list: list[str]) -> pd.DataFrame:
    frames = []
    for folder in folder_list:
        path = os.path.join(folder, TRANSFER_FEATURES_FILE_NAME)
        if os.path.exists(path):
            frames.append(pd.read_parquet(path).assign(run=os.path.basename(folder)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cross_fitted_proba(
    features: pd.DataFrame,
    feature_columns: list[str],
    config_fdr: dict,
    random_state: int | None,
) -> np.ndarray:
    """P(decoy) of every row, each fold scored by a classifier fitted on the other folds (folds by elution group,
    shared by the runs, so a peptide seen in one run is not learned for another)."""
    groups = features["elution_group_idx"].unique()
    fold_of_group = pd.Series(
        np.random.default_rng(random_state).integers(0, N_FOLDS, len(groups)),
        index=groups,
    )
    fold = fold_of_group.loc[features["elution_group_idx"]].to_numpy()
    best_peak_groups = pd.MultiIndex.from_frame(
        features[["run", "precursor_idx"]]
    ).factorize()[0]
    x = features[feature_columns].to_numpy(np.float32)
    y = features["decoy"].to_numpy(float)
    proba = np.ones(len(features))
    for k in range(N_FOLDS):
        train, test = fold != k, fold == k
        if not test.any() or len(np.unique(y[train])) < 2:  # noqa: PLR2004
            continue
        classifier = BinaryClassifierLegacyNewBatching(
            **CLASSIFIER_PARAMETERS,
            experimental_hyperparameter_tuning=config_fdr[
                "enable_nn_hyperparameter_tuning"
            ],
            random_state=random_state,
        )
        classifier.feature_transform = config_fdr["feature_transform"]
        classifier.fit(x[train], y[train], best_peak_groups[train])
        proba[test] = classifier.predict_proba(x[test])[:, 1]
    return proba


def rescore_transfers(
    psm_df: pd.DataFrame,
    transfer: np.ndarray,
    folder_list: list[str],
    config_fdr: dict,
    decoy_multiplicity: int,
    random_state: int | None = None,
) -> pd.DataFrame:
    """`psm_df` (the MBR step's PSMs of all runs, transfers unfiltered) with the transfer rows rescored by the pooled
    classifier and filtered at the configured FDR; the other rows are kept as they are.

    `transfer` marks the transfer rows of `psm_df`. A decoy kept from the first search counts as one false target, a
    fresh decoy as 1 / `decoy_multiplicity`.
    """
    features = load_transfer_features(folder_list)
    keep = ~transfer
    if len(features) == 0:
        logger.warning("No MBR transfer features found, transfers are dropped")
        return psm_df[keep]

    feature_columns = [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
    features["pooled_proba"] = cross_fitted_proba(
        features, feature_columns, config_fdr, random_state
    )
    logger.info(
        f"Pooled MBR transfer model: {len(features):,} transfer rows of "
        f"{features['run'].nunique()} runs, {len(feature_columns)} features"
    )

    rows = psm_df[transfer].merge(
        features[[*ROW_KEY, "pooled_proba", fdr.KEPT_DECOY_COLUMN]],
        on=ROW_KEY,
        how="inner",
    )
    rows["proba"] = rows.pop("pooled_proba")
    rows[fdr.DECOY_WEIGHT_COLUMN] = rows["decoy"].to_numpy() * np.where(
        rows[fdr.KEPT_DECOY_COLUMN].to_numpy(bool), 1.0, 1 / decoy_multiplicity
    )
    rows = pd.concat(
        [
            fdr.get_q_values_by_length(
                run_rows,
                "proba",
                "decoy",
                decoy_weight_column=fdr.DECOY_WEIGHT_COLUMN,
            )
            for _, run_rows in rows.groupby("run")
        ]
    )
    accepted = rows[rows["qval"] <= config_fdr["fdr"]].drop(
        columns=[fdr.KEPT_DECOY_COLUMN, fdr.DECOY_WEIGHT_COLUMN]
    )
    logger.info(
        f"Pooled MBR transfer model: {int((accepted['decoy'] == 0).sum()):,} transfers at q <= {config_fdr['fdr']}"
    )
    return pd.concat([psm_df[keep], accepted], ignore_index=True)


def restrict_run_outputs(
    folder: str, run_psm_df: pd.DataFrame, psm_file: str, frag_file: str
) -> None:
    """Rewrite one run's PSM and fragment tables to the PSMs that passed, so that the quantification, which reads the
    fragment tables by precursor, does not pick up the fragments of transfers rejected in this run."""
    run_psm_df.to_parquet(os.path.join(folder, psm_file), index=False)
    frag_path = os.path.join(folder, frag_file)
    if os.path.exists(frag_path):
        frag_df = pd.read_parquet(frag_path)
        frag_df[frag_df["precursor_idx"].isin(run_psm_df["precursor_idx"])].to_parquet(
            frag_path, index=False
        )

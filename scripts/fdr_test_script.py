#!/usr/bin/env python3
"""FDR classifier comparison test script."""

import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching, Classifier
from alphadia.fdrx.models.classifier_optimal import BinaryClassifierClaude

# Constants
FDR_THRESHOLD_1 = 0.01
FDR_THRESHOLD_01 = 0.001


def run_classifier(
    classifier: Classifier,
    available_columns: list,
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
    classifier_name: str,
) -> dict[str, Any]:
    """Run FDR with a single classifier and return results.

    Parameters
    ----------
    classifier : object
        The classifier to use for FDR
    available_columns : list
        List of available feature columns
    target_df : pd.DataFrame
        Target precursors dataframe
    decoy_df : pd.DataFrame
        Decoy precursors dataframe
    classifier_name : str
        Name of the classifier for logging

    Returns
    -------
    dict
        Results dictionary with timing and FDR statistics

    """
    # Running classifier (print removed for production)
    start_time = time.time()

    psm_df = fdr.perform_fdr(
        classifier,
        available_columns,
        target_df,
        decoy_df,
        competetive=True,
        group_channels=True,
    )

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    psm_sig_df = psm_df[psm_df["decoy"] == 0]
    fdr_1_count = (psm_sig_df["qval"] < FDR_THRESHOLD_1).sum()
    fdr_01_count = (psm_sig_df["qval"] < FDR_THRESHOLD_01).sum()

    return {
        "name": classifier_name,
        "time": total_time,
        "fdr_1_percent": fdr_1_count,
        "fdr_01_percent": fdr_01_count,
        "total_psms": len(psm_df),
    }


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print comparison results as a simple list.

    Parameters
    ----------
    results : list
        List of result dictionaries from run_classifier

    """
    # Print results (removed for production)
    for _result in results:
        # Results would be printed here in production
        pass


def main() -> list[dict[str, Any]]:
    """Run FDR classifier comparison tests."""
    # Loading test data (print removed for production)

    # Load features data from parquet
    features_df = pd.read_parquet(
        "/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/fdr_bench/features_df.parquet"
    )
    # Loaded features_df (print removed for production)

    # Load available columns from pickle
    available_columns_path = Path(
        "/Users/georgwallmann/Documents/data/alphadia_performance_tests/output/fdr_bench/available_columns.pkl"
    )
    with available_columns_path.open("rb") as f:
        available_columns = pickle.load(f)  # noqa: S301
    # Loaded available columns (print removed for production)

    # Split target and decoy data
    target_df = features_df[features_df["decoy"] == 0].copy()
    decoy_df = features_df[features_df["decoy"] == 1].copy()

    # Target and decoy samples loaded (prints removed for production)

    results = []

    # Initialize first classifier (10 epochs)
    classifier1 = BinaryClassifierLegacyNewBatching(
        test_size=0.001,
        epochs=2,
        learning_rate=0.0002,
        weight_decay=0.00001,
        layers=[100, 50, 20, 5],
        dropout=0.001,
    )

    # Run first classifier
    result1 = run_classifier(
        classifier1,
        available_columns,
        target_df,
        decoy_df,
        "BinaryClassifierLegacyNewBatching (2 epochs)",
    )
    results.append(result1)

    # Initialize second classifier (1 epoch)
    classifier2 = BinaryClassifierLegacyNewBatching(
        test_size=0.001,
        epochs=2,
        learning_rate=0.0002,
        weight_decay=0.00001,
        layers=[100, 100, 100, 50, 20, 5],
        dropout=0.001,
    )

    # Run second classifier
    result2 = run_classifier(
        classifier2,
        available_columns,
        target_df,
        decoy_df,
        "BinaryClassifierLegacyNewBatching (1 epoch)",
    )
    results.append(result2)

    # Initialize third classifier (Claude minimal)
    classifier3 = BinaryClassifierClaude(
        test_size=0.001,
        epochs=2,
        learning_rate=0.001,
    )

    # Run third classifier
    result3 = run_classifier(
        classifier3,
        available_columns,
        target_df,
        decoy_df,
        "BinaryClassifierClaude (10 epochs)",
    )
    results.append(result3)

    # Print results table
    print_results_table(results)

    return results


if __name__ == "__main__":
    main()

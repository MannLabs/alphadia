from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from alphadia._fdrx.base import TargetDecoyFDR
from alphadia.fdrexperimental import _get_scaled_training_params


@patch("matplotlib.pyplot.show")
def test_target_decoy_fdr(mock_show):
    # given
    n_samples = 50

    decoy = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    hash = np.arange(n_samples * 2)
    features = np.random.normal(0, 1, (n_samples * 2, 2))
    features += 3 * decoy[:, np.newaxis]

    mock_df = pd.DataFrame(
        {
            "decoy": decoy,
            "mod_seq_charge_hash": hash,
        }
    )

    feature_columns = []

    for i in range(features.shape[1]):
        feature_name = f"feature_{i}"
        mock_df[feature_name] = features[:, i]
        feature_columns.append(feature_name)

    # when
    classifier_mock = LogisticRegression()
    target_decoy_fdr = TargetDecoyFDR(
        classifier=classifier_mock,
        feature_columns=feature_columns,
        competition_columns=["mod_seq_charge_hash"],
    )

    df = target_decoy_fdr.fit_predict_qval(mock_df)

    # then
    assert all([col in df.columns for col in ["decoy_proba", "qval", "pep"]])
    assert np.all(df[["decoy_proba", "qval", "pep"]].values >= 0)
    assert np.all(df[["decoy_proba", "qval", "pep"]].values <= 1)


@pytest.mark.parametrize(
    "n_samples,expected_batch,expected_lr",
    [
        # Large dataset case (≥1M samples)
        (1_000_000, 4096, 0.001),
        (2_000_000, 4096, 0.001),
        # Mid-size dataset cases
        (500_000, 2048, 0.001 * np.sqrt(2048 / 4096)),  # 50% of max
        (250_000, 1024, 0.001 * np.sqrt(1024 / 4096)),  # 25% of max
        # Small dataset cases
        (25_000, 128, 0.001 * np.sqrt(128 / 4096)),  # Should hit min batch size
        (1_000, 128, 0.001 * np.sqrt(128 / 4096)),  # Should hit min batch size
    ],
)
def test_get_scaled_training_params(n_samples, expected_batch, expected_lr):
    # Create dummy dataframe with specified number of samples
    df = pd.DataFrame({"col1": range(n_samples)})

    # Get scaled parameters
    batch_size, learning_rate = _get_scaled_training_params(df)

    # Check batch size matches expected
    assert batch_size == expected_batch

    # Check learning rate matches expected (within floating point precision)
    assert np.isclose(learning_rate, expected_lr, rtol=1e-10)

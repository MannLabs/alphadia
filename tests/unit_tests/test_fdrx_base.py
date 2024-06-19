import numpy as np
import pandas as pd
from unittest.mock import patch
from alphadia.fdrx.base import TargetDecoyFDR
from sklearn.linear_model import LogisticRegression


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

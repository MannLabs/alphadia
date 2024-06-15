from alphadia.fdrx.stats import _keep_best, _get_pep, _get_q_values, _fdr_to_q_values
import pandas as pd
import numpy as np


def test_keep_best():
    # given
    test_df = pd.DataFrame(
        {
            "mod_seq_charge_hash": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "decoy_proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    # when
    best_df = _keep_best(
        test_df, score_column="decoy_proba", group_columns=["mod_seq_charge_hash"]
    )

    # then
    assert best_df.shape[0] == 3
    assert np.allclose(best_df["decoy_proba"].values, np.array([0.1, 0.4, 0.7]))


def test_keep_best_channel():
    # given
    test_df = pd.DataFrame(
        {
            "mod_seq_charge_hash": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "channel": [0, 0, 1, 0, 1, 1, 0, 0, 1],
            "decoy_proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    # when
    best_df = _keep_best(
        test_df,
        score_column="decoy_proba",
        group_columns=["channel", "mod_seq_charge_hash"],
    )

    # then
    assert best_df.shape[0] == 6
    assert np.allclose(
        best_df["decoy_proba"].values, np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.9])
    )


def test_pep():
    # given
    # uniform distribution of decoy probabilities across target and decoy should result in 0.5 PEP on average
    df = pd.DataFrame(
        {
            "decoy_proba": np.random.rand(1000),
            "decoy": np.stack([np.zeros(500), np.ones(500)]).flatten(),
        }
    )

    # when
    pep = _get_pep(df)

    # then
    assert len(pep) == 1000
    assert np.all(pep >= 0)
    assert np.all(pep <= 1)
    assert (np.mean(pep) - 0.5) < 0.1


def test_fdr_to_q_values():
    # given
    test_fdr = np.array([0.2, 0.1, 0.05, 0.3, 0.26, 0.25, 0.5])

    # when
    test_q_values = _fdr_to_q_values(test_fdr)

    # then
    assert np.allclose(
        test_q_values, np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.5])
    )


def test_get_q_values():
    # given
    test_df = pd.DataFrame(
        {
            "mod_seq_charge_hash": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "decoy_proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "decoy": [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        }
    )
    # when
    test_df = _get_q_values(test_df, "decoy_proba", "decoy")

    # then
    assert np.allclose(
        test_df["qval"].values,
        np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]),
    )

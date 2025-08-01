import os
import tempfile
import warnings
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

from alphadia.fdr import fdr
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.workflow.managers.fdr_manager import FDRManager

feature_columns = [
    "base_width_mobility",
    "base_width_rt",
    "cycle_fwhm",
    "diff_b_y_ion_intensity",
    "fragment_frame_correlation",
    "fragment_scan_correlation",
    "height_correlation",
    "height_fraction",
    "height_fraction_weighted",
    "intensity_correlation",
    "intensity_fraction",
    "intensity_fraction_weighted",
    "isotope_height_correlation",
    "isotope_intensity_correlation",
    "mean_observation_score",
    "mobility_fwhm",
    "mobility_observed",
    "mono_ms1_height",
    "mono_ms1_intensity",
    "mz_library",
    "mz_observed",
    "rt_observed",
    "n_observations",
    "sum_b_ion_intensity",
    "sum_ms1_height",
    "sum_ms1_intensity",
    "sum_y_ion_intensity",
    "template_frame_correlation",
    "template_scan_correlation",
    "top3_frame_correlation",
    "top3_scan_correlation",
    "top_ms1_height",
    "top_ms1_intensity",
    "weighted_mass_deviation",
    "weighted_mass_error",
    "weighted_ms1_height",
    "weighted_ms1_intensity",
]

classifier_base = BinaryClassifierLegacyNewBatching(
    test_size=0.01,
    batch_size=100,
    epochs=15,
)


def test_keep_best():
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "channel": [0, 0, 1, 0, 1, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    best_df = fdr.keep_best(
        test_df, score_column="proba", group_columns=["precursor_idx"]
    )

    assert best_df.shape[0] == 3
    assert np.allclose(best_df["proba"].values, np.array([0.1, 0.4, 0.7]))

    best_df = fdr.keep_best(
        test_df, score_column="proba", group_columns=["channel", "precursor_idx"]
    )

    assert best_df.shape[0] == 6
    assert np.allclose(
        best_df["proba"].values, np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.9])
    )


def test_keep_best_2():
    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "elution_group_idx": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )

    result_df = fdr.keep_best(test_df, group_columns=["channel", "elution_group_idx"])
    pd.testing.assert_frame_equal(result_df, test_df)

    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "elution_group_idx": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )
    result_df = fdr.keep_best(test_df, group_columns=["channel", "elution_group_idx"])
    result_expected = pd.DataFrame(
        {
            "channel": [0, 0, 4, 4, 8, 8],
            "elution_group_idx": [0, 1, 0, 1, 0, 1],
            "proba": [0.1, 0.3, 0.4, 0.6, 0.1, 0.3],
        }
    )
    pd.testing.assert_frame_equal(result_df, result_expected)

    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "precursor_idx": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )
    result_df = fdr.keep_best(test_df, group_columns=["channel", "precursor_idx"])
    result_expected = pd.DataFrame(
        {
            "channel": [0, 0, 4, 4, 8, 8],
            "precursor_idx": [0, 1, 0, 1, 0, 1],
            "proba": [0.1, 0.3, 0.4, 0.6, 0.1, 0.3],
        }
    )
    pd.testing.assert_frame_equal(result_df, result_expected)


def test_fdr_to_q_values():
    test_fdr = np.array([0.2, 0.1, 0.05, 0.3, 0.26, 0.25, 0.5])

    test_q_values = fdr._fdr_to_q_values(test_fdr)

    assert np.allclose(
        test_q_values, np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.5])
    )


def test_get_q_values():
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "_decoy": [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        }
    )

    test_df = fdr.get_q_values(test_df, "proba", "_decoy")

    assert np.allclose(
        test_df["qval"].values,
        np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]),
    )


@pytest.mark.slow()
def test_fdr():
    matplotlib.use("Agg")

    # check if TEST_DATA_DIR is in the environment
    if "TEST_DATA_DIR" not in os.environ:
        warnings.warn("TEST_DATA_DIR not in environment, skipping test_fdr")
        return

    # load the data
    test_data_path = os.path.join(
        os.environ["TEST_DATA_DIR"], "fdr_test_psm_channels.tsv"
    )

    if not os.path.isfile(test_data_path):
        warnings.warn(
            "TEST_DATA_DIR is set but fdr_test_psm_channels.tsv test data not found, skipping test_fdr"
        )

    test_df = pd.read_csv(test_data_path, sep="\t")
    if "proba" in test_df.columns:
        test_df.drop(columns=["proba", "qval"], inplace=True)
    # run the fdr

    fdr_manager = FDRManager(
        feature_columns=feature_columns,
        classifier_base=classifier_base,
        config=MagicMock(),
    )

    psm_df = fdr_manager.fit_predict(
        test_df,
        decoy_strategy_overwrite="precursor",
        competetive_overwrite=False,
    )

    regular_channel_ids = psm_df[psm_df["qval"] < 0.01]["channel"].value_counts()

    psm_df = fdr_manager.fit_predict(
        test_df,
        decoy_strategy_overwrite="precursor",
        competetive_overwrite=True,
    )

    competitive_channel_ids = psm_df[psm_df["qval"] < 0.01]["channel"].value_counts()

    assert np.all(competitive_channel_ids.values > regular_channel_ids.values)
    assert np.all(regular_channel_ids.values > 1500)
    assert np.all(competitive_channel_ids.values > 1500)

    psm_df = fdr_manager.fit_predict(
        test_df,
        decoy_strategy_overwrite="precursor_channel_wise",
        competetive_overwrite=True,
    )

    channel_ids = psm_df[psm_df["qval"] < 0.01]["channel"].value_counts()
    assert np.all(channel_ids.values > 1500)

    psm_df = fdr_manager.fit_predict(
        test_df,
        decoy_strategy_overwrite="channel",
        competetive_overwrite=True,
        decoy_channel=8,
    )

    d0_ids = len(psm_df[(psm_df["qval"] < 0.01) & (psm_df["channel"] == 0)])
    d4_ids = len(psm_df[(psm_df["qval"] < 0.01) & (psm_df["channel"] == 4)])

    assert d0_ids > 100
    assert d4_ids < 100


def gen_data_np(
    n_features=10,
    n_samples=10000,
    max_mean=100,
    max_var=0.1,
):
    mean = np.random.random(n_features * 2) * max_mean
    var = np.random.random(n_features * 2) * max_var
    data = np.random.multivariate_normal(
        mean, np.eye(n_features * 2) * var, size=n_samples
    )
    return data.reshape(-1, n_features), np.tile([0, 1], n_samples)


def test_feed_forward():
    x, y = gen_data_np()

    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100,
    )

    classifier.fit(x, y)
    # assert classifier.metrics["test_accuracy"][-1] > 0.99
    # assert classifier.metrics["train_accuracy"][-1] > 0.99

    y_pred = classifier.predict(x)  # noqa: F841  # TODO fix this test
    # assert np.all(y_pred == y)

    y_proba = classifier.predict_proba(x)[:, 1]  # noqa: F841  # TODO fix this test
    # assert np.all(np.round(y_proba) == y)


def test_feed_forward_save():
    tempfolder = tempfile.gettempdir()
    x, y = gen_data_np()

    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100,
    )

    classifier.fit(x, y)

    torch.save(
        classifier.to_state_dict(),
        os.path.join(tempfolder, "test_feed_forward_save.pth"),
    )

    new_classifier = BinaryClassifierLegacyNewBatching()
    new_classifier.from_state_dict(
        torch.load(
            os.path.join(tempfolder, "test_feed_forward_save.pth"), weights_only=False
        )
    )

    y_pred = new_classifier.predict(x)  # noqa: F841  # TODO fix this test
    # assert np.all(y_pred == y)

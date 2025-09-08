import os
import warnings
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pandas as pd
import pytest

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
        decoy_strategy="precursor",
        competetive=False,
    )

    regular_channel_ids = psm_df[psm_df["qval"] < 0.01]["channel"].value_counts()

    psm_df = fdr_manager.fit_predict(
        test_df,
        decoy_strategy="precursor",
        competetive=True,
    )

    competitive_channel_ids = psm_df[psm_df["qval"] < 0.01]["channel"].value_counts()

    assert np.all(competitive_channel_ids.values > regular_channel_ids.values)
    assert np.all(regular_channel_ids.values > 1500)
    assert np.all(competitive_channel_ids.values > 1500)

    psm_df = fdr_manager.fit_predict(
        test_df,
        decoy_strategy="precursor_channel_wise",
        competetive=True,
    )

    channel_ids = psm_df[psm_df["qval"] < 0.01]["channel"].value_counts()
    assert np.all(channel_ids.values > 1500)

    psm_df = fdr_manager.fit_predict(
        test_df,
        decoy_strategy="channel",
        competetive=True,
        decoy_channel=8,
    )

    d0_ids = len(psm_df[(psm_df["qval"] < 0.01) & (psm_df["channel"] == 0)])
    d4_ids = len(psm_df[(psm_df["qval"] < 0.01) & (psm_df["channel"] == 4)])

    assert d0_ids > 100
    assert d4_ids < 100

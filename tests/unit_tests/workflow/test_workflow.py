import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch
from alphadia_search_rs import OptimizationContext

import alphadia
from alphadia.calibration.estimator import CalibrationEstimator
from alphadia.constants.keys import FeatureTransform
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.workflow.managers.base import BaseManager
from alphadia.workflow.managers.calibration_manager import (
    CalibrationManager,
)
from alphadia.workflow.managers.fdr_manager import FDRManager, column_hash
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.peptidecentric import _get_classifier_base


def test_base_manager():
    my_base_manager = BaseManager()
    assert my_base_manager.path is None
    assert my_base_manager.is_loaded_from_file is False


def test_base_manager_save():
    tmp_path = os.path.join(tempfile.gettempdir(), "my_base_manager.pkl")

    my_base_manager = BaseManager(path=tmp_path)
    my_base_manager.save()
    assert os.path.exists(my_base_manager.path)
    os.remove(my_base_manager.path)


def test_base_manager_load():
    tmp_path = os.path.join(tempfile.gettempdir(), "my_base_manager.pkl")

    my_base_manager = BaseManager(path=tmp_path)
    my_base_manager.save()

    my_base_manager_loaded = BaseManager(path=tmp_path, load_from_file=True)
    assert my_base_manager_loaded.path == my_base_manager.path
    assert my_base_manager_loaded.is_loaded_from_file is True

    os.remove(my_base_manager.path)


def test_base_manager_load_version_mismatch_warns_and_loads(tmp_path):
    pkl_path = str(tmp_path / "my_base_manager.pkl")
    my_base_manager = BaseManager(path=pkl_path)
    my_base_manager._version = "0.0.0"
    my_base_manager.save()

    reporter = MagicMock()
    my_base_manager_loaded = BaseManager(
        path=pkl_path, load_from_file=True, reporter=reporter
    )

    assert my_base_manager_loaded.is_loaded_from_file is True
    assert my_base_manager_loaded._version == alphadia.__version__
    warning_calls = [
        c
        for c in reporter.log_string.call_args_list
        if c.kwargs.get("verbosity") == "warning"
    ]
    assert len(warning_calls) == 1
    assert "Version mismatch" in warning_calls[0].args[0]


def test_calibration_manager_init():
    # initialize the calibration manager (mobility not available in the raw data)
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = CalibrationManager(
        path=temp_path, load_from_file=False, has_mobility=False
    )

    assert calibration_manager.path == temp_path
    assert calibration_manager.is_loaded_from_file is False
    assert calibration_manager.all_fitted is False

    assert len(calibration_manager.estimator_groups) == 2
    # precursor: mz + rt (mobility skipped), fragment: mz
    assert len(calibration_manager.estimator_groups["precursor"]) == 2
    assert len(calibration_manager.estimator_groups["fragment"]) == 1

    assert calibration_manager.get_estimator("precursor", "mz").name == "mz"
    assert calibration_manager.get_estimator("precursor", "rt").name == "rt"
    assert calibration_manager.get_estimator("fragment", "mz").name == "mz"

    assert isinstance(
        calibration_manager.get_estimator("precursor", "mz"), CalibrationEstimator
    )
    assert isinstance(
        calibration_manager.get_estimator("precursor", "rt"), CalibrationEstimator
    )
    assert isinstance(
        calibration_manager.get_estimator("fragment", "mz"), CalibrationEstimator
    )

    assert (
        calibration_manager.get_estimator("precursor", "mz").transform_deviation == 1e6
    )
    assert (
        calibration_manager.get_estimator("precursor", "rt").transform_deviation is None
    )


def calibration_testdata():
    np.random.seed(42)
    # create some test data and make sure estimation works
    mz_library = np.linspace(100, 1000, 1000)
    mz_observed = (
        mz_library + np.random.normal(0, 0.0001, 1000) + mz_library * 0.00001 + 0.005
    )

    rt_library = np.linspace(0, 100, 1000)
    rt_observed = (
        rt_library + np.random.normal(0, 0.5, 1000) + np.sin(rt_library * 0.05)
    )

    mobility_library = np.linspace(0, 100, 1000)
    mobility_observed = (
        mobility_library
        + np.random.normal(0, 0.5, 1000)
        + np.sin(mobility_library * 0.05)
    )

    isotope_intensity_correlation = np.linspace(0, 100, 1000)

    return pd.DataFrame(
        {
            "mz_library": mz_library,
            "mz_observed": mz_observed,
            "rt_library": rt_library,
            "rt_observed": rt_observed,
            "mobility_library": mobility_library,
            "mobility_observed": mobility_observed,
            "isotope_intensity_correlation": isotope_intensity_correlation,
        }
    ).copy()


def test_calibration_manager_fit_predict():
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = CalibrationManager(
        path=temp_path, load_from_file=False, has_mobility=False
    )

    test_df = calibration_testdata()

    # fit only the precursor mz calibration
    calibration_manager.fit(test_df, "precursor", plot=False)
    calibration_manager.predict(test_df, "precursor")

    assert "mz_calibrated" in test_df.columns
    assert "rt_calibrated" in test_df.columns
    # will be false as the the fragment mz calibration is not fitted

    assert calibration_manager.all_fitted is False
    assert calibration_manager.is_loaded_from_file is False

    # fit the fragment mz calibration
    calibration_manager.fit(test_df, "fragment", plot=False)

    assert calibration_manager.all_fitted is True


def test_calibration_manager_save_stats():
    calibration_manager = CalibrationManager(load_from_file=False, has_mobility=False)

    test_df = calibration_testdata()
    calibration_manager.fit(test_df, "precursor", plot=False)
    calibration_manager.fit(test_df, "fragment", plot=False)

    assert calibration_manager.all_fitted is True

    stats = calibration_manager.get_stats()
    assert set(stats["precursor"]) == {"mz", "rt"}  # mobility skipped
    assert set(stats["fragment"]) == {"mz"}
    assert set(stats["precursor"]["mz"]) == {"median_bias", "median_variance"}

    stats_path = os.path.join(tempfile.tempdir, "calibration.stats.json")
    calibration_manager.save_stats(stats_path)

    with open(stats_path) as f:
        loaded = json.load(f)
    assert loaded == stats  # round-trips through JSON

    os.remove(stats_path)


TEST_OPTIMIZATION_CONFIG = {
    "search_initial": {
        "ms1_tolerance": 4,
        "ms2_tolerance": 7,
        "rt_tolerance": 200,
        "mobility_tolerance": 0.04,
        "num_candidates": 1,
    },
    "optimization_manager": {
        "fwhm_rt": 5,
        "fwhm_mobility": 0.01,
        "score_cutoff": 50,
    },
}


def test_optimization_manager():
    optimization_manager = OptimizationManager(TEST_OPTIMIZATION_CONFIG)

    assert optimization_manager.fwhm_rt == 5
    assert optimization_manager.fwhm_mobility == 0.01

    assert optimization_manager.is_loaded_from_file is False


def test_optimization_manager_optimization_context_is_a_snapshot():
    # Given: an optimization manager with the test configuration
    optimization_manager = OptimizationManager(TEST_OPTIMIZATION_CONFIG)

    # When: the test takes the optimization context and then updates the manager
    optimization_context = optimization_manager.get_optimization_context()
    optimization_manager.update(ms2_error=3.0)

    # Then: the taken context keeps the old values, and a new context has the update
    assert optimization_context == OptimizationContext(
        rt_error=200.0,
        ms1_error=4.0,
        ms2_error=7.0,
        mobility_error=0.04,
        fwhm_rt=5.0,
        fwhm_mobility=0.01,
        num_candidates=1,
        score_cutoff=50.0,
    )
    assert optimization_manager.get_optimization_context().ms2_error == 3.0


def test_optimization_manager_rt_proportion():
    TEST_OPTIMIZATION_CONFIG_PROPORTION = deepcopy(TEST_OPTIMIZATION_CONFIG)
    TEST_OPTIMIZATION_CONFIG_PROPORTION["search_initial"]["rt_tolerance"] = 0.5
    optimization_manager = OptimizationManager(
        TEST_OPTIMIZATION_CONFIG_PROPORTION, 1200
    )

    assert optimization_manager.fwhm_rt == 5
    assert optimization_manager.fwhm_mobility == 0.01
    optimization_manager.rt_error = 600

    assert optimization_manager.is_loaded_from_file is False


def test_optimization_manager_save_load():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")

    optimization_manager = OptimizationManager(
        TEST_OPTIMIZATION_CONFIG, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file is False

    optimization_manager.save()

    optimization_manager_loaded = OptimizationManager(
        TEST_OPTIMIZATION_CONFIG, path=temp_path, load_from_file=True
    )

    assert optimization_manager_loaded.is_loaded_from_file is True

    os.remove(temp_path)


def test_optimization_manager_fit():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")
    optimization_manager = OptimizationManager(
        TEST_OPTIMIZATION_CONFIG, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file is False

    optimization_manager.update(fwhm_mobility=0.02)

    assert optimization_manager.is_loaded_from_file is False

    assert optimization_manager.fwhm_mobility == 0.02

    optimization_manager.save()
    os.remove(temp_path)


FDR_TEST_BASE_CLASSIFIER = BinaryClassifierLegacyNewBatching(
    test_size=0.001, batch_size=2, learning_rate=0.001, epochs=1
)
FDR_TEST_FEATURES = ["feature_a", "feature_b"]


def fdr_testdata(features):
    test_dict = {}

    for feature in features:
        test_dict[feature] = np.random.normal(50, 2, 100)
    test_dict["decoy"] = np.random.randint(0, 2, 100)
    test_dict["precursor_idx"] = np.random.randint(1000, 10000, 100)

    return pd.DataFrame(test_dict)


def test_fdr_manager():
    fdr_manager = FDRManager(
        feature_columns=FDR_TEST_FEATURES,
        classifier_base=FDR_TEST_BASE_CLASSIFIER,
        config=MagicMock(),
    )

    assert fdr_manager.is_loaded_from_file is False

    assert fdr_manager.feature_columns == FDR_TEST_FEATURES
    assert fdr_manager.classifier_base == FDR_TEST_BASE_CLASSIFIER


def test_fdr_manager_fit_predict_leaves_out_excluded_columns():
    fdr_manager = FDRManager(
        feature_columns=FDR_TEST_FEATURES,
        classifier_base=FDR_TEST_BASE_CLASSIFIER,
        config={
            "search": {"compete_for_fragments": False},
        },
        dia_cycle=None,
    )

    fdr_manager.fit_predict(
        fdr_testdata(FDR_TEST_FEATURES),
        decoy_strategy="precursor",
        competitive=False,
        excluded_columns=["feature_b"],
    )

    assert column_hash(["feature_a"]) in fdr_manager.classifier_store
    assert column_hash(FDR_TEST_FEATURES) not in fdr_manager.classifier_store


def test_fdr_manager_fit_predict():
    fdr_manager = FDRManager(
        feature_columns=FDR_TEST_FEATURES,
        classifier_base=FDR_TEST_BASE_CLASSIFIER,
        config={
            "search": {"compete_for_fragments": False},
        },
        dia_cycle=None,
    )
    test_features_df = fdr_testdata(FDR_TEST_FEATURES)

    # the shipped classifier has a two-output head and is not loaded
    assert len(fdr_manager.classifier_store) == 0

    fdr_manager.fit_predict(
        test_features_df,
        decoy_strategy="precursor",
        competitive=False,
        df_fragments=None,
    )

    assert len(fdr_manager.classifier_store) == 1
    assert fdr_manager.current_version == 0
    assert column_hash(FDR_TEST_FEATURES) in fdr_manager.classifier_store

    fdr_manager.fit_predict(
        test_features_df,
        decoy_strategy="precursor",
        competitive=False,
        df_fragments=None,
        version=0,
    )

    assert fdr_manager.current_version == 1
    assert fdr_manager.get_classifier(FDR_TEST_FEATURES, 0).fitted is True

    fdr_manager.save_classifier_store(tempfile.tempdir)

    fdr_manager_new = FDRManager(
        feature_columns=FDR_TEST_FEATURES,
        classifier_base=FDR_TEST_BASE_CLASSIFIER,
        config=MagicMock(),
    )
    fdr_manager_new.load_classifier_store(tempfile.tempdir)

    temp_path = os.path.join(tempfile.tempdir, f"{column_hash(FDR_TEST_FEATURES)}.pth")

    assert os.path.exists(temp_path)
    assert fdr_manager_new.get_classifier(FDR_TEST_FEATURES).fitted is True

    os.remove(temp_path)


def test_get_classifier_base_passes_feature_transform():
    # When: the base classifier is created with the quantile feature transform
    classifier = _get_classifier_base(feature_transform=FeatureTransform.QUANTILE)

    # Then: the classifier applies the transform
    assert classifier.feature_transform == FeatureTransform.QUANTILE


def test_fdr_manager_skips_stored_two_output_classifiers():
    """The shipped classifier was saved with the former two-output softmax head.

    Loading its weights into the one-logit network would fail, so the store leaves it out.
    """
    shipped_dir = Path(alphadia.__file__).parent / "constants" / "classifier"
    two_output_hashes = [
        f.stem
        for f in shipped_dir.glob("*.pth")
        if torch.load(f, weights_only=False).get("output_dim", 1) != 1
    ]
    assert two_output_hashes

    fdr_manager = FDRManager(
        feature_columns=FDR_TEST_FEATURES,
        classifier_base=FDR_TEST_BASE_CLASSIFIER,
        config=MagicMock(),
    )

    assert not set(two_output_hashes) & set(fdr_manager.classifier_store)

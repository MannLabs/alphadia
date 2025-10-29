import os
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from alphadia.calibration.estimator import CalibrationEstimator
from alphadia.calibration.models import LOESSRegression
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching
from alphadia.reporting import reporting
from alphadia.workflow.config import Config
from alphadia.workflow.managers.base import BaseManager
from alphadia.workflow.managers.calibration_manager import CalibrationManager
from alphadia.workflow.managers.fdr_manager import FDRManager, column_hash
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.optimizers.automatic import (
    AutomaticMobilityOptimizer,
    AutomaticMS1Optimizer,
    AutomaticMS2Optimizer,
    AutomaticRTOptimizer,
)
from alphadia.workflow.optimizers.targeted import (
    TargetedMobilityOptimizer,
    TargetedMS1Optimizer,
    TargetedMS2Optimizer,
    TargetedRTOptimizer,
)
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.peptidecentric import (
    PeptideCentricWorkflow,
    _get_classifier_base,
)
from alphadia.workflow.peptidecentric.utils import feature_columns


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


TEST_CALIBRATION_GROUPS_CONFIG = {
    "precursor": {
        "mz": {
            "model": "LinearRegression",
            "input_columns": ["mz_library"],
            "target_columns": ["mz_observed"],
            "output_columns": ["mz_calibrated"],
            "transform_deviation": 1e6,
        },
        "rt": {
            "model": "LOESSRegression",
            "model_args": {"n_kernels": 2},
            "input_columns": ["rt_library"],
            "target_columns": ["rt_observed"],
            "output_columns": ["rt_calibrated"],
            "transform_deviation": None,
        },
    },
    "fragment": {
        "mz": {
            "model": "LinearRegression",
            "input_columns": ["mz_library"],
            "target_columns": ["mz_observed"],
            "output_columns": ["mz_calibrated"],
            "transform_deviation": 1e6,
        }
    },
}


def test_calibration_manager_init():
    # initialize the calibration manager
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    with patch(
        "alphadia.workflow.managers.calibration_manager.CALIBRATION_GROUPS_CONFIG",
        TEST_CALIBRATION_GROUPS_CONFIG,
    ):
        calibration_manager = CalibrationManager(
            path=temp_path, load_from_file=False, has_mobility=False
        )

    assert calibration_manager.path == temp_path
    assert calibration_manager.is_loaded_from_file is False
    assert calibration_manager.all_fitted is False

    assert len(calibration_manager.estimator_groups) == 2
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

    assert isinstance(
        calibration_manager.get_estimator("precursor", "mz")._model, LinearRegression
    )
    assert isinstance(
        calibration_manager.get_estimator("precursor", "rt")._model, LOESSRegression
    )
    assert isinstance(
        calibration_manager.get_estimator("fragment", "mz")._model, LinearRegression
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
    with patch(
        "alphadia.workflow.managers.calibration_manager.CALIBRATION_GROUPS_CONFIG",
        TEST_CALIBRATION_GROUPS_CONFIG,
    ):
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


def test_calibration_manager_save_load():
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    with patch(
        "alphadia.workflow.managers.calibration_manager.CALIBRATION_GROUPS_CONFIG",
        TEST_CALIBRATION_GROUPS_CONFIG,
    ):
        calibration_manager = CalibrationManager(
            path=temp_path, load_from_file=False, has_mobility=False
        )

    test_df = calibration_testdata()
    calibration_manager.fit(test_df, "precursor", plot=False)
    calibration_manager.fit(test_df, "fragment", plot=False)

    assert calibration_manager.all_fitted is True
    assert calibration_manager.is_loaded_from_file is False

    calibration_manager.save()

    with patch(
        "alphadia.workflow.managers.calibration_manager.CALIBRATION_GROUPS_CONFIG",
        TEST_CALIBRATION_GROUPS_CONFIG,
    ):
        calibration_manager_loaded = CalibrationManager(
            path=temp_path, load_from_file=True, has_mobility=False
        )
    assert calibration_manager_loaded.all_fitted is True
    assert calibration_manager_loaded.is_loaded_from_file is True

    calibration_manager_loaded.predict(test_df, "precursor")

    assert "mz_calibrated" in test_df.columns
    assert "rt_calibrated" in test_df.columns

    os.remove(temp_path)


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

    assert len(fdr_manager.classifier_store) == 1

    fdr_manager.fit_predict(
        test_features_df,
        decoy_strategy="precursor",
        competitive=False,
        df_fragments=None,
    )

    assert len(fdr_manager.classifier_store) == 2
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


def create_workflow_instance():
    config_base_path = os.path.join(
        Path(__file__).parent, "..", "..", "..", "alphadia", "constants", "default.yaml"
    )

    config = Config()
    config.from_yaml(config_base_path)
    config.update([Config({"output_directory": tempfile.mkdtemp()})])
    workflow = PeptideCentricWorkflow(
        "test",
        config,
    )
    workflow.reporter = reporting.Pipeline(
        backends=[
            reporting.LogBackend(),
            reporting.JSONLBackend(path=workflow.path),
            reporting.FigureBackend(path=workflow.path),
        ]
    )
    workflow._calibration_manager = CalibrationManager(
        path=os.path.join(workflow.path, workflow.CALIBRATION_MANAGER_PKL_NAME),
        load_from_file=workflow.config["general"]["reuse_calibration"],
        reporter=workflow.reporter,
        has_mobility=True,
    )

    workflow._optimization_manager = OptimizationManager(
        TEST_OPTIMIZATION_CONFIG,
        path=os.path.join(workflow.path, workflow.OPTIMIZATION_MANAGER_PKL_NAME),
        load_from_file=workflow.config["general"]["reuse_calibration"],
        figure_path=workflow._figure_path,
        reporter=workflow.reporter,
    )

    workflow._fdr_manager = FDRManager(
        feature_columns=feature_columns,
        classifier_base=_get_classifier_base(
            enable_two_step_classifier=workflow.config["fdr"][
                "enable_two_step_classifier"
            ],
            two_step_classifier_max_iterations=workflow.config["fdr"][
                "two_step_classifier_max_iterations"
            ],
            enable_nn_hyperparameter_tuning=workflow.config["fdr"][
                "enable_nn_hyperparameter_tuning"
            ],
            fdr_cutoff=workflow.config["fdr"]["fdr"],
        ),
        config=MagicMock(),
        figure_path=workflow._figure_path,
    )

    class MockDIAData:
        has_mobility = True
        has_ms1 = True

    workflow._dia_data = MockDIAData()

    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        workflow._optimization_handler = OptimizationHandler(
            workflow.config,
            workflow._optimization_manager,
            workflow._calibration_manager,
            fdr_manager=workflow._fdr_manager,
            reporter=workflow.reporter,
            spectral_library=None,
            dia_data=workflow._dia_data,
        )

    class MockOptlock:
        total_elution_groups = 2000
        batch_idx = 1

    workflow._optimization_handler._optlock = MockOptlock()

    return workflow


def test_automatic_ms2_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df2, "fragment", plot=False)

    ms2_optimizer = AutomaticMS2Optimizer(
        100,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow._optimization_handler._optlock,
        workflow.reporter,
    )

    assert ms2_optimizer.has_converged is False
    assert ms2_optimizer.parameter_name == "ms2_error"

    workflow._fdr_manager._current_version += 1
    ms2_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(ms2_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    ms2_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    ms2_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert ms2_optimizer.has_converged is True
    assert (
        ms2_optimizer.history_df.precursor_proportion_detected
        == pd.Series([1000 / 2000, 1001 / 2000, 1002 / 2000])
    ).all()
    assert (
        workflow.optimization_manager.ms2_error
        == ms2_optimizer.history_df.parameter[
            ms2_optimizer.history_df.precursor_proportion_detected.idxmax()
        ]
    )
    assert workflow.optimization_manager.classifier_version == 2


@pytest.mark.parametrize("favour_narrower_optimum", [True, False])
def test_automatic_ms2_optimizer_no_convergence(favour_narrower_optimum):
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df2, "fragment", plot=False)

    ms2_optimizer = AutomaticMS2Optimizer(
        100,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow._optimization_handler._optlock,
        workflow.reporter,
    )
    ms2_optimizer._favour_narrower_optimum = favour_narrower_optimum
    ms2_optimizer.proceed_with_insufficient_precursors(
        calibration_test_df1, calibration_test_df2
    )

    assert ms2_optimizer.has_converged is False
    assert len(ms2_optimizer.history_df) == 1


def test_automatic_rt_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    rt_optimizer = AutomaticRTOptimizer(
        100,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow._optimization_handler._optlock,
        workflow.reporter,
    )

    assert rt_optimizer.has_converged is False
    assert rt_optimizer.parameter_name == "rt_error"

    workflow._fdr_manager._current_version += 1
    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(rt_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert rt_optimizer.has_converged is True
    assert (
        rt_optimizer.history_df.precursor_proportion_detected
        == pd.Series([1000 / 2000, 1001 / 2000, 1002 / 2000])
    ).all()
    assert (
        workflow.optimization_manager.rt_error
        == rt_optimizer.history_df.parameter[
            rt_optimizer.history_df.precursor_proportion_detected.idxmax()
        ]
    )
    assert workflow.optimization_manager.classifier_version == 2


def test_automatic_ms1_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    ms1_optimizer = AutomaticMS1Optimizer(
        100,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow._optimization_handler._optlock,
        workflow.reporter,
    )

    assert ms1_optimizer.has_converged is False
    assert ms1_optimizer.parameter_name == "ms1_error"

    workflow._fdr_manager._current_version += 1
    ms1_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(ms1_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    ms1_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    ms1_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert ms1_optimizer.has_converged is True
    assert (
        workflow.optimization_manager.ms1_error
        == ms1_optimizer.history_df.parameter[
            ms1_optimizer.history_df.mean_isotope_intensity_correlation.idxmax()
        ]
    )
    assert workflow.optimization_manager.classifier_version == 0


def test_automatic_mobility_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    mobility_optimizer = AutomaticMobilityOptimizer(
        100,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow._optimization_handler._optlock,
        workflow.reporter,
    )

    assert mobility_optimizer.has_converged is False
    assert mobility_optimizer.parameter_name == "mobility_error"

    workflow._fdr_manager._current_version += 1
    mobility_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(mobility_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1
    mobility_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow._fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    mobility_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert mobility_optimizer.has_converged is True
    assert (
        mobility_optimizer.history_df.precursor_proportion_detected
        == pd.Series([1000 / 2000, 1001 / 2000, 1002 / 2000])
    ).all()
    assert (
        workflow.optimization_manager.mobility_error
        == mobility_optimizer.history_df.parameter[
            mobility_optimizer.history_df.precursor_proportion_detected.idxmax()
        ]
    )
    assert workflow.optimization_manager.classifier_version == 2


def test_targeted_ms2_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    optimizer = TargetedMS2Optimizer(
        100,
        7,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow.reporter,
    )

    assert optimizer.parameter_name == "ms2_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow._fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True
    assert workflow.optimization_manager.ms2_error == optimizer.target_parameter


def test_targeted_rt_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    optimizer = TargetedRTOptimizer(
        100,
        7,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow.reporter,
    )

    assert optimizer.parameter_name == "rt_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow._fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True
    assert workflow.optimization_manager.rt_error == optimizer.target_parameter


def test_targeted_ms1_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    optimizer = TargetedMS1Optimizer(
        100,
        7,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow.reporter,
    )

    assert optimizer.parameter_name == "ms1_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow._fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True
    assert workflow.optimization_manager.ms1_error == optimizer.target_parameter


def test_targeted_mobility_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    optimizer = TargetedMobilityOptimizer(
        100,
        7,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow.reporter,
    )

    assert optimizer.parameter_name == "mobility_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow._fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True

    assert workflow.optimization_manager.mobility_error == optimizer.target_parameter


def test_configurability():
    workflow = create_workflow_instance()
    workflow.config["optimization"].update(
        {
            "order_of_optimization": [
                ["rt_error"],
                ["ms1_error", "ms2_error"],
                ["mobility_error"],
            ],
            "rt_error": {
                "automatic_update_percentile_range": 0.99,
                "automatic_update_factor": 1.3,
                "try_narrower_values": True,
                "maximal_decrease": 0.4,
                "favour_narrower_optimum": True,
                "maximum_decrease_from_maximum": 0.3,
            },
            "ms2_error": {
                "automatic_update_percentile_range": 0.80,
                "targeted_update_percentile_range": 0.995,
                "targeted_update_factor": 1.2,
                "favour_narrower_optimum": False,
            },
        }
    )
    workflow.config["search"].update(
        {
            "target_rt_tolerance": -1,
        }
    )

    ordered_optimizers = workflow._optimization_handler._get_ordered_optimizers()

    assert len(ordered_optimizers) == 3

    assert ordered_optimizers[0][0].parameter_name == "rt_error"
    assert isinstance(ordered_optimizers[0][0], AutomaticRTOptimizer)
    assert ordered_optimizers[0][0].update_percentile_range == 0.99
    assert ordered_optimizers[0][0].update_factor == 1.3

    assert ordered_optimizers[1][0].parameter_name == "ms1_error"
    assert ordered_optimizers[1][0].update_percentile_range == 0.95
    assert isinstance(ordered_optimizers[1][0], TargetedMS1Optimizer)

    assert ordered_optimizers[1][1].parameter_name == "ms2_error"
    assert isinstance(ordered_optimizers[1][1], TargetedMS2Optimizer)
    assert ordered_optimizers[1][1].update_percentile_range == 0.995
    assert ordered_optimizers[1][1].update_factor == 1.2

    assert ordered_optimizers[2][0].parameter_name == "mobility_error"


def test_optimizer_skipping():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    rt_optimizer = AutomaticRTOptimizer(
        100,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow._optimization_handler._optlock,
        workflow.reporter,
    )

    assert rt_optimizer.has_converged is False
    assert rt_optimizer.parameter_name == "rt_error"

    workflow._fdr_manager._current_version += 1
    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(rt_optimizer.history_df) == 1

    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert rt_optimizer.has_converged is False

    rt_optimizer.skip()

    rt_optimizer.skip()

    assert len(rt_optimizer.history_df) == 2
    assert rt_optimizer.has_converged is True

    rt_optimizer = TargetedRTOptimizer(
        100,
        10,
        workflow.config,
        workflow.optimization_manager,
        workflow.calibration_manager,
        workflow._fdr_manager,
        workflow.reporter,
    )

    workflow._fdr_manager._current_version += 1
    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    rt_optimizer.skip()

    rt_optimizer.skip()

    assert rt_optimizer.has_converged is False

    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert rt_optimizer.has_converged is True

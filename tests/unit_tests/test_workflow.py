import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LinearRegression

from alphadia.calibration.models import LOESSRegression
from alphadia.calibration.property import Calibration
from alphadia.fdrexperimental import BinaryClassifierLegacyNewBatching
from alphadia.workflow import base, manager


def test_base_manager():
    my_base_manager = manager.BaseManager()
    assert my_base_manager.path is None
    assert my_base_manager.is_loaded_from_file is False
    assert my_base_manager.is_fitted is False


def test_base_manager_save():
    tmp_path = os.path.join(tempfile.gettempdir(), "my_base_manager.pkl")

    my_base_manager = manager.BaseManager(path=tmp_path)
    my_base_manager.save()
    assert os.path.exists(my_base_manager.path)
    os.remove(my_base_manager.path)


def test_base_manager_load():
    tmp_path = os.path.join(tempfile.gettempdir(), "my_base_manager.pkl")

    my_base_manager = manager.BaseManager(path=tmp_path)
    my_base_manager.save()

    my_base_manager_loaded = manager.BaseManager(path=tmp_path, load_from_file=True)
    assert my_base_manager_loaded.path == my_base_manager.path
    assert my_base_manager_loaded.is_loaded_from_file is True
    assert my_base_manager_loaded.is_fitted is False

    os.remove(my_base_manager.path)


TEST_CONFIG = [
    {
        "name": "precursor",
        "estimators": [
            {
                "name": "mz",
                "model": "LinearRegression",
                "input_columns": ["mz_library"],
                "target_columns": ["mz_observed"],
                "output_columns": ["mz_calibrated"],
                "transform_deviation": 1e6,
            },
            {
                "name": "rt",
                "model": "LOESSRegression",
                "model_args": {"n_kernels": 2},
                "input_columns": ["rt_library"],
                "target_columns": ["rt_observed"],
                "output_columns": ["rt_calibrated"],
                "transform_deviation": None,
            },
        ],
    },
    {
        "name": "fragment",
        "estimators": [
            {
                "name": "mz",
                "model": "LinearRegression",
                "input_columns": ["mz_library"],
                "target_columns": ["mz_observed"],
                "output_columns": ["mz_calibrated"],
                "transform_deviation": 1e6,
            }
        ],
    },
]


def test_calibration_manager_init():
    # initialize the calibration manager
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=False
    )

    assert calibration_manager.path == temp_path
    assert calibration_manager.is_loaded_from_file is False
    assert calibration_manager.is_fitted is False

    assert len(calibration_manager.estimator_groups) == 2
    assert len(calibration_manager.get_estimator_names("precursor")) == 2
    assert len(calibration_manager.get_estimator_names("fragment")) == 1

    assert calibration_manager.get_estimator("precursor", "mz").name == "mz"
    assert calibration_manager.get_estimator("precursor", "rt").name == "rt"
    assert calibration_manager.get_estimator("fragment", "mz").name == "mz"

    assert isinstance(calibration_manager.get_estimator("precursor", "mz"), Calibration)
    assert isinstance(calibration_manager.get_estimator("precursor", "rt"), Calibration)
    assert isinstance(calibration_manager.get_estimator("fragment", "mz"), Calibration)

    assert isinstance(
        calibration_manager.get_estimator("precursor", "mz").function, LinearRegression
    )
    assert isinstance(
        calibration_manager.get_estimator("precursor", "rt").function, LOESSRegression
    )
    assert isinstance(
        calibration_manager.get_estimator("fragment", "mz").function, LinearRegression
    )


def calibration_testdata():
    # create some test data and make sure estimation works
    mz_library = np.linspace(100, 1000, 1000)
    mz_observed = (
        mz_library + np.random.normal(0, 0.001, 1000) + mz_library * 0.00001 + 0.005
    )

    rt_library = np.linspace(0, 100, 1000)
    rt_observed = (
        rt_library + np.random.normal(0, 0.5, 1000) + np.sin(rt_library * 0.05)
    )

    return pd.DataFrame(
        {
            "mz_library": mz_library,
            "mz_observed": mz_observed,
            "rt_library": rt_library,
            "rt_observed": rt_observed,
        }
    ).copy()


def test_calibration_manager_fit_predict():
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=False
    )

    test_df = calibration_testdata()

    # fit only the precursor mz calibration
    calibration_manager.fit_predict(test_df, "precursor", plot=False)

    assert "mz_calibrated" in test_df.columns
    assert "rt_calibrated" in test_df.columns
    # will be false as the the fragment mz calibration is not fitted

    assert calibration_manager.is_fitted is False
    assert calibration_manager.is_loaded_from_file is False

    # fit the fragment mz calibration
    calibration_manager.fit(test_df, "fragment", plot=False)

    assert calibration_manager.is_fitted is True


def test_calibration_manager_save_load():
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=False
    )

    test_df = calibration_testdata()
    calibration_manager.fit(test_df, "precursor", plot=False)
    calibration_manager.fit(test_df, "fragment", plot=False)

    assert calibration_manager.is_fitted is True
    assert calibration_manager.is_loaded_from_file is False

    calibration_manager.save()

    calibration_manager_loaded = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=True
    )
    assert calibration_manager_loaded.is_fitted is True
    assert calibration_manager_loaded.is_loaded_from_file is True

    calibration_manager_loaded.predict(test_df, "precursor")

    assert "mz_calibrated" in test_df.columns
    assert "rt_calibrated" in test_df.columns

    os.remove(temp_path)


OPTIMIZATION_TEST_DATA = {
    "fwhm_cycles": 5,
    "fwhm_mobility": 0.01,
}


def test_optimization_manager():
    optimization_manager = manager.OptimizationManager(OPTIMIZATION_TEST_DATA)

    assert optimization_manager.fwhm_cycles == 5
    assert optimization_manager.fwhm_mobility == 0.01

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is False


def test_optimization_manager_save_load():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")

    optimization_manager = manager.OptimizationManager(
        OPTIMIZATION_TEST_DATA, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is False

    optimization_manager.save()

    optimization_manager_loaded = manager.OptimizationManager(
        OPTIMIZATION_TEST_DATA, path=temp_path, load_from_file=True
    )

    assert optimization_manager_loaded.is_loaded_from_file is True
    assert optimization_manager_loaded.is_fitted is False

    os.remove(temp_path)


def test_optimization_manager_fit():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")
    optimization_manager = manager.OptimizationManager(
        OPTIMIZATION_TEST_DATA, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is False

    optimization_manager.fit({"fwhm_cycles": 10, "fwhm_mobility": 0.02})

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is True

    assert optimization_manager.fwhm_cycles == 10
    assert optimization_manager.fwhm_mobility == 0.02

    os.remove(temp_path)


@pytest.mark.slow
def test_workflow_base():
    if pytest.test_data is None:
        pytest.skip("No test data found")
        return

    for _, file_list in pytest.test_data.items():
        for file in file_list:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "misc", "config", "default.yaml"
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["output"] = tempfile.gettempdir()

            workflow_name = Path(file).stem

            my_workflow = base.WorkflowBase(
                workflow_name,
                config,
            )
            my_workflow.load(file, pd.DataFrame({}))

            assert my_workflow.config["output"] == config["output"]
            assert my_workflow.instance_name == workflow_name
            assert my_workflow.parent_path == os.path.join(
                config["output"], base.TEMP_FOLDER
            )
            assert my_workflow.path == os.path.join(
                my_workflow.parent_path, workflow_name
            )

            assert os.path.exists(my_workflow.path)

            # assert isinstance(my_workflow.dia_data, bruker.TimsTOFTranspose) or isinstance(my_workflow.dia_data, thermo.Thermo)
            assert isinstance(
                my_workflow.calibration_manager, manager.CalibrationManager
            )
            assert isinstance(
                my_workflow.optimization_manager, manager.OptimizationManager
            )

            # os.rmdir(os.path.join(my_workflow.path, my_workflow.FIGURE_PATH))
            # os.rmdir(os.path.join(my_workflow.path))
            shutil.rmtree(os.path.join(my_workflow.parent_path))


FDR_TEST_BASE_CLASSIFIER = BinaryClassifierLegacyNewBatching(
    test_size=0.001, batch_size=50, learning_rate=0.001, epochs=1
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
    fdr_manager = manager.FDRManager(FDR_TEST_FEATURES, FDR_TEST_BASE_CLASSIFIER)

    assert fdr_manager.is_loaded_from_file is False
    assert fdr_manager.is_fitted is False

    assert fdr_manager.feature_columns == FDR_TEST_FEATURES
    assert fdr_manager.classifier_base == FDR_TEST_BASE_CLASSIFIER


@pytest.mark.slow
def test_fdr_manager_fit_predict():
    fdr_manager = manager.FDRManager(FDR_TEST_FEATURES, FDR_TEST_BASE_CLASSIFIER)
    test_features_df = fdr_testdata(FDR_TEST_FEATURES)

    assert len(fdr_manager.classifier_store) == 1

    fdr_manager.fit_predict(
        test_features_df,
        decoy_strategy="precursor",
        competetive=False,
        df_fragments=None,
        dia_cycle=None,
        # neptune_run=self.neptune
    )

    desired_version_info = FDR_TEST_FEATURES

    assert manager.column_hash(desired_version_info) in fdr_manager.classifier_store
    assert len(fdr_manager.classifier_store) == 2

    test_epoch = 0
    test_param_info = ["ms1", "mobility"]

    fdr_manager.fit_predict(
        test_features_df,
        decoy_strategy="precursor",
        competetive=False,
        df_fragments=None,
        dia_cycle=None,
        epoch_input=test_epoch,
        param_info_ouptut=test_param_info,
    )

    # Neither input nor output were fully specified so the function should default to using just the feature columns and hence save to the same classifier as before.
    assert len(fdr_manager.classifier_store) == 2

    fdr_manager.fit_predict(
        test_features_df,
        decoy_strategy="precursor",
        competetive=False,
        df_fragments=None,
        dia_cycle=None,
        epoch_output=test_epoch,
        param_info_ouptut=test_param_info,
    )

    desired_version_info = np.append(
        np.concatenate([FDR_TEST_FEATURES, test_param_info]), [str(test_epoch)]
    )

    assert len(fdr_manager.classifier_store) == 3
    assert manager.column_hash(desired_version_info) in fdr_manager.classifier_store

import tempfile
import os
import shutil
import pickle
import numpy as np
import pandas as pd
import yaml
import pytest
from pathlib import Path

from alphadia import calibration
from sklearn.linear_model import LinearRegression
from alphabase.statistics.regression import LOESSRegression
from alphadia.workflow import manager, base
from alphadia.data import bruker, thermo


def test_base_manager():
    my_base_manager = manager.BaseManager()
    assert my_base_manager.path == None
    assert my_base_manager.is_loaded_from_file == False
    assert my_base_manager.is_fitted == False


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
    assert my_base_manager_loaded.is_loaded_from_file == True
    assert my_base_manager_loaded.is_fitted == False

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
    temp_path = path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=False
    )

    assert calibration_manager.path == temp_path
    assert calibration_manager.is_loaded_from_file == False
    assert calibration_manager.is_fitted == False

    assert len(calibration_manager.estimator_groups) == 2
    assert len(calibration_manager.get_estimator_names("precursor")) == 2
    assert len(calibration_manager.get_estimator_names("fragment")) == 1

    assert calibration_manager.get_estimator("precursor", "mz").name == "mz"
    assert calibration_manager.get_estimator("precursor", "rt").name == "rt"
    assert calibration_manager.get_estimator("fragment", "mz").name == "mz"

    assert isinstance(
        calibration_manager.get_estimator("precursor", "mz"), calibration.Calibration
    )
    assert isinstance(
        calibration_manager.get_estimator("precursor", "rt"), calibration.Calibration
    )
    assert isinstance(
        calibration_manager.get_estimator("fragment", "mz"), calibration.Calibration
    )

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
    temp_path = path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=False
    )

    test_df = calibration_testdata()

    # fit only the precursor mz calibration
    calibration_manager.fit_predict(test_df, "precursor", plot=False)

    assert "mz_calibrated" in test_df.columns
    assert "rt_calibrated" in test_df.columns
    # will be false as the the fragment mz calibration is not fitted

    assert calibration_manager.is_fitted == False
    assert calibration_manager.is_loaded_from_file == False

    # fit the fragment mz calibration
    calibration_manager.fit(test_df, "fragment", plot=False)

    assert calibration_manager.is_fitted == True


def test_calibration_manager_save_load():
    temp_path = os.path.join(tempfile.tempdir, "calibration_manager.pkl")
    calibration_manager = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=False
    )

    test_df = calibration_testdata()
    calibration_manager.fit(test_df, "precursor", plot=False)
    calibration_manager.fit(test_df, "fragment", plot=False)

    assert calibration_manager.is_fitted == True
    assert calibration_manager.is_loaded_from_file == False

    calibration_manager.save()

    calibration_manager_loaded = manager.CalibrationManager(
        TEST_CONFIG, path=temp_path, load_from_file=True
    )
    assert calibration_manager_loaded.is_fitted == True
    assert calibration_manager_loaded.is_loaded_from_file == True

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

    assert optimization_manager.is_loaded_from_file == False
    assert optimization_manager.is_fitted == False


def test_optimization_manager_save_load():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")

    optimization_manager = manager.OptimizationManager(
        OPTIMIZATION_TEST_DATA, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file == False
    assert optimization_manager.is_fitted == False

    optimization_manager.save()

    optimization_manager_loaded = manager.OptimizationManager(
        OPTIMIZATION_TEST_DATA, path=temp_path, load_from_file=True
    )

    assert optimization_manager_loaded.is_loaded_from_file == True
    assert optimization_manager_loaded.is_fitted == False

    os.remove(temp_path)


def test_optimization_manager_fit():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")
    optimization_manager = manager.OptimizationManager(
        OPTIMIZATION_TEST_DATA, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file == False
    assert optimization_manager.is_fitted == False

    optimization_manager.fit({"fwhm_cycles": 10, "fwhm_mobility": 0.02})

    assert optimization_manager.is_loaded_from_file == False
    assert optimization_manager.is_fitted == True

    assert optimization_manager.fwhm_cycles == 10
    assert optimization_manager.fwhm_mobility == 0.02

    os.remove(temp_path)


@pytest.mark.slow
def test_workflow_base():
    if pytest.test_data is None:
        pytest.skip("No test data found")
        return

    for name, file_list in pytest.test_data.items():
        for file in file_list:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "misc", "config", "default.yaml"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            config["output"] = tempfile.gettempdir()

            workflow_name = Path(file).stem

            my_workflow = base.WorkflowBase(
<<<<<<< HEAD
                workflow_name,
                config,
=======
                workflow_name, config, file, pd.DataFrame({})
>>>>>>> main
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

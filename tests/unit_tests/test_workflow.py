import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from alphabase.spectral_library.flat import SpecLibFlat
from sklearn.linear_model import LinearRegression

from alphadia.calibration.models import LOESSRegression
from alphadia.calibration.property import Calibration
from alphadia.fdrexperimental import BinaryClassifierLegacyNewBatching
from alphadia.workflow import base, manager, optimization, peptidecentric, reporting
from alphadia.workflow.config import Config


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


TEST_OPTIMIZATION_CONFIG = {
    "search_initial": {
        "initial_ms1_tolerance": 4,
        "initial_ms2_tolerance": 7,
        "initial_rt_tolerance": 200,
        "initial_mobility_tolerance": 0.04,
        "initial_num_candidates": 1,
    },
    "optimization_manager": {
        "fwhm_rt": 5,
        "fwhm_mobility": 0.01,
        "score_cutoff": 50,
    },
}


def test_optimization_manager():
    optimization_manager = manager.OptimizationManager(TEST_OPTIMIZATION_CONFIG)

    assert optimization_manager.fwhm_rt == 5
    assert optimization_manager.fwhm_mobility == 0.01

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is False


def test_optimization_manager_rt_proportion():
    TEST_OPTIMIZATION_CONFIG_PROPORTION = deepcopy(TEST_OPTIMIZATION_CONFIG)
    TEST_OPTIMIZATION_CONFIG_PROPORTION["search_initial"]["initial_rt_tolerance"] = 0.5
    optimization_manager = manager.OptimizationManager(
        TEST_OPTIMIZATION_CONFIG_PROPORTION, 1200
    )

    assert optimization_manager.fwhm_rt == 5
    assert optimization_manager.fwhm_mobility == 0.01
    optimization_manager.rt_error = 600

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is False


def test_optimization_manager_save_load():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")

    optimization_manager = manager.OptimizationManager(
        TEST_OPTIMIZATION_CONFIG, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is False

    optimization_manager.save()

    optimization_manager_loaded = manager.OptimizationManager(
        TEST_OPTIMIZATION_CONFIG, path=temp_path, load_from_file=True
    )

    assert optimization_manager_loaded.is_loaded_from_file is True
    assert optimization_manager_loaded.is_fitted is False

    os.remove(temp_path)


def test_optimization_manager_fit():
    temp_path = os.path.join(tempfile.tempdir, "optimization_manager.pkl")
    optimization_manager = manager.OptimizationManager(
        TEST_OPTIMIZATION_CONFIG, path=temp_path, load_from_file=False
    )

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is False

    optimization_manager.fit({"fwhm_cycles": 10, "fwhm_mobility": 0.02})

    assert optimization_manager.is_loaded_from_file is False
    assert optimization_manager.is_fitted is True

    assert optimization_manager.fwhm_cycles == 10
    assert optimization_manager.fwhm_mobility == 0.02

    optimization_manager.save()
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
    fdr_manager = manager.FDRManager(FDR_TEST_FEATURES, FDR_TEST_BASE_CLASSIFIER)

    assert fdr_manager.is_loaded_from_file is False
    assert fdr_manager.is_fitted is False

    assert fdr_manager.feature_columns == FDR_TEST_FEATURES
    assert fdr_manager.classifier_base == FDR_TEST_BASE_CLASSIFIER


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
    )

    assert len(fdr_manager.classifier_store) == 2
    assert fdr_manager.current_version == 0
    assert manager.column_hash(FDR_TEST_FEATURES) in fdr_manager.classifier_store

    fdr_manager.fit_predict(
        test_features_df,
        decoy_strategy="precursor",
        competetive=False,
        df_fragments=None,
        dia_cycle=None,
        version=0,
    )

    assert fdr_manager.current_version == 1
    assert fdr_manager.get_classifier(FDR_TEST_FEATURES, 0).fitted is True

    fdr_manager.save_classifier_store(tempfile.tempdir)

    fdr_manager_new = manager.FDRManager(FDR_TEST_FEATURES, FDR_TEST_BASE_CLASSIFIER)
    fdr_manager_new.load_classifier_store(tempfile.tempdir)

    temp_path = os.path.join(
        tempfile.tempdir, f"{manager.column_hash(FDR_TEST_FEATURES)}.pth"
    )

    assert os.path.exists(temp_path)
    assert fdr_manager_new.get_classifier(FDR_TEST_FEATURES).fitted is True

    os.remove(temp_path)


def create_workflow_instance():
    config_base_path = os.path.join(
        Path(__file__).parents[2], "alphadia", "constants", "default.yaml"
    )

    config = Config()
    config.from_yaml(config_base_path)
    config["output"] = tempfile.mkdtemp()
    workflow = peptidecentric.PeptideCentricWorkflow(
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
    workflow._calibration_manager = manager.CalibrationManager(
        workflow.config["calibration_manager"],
        path=os.path.join(workflow.path, workflow.CALIBRATION_MANAGER_PATH),
        load_from_file=workflow.config["general"]["reuse_calibration"],
        reporter=workflow.reporter,
    )

    workflow._optimization_manager = manager.OptimizationManager(
        TEST_OPTIMIZATION_CONFIG,
        path=os.path.join(workflow.path, workflow.OPTIMIZATION_MANAGER_PATH),
        load_from_file=workflow.config["general"]["reuse_calibration"],
        figure_path=os.path.join(workflow.path, workflow.FIGURE_PATH),
        reporter=workflow.reporter,
    )

    workflow.init_fdr_manager()

    class MockOptlock:
        total_elution_groups = 2000
        batch_idx = 1

    workflow.optlock = MockOptlock()

    class MockDIAData:
        has_mobility = True
        has_ms1 = True

    workflow._dia_data = MockDIAData()

    return workflow


def test_automatic_ms2_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df2, "fragment", plot=False)

    ms2_optimizer = optimization.AutomaticMS2Optimizer(
        100,
        workflow,
    )

    assert ms2_optimizer.has_converged is False
    assert ms2_optimizer.parameter_name == "ms2_error"

    workflow.fdr_manager._current_version += 1
    ms2_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(ms2_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    ms2_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

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


def test_automatic_rt_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    rt_optimizer = optimization.AutomaticRTOptimizer(
        100,
        workflow,
    )

    assert rt_optimizer.has_converged is False
    assert rt_optimizer.parameter_name == "rt_error"

    workflow.fdr_manager._current_version += 1
    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(rt_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

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

    ms1_optimizer = optimization.AutomaticMS1Optimizer(
        100,
        workflow,
    )

    assert ms1_optimizer.has_converged is False
    assert ms1_optimizer.parameter_name == "ms1_error"

    workflow.fdr_manager._current_version += 1
    ms1_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(ms1_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1

    ms1_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

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

    mobility_optimizer = optimization.AutomaticMobilityOptimizer(
        100,
        workflow,
    )

    assert mobility_optimizer.has_converged is False
    assert mobility_optimizer.parameter_name == "mobility_error"

    workflow.fdr_manager._current_version += 1
    mobility_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(mobility_optimizer.history_df) == 1

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

    assert workflow.optimization_manager.classifier_version == -1
    mobility_optimizer.step(calibration_test_df1, calibration_test_df2)

    calibration_test_df1 = pd.concat(
        [calibration_test_df1, pd.DataFrame(calibration_test_df1.loc[0]).T],
        ignore_index=True,
    )
    workflow.fdr_manager._current_version += 1

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

    optimizer = optimization.TargetedMS2Optimizer(
        100,
        7,
        workflow,
    )

    assert optimizer.parameter_name == "ms2_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow.fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True
    assert workflow.optimization_manager.ms2_error == optimizer.target_parameter


def test_targeted_rt_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    optimizer = optimization.TargetedRTOptimizer(
        100,
        7,
        workflow,
    )

    assert optimizer.parameter_name == "rt_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow.fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True
    assert workflow.optimization_manager.rt_error == optimizer.target_parameter


def test_targeted_ms1_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    optimizer = optimization.TargetedMS1Optimizer(
        100,
        7,
        workflow,
    )

    assert optimizer.parameter_name == "ms1_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow.fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True
    assert workflow.optimization_manager.ms1_error == optimizer.target_parameter


def test_targeted_mobility_optimizer():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    optimizer = optimization.TargetedMobilityOptimizer(
        100,
        7,
        workflow,
    )

    assert optimizer.parameter_name == "mobility_error"

    for current_step in range(workflow.config["calibration"]["min_steps"]):
        assert optimizer.has_converged is False

        workflow.fdr_manager._current_version += 1
        optimizer.step(calibration_test_df1, calibration_test_df2)

        assert workflow.optimization_manager.classifier_version == current_step

    assert optimizer.has_converged is True

    assert workflow.optimization_manager.mobility_error == optimizer.target_parameter


def create_test_library():
    lib = SpecLibFlat()
    precursor_idx = np.arange(100000)
    elution_group_idx = np.concatenate(
        [np.full(2, i, dtype=int) for i in np.arange(len(precursor_idx) / 2)]
    )
    flat_frag_start_idx = precursor_idx * 10
    flat_frag_stop_idx = (precursor_idx + 1) * 10
    lib._precursor_df = pd.DataFrame(
        {
            "elution_group_idx": elution_group_idx,
            "precursor_idx": precursor_idx,
            "flat_frag_start_idx": flat_frag_start_idx,
            "flat_frag_stop_idx": flat_frag_stop_idx,
        }
    )

    lib._fragment_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(0, flat_frag_stop_idx[-1]),
        }
    )

    return lib


def create_test_library_for_indexing():
    lib = SpecLibFlat()
    precursor_idx = np.arange(1000)
    elution_group_idx = np.concatenate(
        [np.full(2, i, dtype=int) for i in np.arange(len(precursor_idx) / 2)]
    )
    flat_frag_start_idx = precursor_idx**2
    flat_frag_stop_idx = (precursor_idx + 1) ** 2
    lib._precursor_df = pd.DataFrame(
        {
            "elution_group_idx": elution_group_idx,
            "precursor_idx": precursor_idx,
            "flat_frag_start_idx": flat_frag_start_idx,
            "flat_frag_stop_idx": flat_frag_stop_idx,
        }
    )

    lib._fragment_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(0, flat_frag_stop_idx[-1]),
        }
    )

    return lib


TEST_OPTLOCK_CONFIG = {
    "calibration": {
        "batch_size": 8000,
        "optimization_lock_target": 200,
    }
}


def test_optlock():
    library = create_test_library()
    optlock = optimization.OptimizationLock(library, TEST_OPTLOCK_CONFIG)

    assert optlock.start_idx == optlock.batch_plan[0][0]

    feature_df = pd.DataFrame({"elution_group_idx": np.arange(0, 1000)})
    fragment_df = pd.DataFrame({"elution_group_idx": np.arange(0, 10000)})

    optlock.update_with_extraction(feature_df, fragment_df)

    assert optlock.total_elution_groups == 1000
    precursor_df = pd.DataFrame(
        {"qval": np.concatenate([np.full(100, 0.005), np.full(1000, 0.05)])}
    )
    optlock.update_with_fdr(precursor_df)

    assert optlock.has_target_num_precursors is False
    assert optlock.previously_calibrated is False
    optlock.update()

    assert optlock.start_idx == optlock.batch_plan[1][0]

    feature_df = pd.DataFrame({"elution_group_idx": np.arange(1000, 2000)})
    fragment_df = pd.DataFrame({"elution_group_idx": np.arange(10000, 20000)})

    optlock.update_with_extraction(feature_df, fragment_df)

    assert optlock.total_elution_groups == 2000

    precursor_df = pd.DataFrame(
        {"qval": np.concatenate([np.full(200, 0.005), np.full(1000, 0.05)])}
    )

    optlock.update_with_fdr(precursor_df)

    assert optlock.has_target_num_precursors is True
    assert optlock.previously_calibrated is False

    optlock.update()

    assert optlock.start_idx == 0

    assert optlock.total_elution_groups == 2000


def test_optlock_batch_idx():
    library = create_test_library()
    optlock = optimization.OptimizationLock(library, TEST_OPTLOCK_CONFIG)

    optlock.batch_plan = [[0, 100], [100, 2000], [2000, 8000]]

    assert optlock.start_idx == 0

    optlock.update()
    assert optlock.start_idx == 100

    optlock.update()
    assert optlock.start_idx == 2000

    precursor_df = pd.DataFrame({"qval": np.full(4500, 0.005)})

    optlock.update_with_fdr(precursor_df)

    optlock.has_target_num_precursors = True
    optlock.update()

    assert optlock.start_idx == 0
    assert optlock.stop_idx == 2000


def test_optlock_reindex():
    library = create_test_library_for_indexing()
    optlock = optimization.OptimizationLock(library, TEST_OPTLOCK_CONFIG)
    optlock.batch_plan = [[0, 100], [100, 200]]
    optlock.set_batch_dfs(
        optlock.elution_group_order[optlock.start_idx : optlock.stop_idx]
    )

    assert (
        (
            optlock.batch_library._precursor_df["flat_frag_stop_idx"].iloc[100]
            - optlock.batch_library._precursor_df["flat_frag_start_idx"].iloc[100]
        )
        == (
            (optlock.batch_library._precursor_df["precursor_idx"].iloc[100] + 1) ** 2
            - optlock.batch_library._precursor_df["precursor_idx"].iloc[100] ** 2
        )
    )  # Since each precursor was set (based on its original ID) to have a number of fragments equal to its original ID squared, the difference between the start and stop index should be equal to the original ID squared (even if the start and stop index have been changed to different values)
    assert (
        optlock.batch_library._fragment_df.iloc[
            optlock.batch_library._precursor_df.iloc[50]["flat_frag_start_idx"]
        ]["precursor_idx"]
        == optlock.batch_library._precursor_df.iloc[50]["precursor_idx"] ** 2
    )  # The original start index of any precursor should be equal to the square of the its original ID


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
                "favour_narrower_parameter": True,
                "maximal_decrease": 0.8,
                "minimum_proportion_of_maximum": 0.95,
            },
            "ms2_error": {
                "automatic_update_percentile_range": 0.80,
                "targeted_update_percentile_range": 0.995,
                "targeted_update_factor": 1.2,
                "favour_narrower_parameter": False,
            },
        }
    )
    workflow.config["search"].update(
        {
            "target_rt_tolerance": -1,
        }
    )

    ordered_optimizers = workflow.get_ordered_optimizers()

    assert len(ordered_optimizers) == 3

    assert ordered_optimizers[0][0].parameter_name == "rt_error"
    assert isinstance(ordered_optimizers[0][0], optimization.AutomaticRTOptimizer)
    assert ordered_optimizers[0][0].update_percentile_range == 0.99
    assert ordered_optimizers[0][0].update_factor == 1.3

    assert ordered_optimizers[1][0].parameter_name == "ms1_error"
    assert ordered_optimizers[1][0].update_percentile_range == 0.95
    assert isinstance(ordered_optimizers[1][0], optimization.TargetedMS1Optimizer)

    assert ordered_optimizers[1][1].parameter_name == "ms2_error"
    assert isinstance(ordered_optimizers[1][1], optimization.TargetedMS2Optimizer)
    assert ordered_optimizers[1][1].update_percentile_range == 0.995
    assert ordered_optimizers[1][1].update_factor == 1.2

    assert ordered_optimizers[2][0].parameter_name == "mobility_error"


def test_optimizer_skipping():
    workflow = create_workflow_instance()

    calibration_test_df1 = calibration_testdata()
    calibration_test_df2 = calibration_testdata()

    workflow.calibration_manager.fit(calibration_test_df1, "precursor", plot=False)

    rt_optimizer = optimization.AutomaticRTOptimizer(
        100,
        workflow,
    )

    assert rt_optimizer.has_converged is False
    assert rt_optimizer.parameter_name == "rt_error"

    workflow.fdr_manager._current_version += 1
    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert len(rt_optimizer.history_df) == 1

    rt_optimizer.skip()

    rt_optimizer.skip()

    assert len(rt_optimizer.history_df) == 1
    assert rt_optimizer.has_converged is True

    rt_optimizer = optimization.TargetedRTOptimizer(
        100,
        10,
        workflow,
    )

    workflow.fdr_manager._current_version += 1
    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    rt_optimizer.skip()

    rt_optimizer.skip()

    assert rt_optimizer.has_converged is False

    rt_optimizer.step(calibration_test_df1, calibration_test_df2)

    assert rt_optimizer.has_converged is True

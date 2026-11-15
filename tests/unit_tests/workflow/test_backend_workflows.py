"""Unit tests for the backend-specific parts of the Python and Rust workflows."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphadia.constants.keys import SearchTolerance
from alphadia.exceptions import GenericUserError
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.optimization_handler import CalibrationResult
from alphadia.workflow.peptidecentric.python.extraction_handler import (
    PythonExtractionHandler,
)
from alphadia.workflow.peptidecentric.python.optimization_handler import (
    PythonOptimizationHandler,
)
from alphadia.workflow.peptidecentric.python.utils import feature_columns
from alphadia.workflow.peptidecentric.python.workflow import (
    PythonPeptideCentricWorkflow,
)
from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    RustExtractionHandler,
)
from alphadia.workflow.peptidecentric.rust.mapper import (
    INTERFERENCE_FEATURE_GROUPS,
    get_classifier_feature_columns,
)
from alphadia.workflow.peptidecentric.rust.optimization_handler import (
    RustOptimizationHandler,
)
from alphadia.workflow.peptidecentric.rust.workflow import RustPeptideCentricWorkflow

THREAD_COUNT = 3

OPTIMIZATION_CONFIG = {
    "search_initial": {
        "ms1_tolerance": 4,
        "ms2_tolerance": 7,
        "rt_tolerance": 200,
        "mobility_tolerance": 0.04,
        "num_candidates": 1,
    },
    "optimization_manager": {"fwhm_rt": 5, "fwhm_mobility": 0.01, "score_cutoff": 50},
}


def _create_workflow(workflow_class, tmp_path):
    config = {
        "general": {
            "save_figures": False,
            "reuse_calibration": False,
            "thread_count": THREAD_COUNT,
        },
        "output_directory": str(tmp_path),
        "search": {
            "target_num_candidates": 3,
            "top_k_fragments_selection": 12,
            "exclude_shared_ions": True,
            "quant_window": 3,
            "quant_all": True,
            "experimental_xic": False,
        },
    }
    workflow = workflow_class("raw1", config)
    workflow.reporter = MagicMock()
    workflow._optimization_manager = OptimizationManager(OPTIMIZATION_CONFIG)
    workflow._calibration_manager = MagicMock()
    workflow._dia_data = MagicMock(has_ms1=True, has_mobility=False)
    return workflow


@pytest.mark.parametrize(
    ("workflow_class", "expected_columns"),
    [
        (PythonPeptideCentricWorkflow, feature_columns),
        (RustPeptideCentricWorkflow, get_classifier_feature_columns()),
    ],
)
def test_classifier_feature_columns_match_backend(
    workflow_class, expected_columns, tmp_path
):
    # given
    workflow = _create_workflow(workflow_class, tmp_path)

    # when
    columns = workflow._classifier_feature_columns()

    # then
    assert columns == expected_columns


def test_rust_classifier_feature_columns_follow_the_interference_flags(tmp_path):
    # given: a rust workflow with the window share switched on, and a backend that scores both interference groups
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)
    workflow.config["search"]["window_signal_share_features"] = True
    interference = [c for cols in INTERFERENCE_FEATURE_GROUPS.values() for c in cols]

    # when
    with patch(
        "alphadia.workflow.peptidecentric.rust.mapper.get_feature_names",
        return_value=["correlation_mean", *interference],
    ):
        columns = workflow._classifier_feature_columns()

    # then: the window share enters the classifier, the fitted spread does not
    assert columns == [
        "correlation_mean",
        *INTERFERENCE_FEATURE_GROUPS["window_signal_share_features"],
    ]


@pytest.mark.parametrize(
    ("workflow_class", "extraction_handler_class"),
    [
        (PythonPeptideCentricWorkflow, PythonExtractionHandler),
        (RustPeptideCentricWorkflow, RustExtractionHandler),
    ],
)
@patch("alphadia.workflow.peptidecentric.rust.extraction_handler.speclib_to_rust")
def test_workflow_creates_extraction_handler_of_its_backend(
    _, workflow_class, extraction_handler_class, tmp_path
):
    # Given: a workflow of a backend and a spectral library
    workflow = _create_workflow(workflow_class, tmp_path)
    spectral_library = MagicMock(name="spectral_library")

    # When: the workflow creates an extraction handler for the library
    extraction_handler = workflow._create_extraction_handler(spectral_library)

    # Then: the handler is of the backend and works on the DIA data, the library and the current optimization context
    assert type(extraction_handler) is extraction_handler_class
    assert extraction_handler._dia_data is workflow.dia_data
    assert extraction_handler._spectral_library is spectral_library
    assert (
        extraction_handler._optimization_context
        == workflow.optimization_manager.get_optimization_context()
    )


@patch("alphadia.workflow.peptidecentric.optimization_handler.log_precursor_df")
@patch("alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock")
def test_process_batch_collects_extraction_before_fdr(mock_optlock_cls, _):
    """Test that a batch is passed to the optimization lock before FDR runs on everything collected so far."""
    # given
    candidates_df = pd.DataFrame({"precursor_idx": [1], "rank": [0]})
    extraction_handler = MagicMock()
    extraction_handler.select_candidates.return_value = candidates_df
    extraction_handler.score_candidates.return_value = pd.DataFrame(
        {"precursor_idx": [1], "rank": [0], "feature": [0.5]}
    )
    fragments_df = pd.DataFrame({"precursor_idx": [1]})
    extraction_handler.quantify_candidates.return_value = (
        pd.DataFrame({"precursor_idx": [1], "rank": [0], "intensity": [10.0]}),
        fragments_df,
    )
    precursor_fdr_df = pd.DataFrame({"precursor_idx": [1], "qval": [0.0]})
    extraction_handler.perform_fdr_and_filter_candidates.return_value = (
        None,
        precursor_fdr_df,
    )
    optlock = mock_optlock_cls.return_value
    optlock.features_df = pd.DataFrame({"precursor_idx": [1]})
    optlock.fragments_df = fragments_df
    calls = MagicMock()
    calls.attach_mock(optlock.update_with_extraction, "update_with_extraction")
    calls.attach_mock(
        extraction_handler.perform_fdr_and_filter_candidates, "perform_fdr"
    )
    create_extraction_handler = MagicMock(return_value=extraction_handler)
    handler = RustOptimizationHandler(
        {},
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        create_extraction_handler=create_extraction_handler,
    )

    # when
    search = handler._process_batch()

    # then
    create_extraction_handler.assert_called_once_with(optlock.batch_library)
    assert [c[0] for c in calls.mock_calls] == ["update_with_extraction", "perform_fdr"]
    precursors_with_features_df = optlock.update_with_extraction.call_args.args[0]
    assert list(precursors_with_features_df.columns) == [
        "precursor_idx",
        "rank",
        "intensity",
        "feature",
    ]
    assert extraction_handler.perform_fdr_and_filter_candidates.call_args.args[1] is (
        candidates_df
    )
    optlock.update_with_fdr.assert_called_once_with(precursor_fdr_df)
    pd.testing.assert_frame_equal(search.precursor_df, precursor_fdr_df)


def test_python_workflow_keeps_dia_data(tmp_path):
    # given
    workflow = _create_workflow(PythonPeptideCentricWorkflow, tmp_path)
    dia_data = MagicMock(name="dia_data")

    # when
    prepared = workflow._convert_dia_data(dia_data)

    # then
    assert prepared is dia_data


@patch("alphadia.workflow.peptidecentric.rust.workflow.set_thread_count")
@patch("alphadia.workflow.peptidecentric.rust.workflow.dia_data_to_rust")
def test_rust_workflow_converts_dia_data(
    mock_dia_data_to_rust, mock_set_thread_count, tmp_path
):
    # given
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)
    dia_data = MagicMock(
        name="dia_data", cycle=np.zeros((1, 4)), rt_values=np.arange(3)
    )
    mock_dia_data_to_rust.return_value = MagicMock(
        cycle=dia_data.cycle, rt_values=dia_data.rt_values
    )

    # when
    prepared = workflow._convert_dia_data(dia_data)

    # then
    mock_set_thread_count.assert_called_once_with(THREAD_COUNT)
    mock_dia_data_to_rust.assert_called_once_with(dia_data)
    assert prepared is mock_dia_data_to_rust.return_value


def test_rust_workflow_rejects_timstof_data(tmp_path):
    # given
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)

    # when / then
    with pytest.raises(GenericUserError):
        workflow._convert_dia_data(MagicMock(spec=TimsTOFTranspose))


def test_rust_workflow_does_not_support_multiplexing(tmp_path):
    # given
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)

    # when / then
    with pytest.raises(NotImplementedError):
        workflow.requantify(MagicMock())


@patch("alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock")
def test_calibration_fdr_starts_from_the_latest_classifier(_):
    """Test that the FDR of a calibration search starts from the latest classifier of this run, not from the version
    of an optimization manager loaded from an earlier run (`general.reuse_calibration`)."""
    # Given: an optimization manager that carries classifier version 3 from an earlier run
    optimization_manager = MagicMock(classifier_version=3)
    fdr_manager = MagicMock()
    extraction_handler = MagicMock()
    extraction_handler.perform_fdr_and_filter_candidates.return_value = (None, None)
    rust_handler = RustOptimizationHandler(
        {},
        optimization_manager,
        MagicMock(),
        fdr_manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        create_extraction_handler=MagicMock(),
    )
    python_handler = PythonOptimizationHandler(
        {"search": {}, "fdr": {"competitive_scoring": True}},
        optimization_manager,
        MagicMock(),
        fdr_manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        create_extraction_handler=MagicMock(),
    )

    # When: both backends perform the FDR of a calibration search
    rust_handler._perform_fdr(extraction_handler, MagicMock())
    with patch(
        "alphadia.workflow.peptidecentric.python.optimization_handler.get_decoy_strategy"
    ):
        python_handler._perform_fdr(MagicMock(), MagicMock())

    # Then: both start from the latest classifier
    rust_call = extraction_handler.perform_fdr_and_filter_candidates.call_args
    assert rust_call.args[2] == FDRManager.LATEST_VERSION
    assert fdr_manager.fit_predict.call_args.kwargs["version"] == (
        FDRManager.LATEST_VERSION
    )


def test_search_parameter_optimization_applies_the_calibration_result(tmp_path):
    """Test that the workflow applies the tolerances and the classifier version that the calibration returns."""
    # Given: a workflow whose calibration returns an MS2 tolerance of 9 and classifier version 4
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)
    workflow._fdr_manager = MagicMock()
    workflow._spectral_library = MagicMock()
    result = CalibrationResult({SearchTolerance.MS2_ERROR: 9.0}, classifier_version=4)

    # When: the workflow runs the search parameter optimization
    with (
        patch.object(
            RustPeptideCentricWorkflow, "OPTIMIZATION_HANDLER_CLASS"
        ) as handler,
        patch.object(RustPeptideCentricWorkflow, "_save_managers"),
    ):
        handler.return_value.search_parameter_optimization.return_value = result
        workflow.search_parameter_optimization()

    # Then: the optimization manager holds the result for the main search
    assert workflow.optimization_manager.ms2_error == 9.0
    assert workflow.optimization_manager.classifier_version == 4

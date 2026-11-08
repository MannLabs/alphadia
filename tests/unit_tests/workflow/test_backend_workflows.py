"""Unit tests for the backend-specific parts of the Python and Rust workflows."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from alphadia_search_rs import FeatureSet

from alphadia.exceptions import GenericUserError
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.workflow.peptidecentric.python.extraction_handler import (
    PythonExtractionHandler,
)
from alphadia.workflow.peptidecentric.python.utils import feature_columns
from alphadia.workflow.peptidecentric.python.workflow import (
    PythonPeptideCentricWorkflow,
)
from alphadia.workflow.peptidecentric.rust.extraction_handler import (
    RustExtractionHandler,
)
from alphadia.workflow.peptidecentric.rust.ng_mapper import get_feature_names
from alphadia.workflow.peptidecentric.rust.optimization_handler import (
    RustOptimizationHandler,
)
from alphadia.workflow.peptidecentric.rust.workflow import RustPeptideCentricWorkflow

THREAD_COUNT = 3


def _create_workflow(workflow_class, tmp_path):
    config = {
        "general": {
            "save_figures": False,
            "reuse_calibration": False,
            "thread_count": THREAD_COUNT,
        },
        "output_directory": str(tmp_path),
        "search": {
            "top_k_fragments_selection": 12,
            "exclude_shared_ions": True,
            "quant_window": 3,
            "quant_all": True,
            "experimental_xic": False,
        },
    }
    workflow = workflow_class("raw1", config)
    workflow.reporter = MagicMock()
    workflow._optimization_manager = MagicMock()
    workflow._calibration_manager = MagicMock()
    workflow._dia_data = MagicMock(has_ms1=True, has_mobility=False)
    return workflow


@pytest.mark.parametrize(
    ("workflow_class", "expected_columns"),
    [
        (PythonPeptideCentricWorkflow, feature_columns),
        (RustPeptideCentricWorkflow, get_feature_names(FeatureSet.Full)),
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


@pytest.mark.parametrize(
    ("workflow_class", "extraction_handler_class"),
    [
        (PythonPeptideCentricWorkflow, PythonExtractionHandler),
        (RustPeptideCentricWorkflow, RustExtractionHandler),
    ],
)
def test_workflow_creates_extraction_handler_of_its_backend(
    workflow_class, extraction_handler_class, tmp_path
):
    # given
    workflow = _create_workflow(workflow_class, tmp_path)

    # when
    extraction_handler = workflow._create_extraction_handler()

    # then
    assert type(extraction_handler) is extraction_handler_class


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
    handler = RustOptimizationHandler(
        {},
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        create_extraction_handler=lambda: extraction_handler,
    )

    # when
    precursor_df = handler._process_batch()

    # then
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
    pd.testing.assert_frame_equal(precursor_df, precursor_fdr_df)


def test_python_workflow_keeps_dia_data(tmp_path):
    # given
    workflow = _create_workflow(PythonPeptideCentricWorkflow, tmp_path)
    dia_data = MagicMock(name="dia_data")

    # when
    prepared = workflow._prepare_dia_data(dia_data)

    # then
    assert prepared is dia_data


@patch("alphadia.workflow.peptidecentric.rust.workflow.set_ng_thread_count")
@patch("alphadia.workflow.peptidecentric.rust.workflow.dia_data_to_ng")
def test_rust_workflow_converts_dia_data(
    mock_dia_data_to_ng, mock_set_ng_thread_count, tmp_path
):
    # given
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)
    dia_data = MagicMock(
        name="dia_data", cycle=np.zeros((1, 4)), rt_values=np.arange(3)
    )
    mock_dia_data_to_ng.return_value = MagicMock(
        cycle=dia_data.cycle, rt_values=dia_data.rt_values
    )

    # when
    prepared = workflow._prepare_dia_data(dia_data)

    # then
    mock_set_ng_thread_count.assert_called_once_with(THREAD_COUNT)
    mock_dia_data_to_ng.assert_called_once_with(dia_data)
    assert prepared is mock_dia_data_to_ng.return_value


def test_rust_workflow_rejects_timstof_data(tmp_path):
    # given
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)

    # when / then
    with pytest.raises(GenericUserError):
        workflow._prepare_dia_data(MagicMock(spec=TimsTOFTranspose))


def test_rust_workflow_does_not_support_multiplexing(tmp_path):
    # given
    workflow = _create_workflow(RustPeptideCentricWorkflow, tmp_path)

    # when / then
    with pytest.raises(NotImplementedError):
        workflow.requantify(MagicMock())

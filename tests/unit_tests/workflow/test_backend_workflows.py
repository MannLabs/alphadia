"""Unit tests for the backend-specific parts of the Python and Rust workflows."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from alphadia_search_rs import FeatureSet

from alphadia.exceptions import GenericUserError
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.workflow.peptidecentric.extraction_handler import (
    ClassicExtractionHandler,
    NgExtractionHandler,
)
from alphadia.workflow.peptidecentric.ng.ng_mapper import get_feature_names
from alphadia.workflow.peptidecentric.python_workflow import (
    PythonOptimizationHandler,
    PythonPeptideCentricWorkflow,
)
from alphadia.workflow.peptidecentric.rust_workflow import (
    RustOptimizationHandler,
    RustPeptideCentricWorkflow,
)
from alphadia.workflow.peptidecentric.utils import feature_columns

THREAD_COUNT = 3


def _create_workflow(workflow_class, tmp_path):
    config = {
        "general": {
            "save_figures": False,
            "reuse_calibration": False,
            "thread_count": THREAD_COUNT,
        },
        "output_directory": str(tmp_path),
    }
    workflow = workflow_class("raw1", config)
    workflow.reporter = MagicMock()
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
    ("workflow_class", "optimization_handler_class", "extraction_handler_class"),
    [
        (
            PythonPeptideCentricWorkflow,
            PythonOptimizationHandler,
            ClassicExtractionHandler,
        ),
        (RustPeptideCentricWorkflow, RustOptimizationHandler, NgExtractionHandler),
    ],
)
def test_workflow_uses_handlers_of_its_backend(
    workflow_class, optimization_handler_class, extraction_handler_class
):
    # when / then
    assert workflow_class.OPTIMIZATION_HANDLER_CLASS is optimization_handler_class
    assert workflow_class.EXTRACTION_HANDLER_CLASS is extraction_handler_class
    assert (
        optimization_handler_class.EXTRACTION_HANDLER_CLASS is extraction_handler_class
    )


def test_python_workflow_keeps_dia_data(tmp_path):
    # given
    workflow = _create_workflow(PythonPeptideCentricWorkflow, tmp_path)
    dia_data = MagicMock(name="dia_data")

    # when
    prepared = workflow._prepare_dia_data(dia_data)

    # then
    assert prepared is dia_data


@patch("alphadia.workflow.peptidecentric.rust_workflow.set_ng_thread_count")
@patch("alphadia.workflow.peptidecentric.rust_workflow.dia_data_to_ng")
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

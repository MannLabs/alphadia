"""Unit tests for selecting the workflow implementation by extraction backend."""

import pytest

from alphadia.constants.keys import ExtractionBackend
from alphadia.exceptions import ConfigError
from alphadia.workflow.peptidecentric.python.workflow import (
    PythonPeptideCentricWorkflow,
)
from alphadia.workflow.peptidecentric.rust.workflow import RustPeptideCentricWorkflow
from alphadia.workflow.peptidecentric.workflow_factory import get_workflow_class


@pytest.mark.parametrize(
    ("backend", "expected_class"),
    [
        (ExtractionBackend.PYTHON, PythonPeptideCentricWorkflow),
        (ExtractionBackend.RUST, RustPeptideCentricWorkflow),
        ("RUST", RustPeptideCentricWorkflow),
    ],
)
def test_get_workflow_class_returns_class_for_backend(backend, expected_class):
    # given
    config = {"search": {"extraction_backend": backend}}

    # when
    workflow_class = get_workflow_class(config)

    # then
    assert workflow_class is expected_class


def test_get_workflow_class_raises_for_unknown_backend():
    # given
    config = {"search": {"extraction_backend": "fortran"}}

    # when / then
    with pytest.raises(ConfigError, match="CONFIG_ERROR"):
        get_workflow_class(config)

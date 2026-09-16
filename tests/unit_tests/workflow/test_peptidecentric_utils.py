"""Unit tests for the peptidecentric workflow utils."""

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    get_feature_names,
)
from alphadia.workflow.peptidecentric.utils import (
    feature_columns,
    get_classifier_feature_columns,
)


def test_get_classifier_feature_columns_for_rust_backend():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert columns == get_feature_names() + get_context_feature_names()


def test_get_classifier_feature_columns_for_python_backend():
    # when
    columns = get_classifier_feature_columns("python")

    # then
    assert columns == feature_columns

"""Unit tests for the peptidecentric workflow utils."""

import pytest

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    get_feature_names,
)
from alphadia.workflow.peptidecentric.utils import (
    IDF_FEATURE_COLUMNS,
    feature_columns,
    get_classifier_feature_columns,
)


def test_get_classifier_feature_columns_for_rust_backend():
    # when
    columns = get_classifier_feature_columns("rust", "own")

    # then
    assert columns == get_feature_names() + get_context_feature_names()


def test_get_classifier_feature_columns_leaves_out_the_idf_features():
    # when
    columns = get_classifier_feature_columns("rust", "none")

    # then
    assert set(IDF_FEATURE_COLUMNS) <= set(get_feature_names())
    assert columns == [
        name
        for name in get_feature_names() + get_context_feature_names()
        if name not in IDF_FEATURE_COLUMNS
    ]


def test_get_classifier_feature_columns_for_python_backend():
    # when
    columns = get_classifier_feature_columns("python", "none")

    # then
    assert columns == feature_columns


def test_get_classifier_feature_columns_rejects_unknown_idf_features():
    # when / then
    with pytest.raises(ValueError, match="Unknown idf_features"):
        get_classifier_feature_columns("rust", "decoy")

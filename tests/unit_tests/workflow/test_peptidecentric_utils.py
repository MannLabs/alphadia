"""Unit tests for the peptidecentric workflow utils."""

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    get_feature_names,
)
from alphadia.workflow.peptidecentric.utils import (
    IDF_FEATURE_MARKER,
    LIBRARY_CROWDING_FEATURES,
    feature_columns,
    get_classifier_feature_columns,
)


def test_get_classifier_feature_columns_for_rust_backend():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert columns == [
        name
        for name in get_feature_names() + get_context_feature_names()
        if IDF_FEATURE_MARKER not in name and name not in LIBRARY_CROWDING_FEATURES
    ]


def test_get_classifier_feature_columns_leaves_out_the_idf_features():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert any(IDF_FEATURE_MARKER in name for name in get_feature_names())
    assert not any(IDF_FEATURE_MARKER in name for name in columns)


def test_get_classifier_feature_columns_leaves_out_the_library_crowding_features():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert set(LIBRARY_CROWDING_FEATURES) <= set(get_context_feature_names())
    assert not set(LIBRARY_CROWDING_FEATURES) & set(columns)


def test_get_classifier_feature_columns_for_python_backend():
    # when
    columns = get_classifier_feature_columns("python")

    # then
    assert columns == feature_columns

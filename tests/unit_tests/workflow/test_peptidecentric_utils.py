"""Unit tests for the peptidecentric workflow utils."""

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    get_feature_names,
)
from alphadia.workflow.peptidecentric.utils import (
    CROWDING_CONTEXT_FEATURES,
    DECOY_SCHEME_FEATURES,
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
        if name not in DECOY_SCHEME_FEATURES and name not in CROWDING_CONTEXT_FEATURES
    ]


def test_get_classifier_feature_columns_leaves_out_the_idf_features():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert set(DECOY_SCHEME_FEATURES) <= set(get_feature_names())
    assert not set(DECOY_SCHEME_FEATURES) & set(columns)


def test_get_classifier_feature_columns_for_python_backend():
    # when
    columns = get_classifier_feature_columns("python")

    # then
    assert columns == feature_columns


def test_get_classifier_feature_columns_leaves_out_the_crowding_context_features():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert set(CROWDING_CONTEXT_FEATURES) <= set(get_context_feature_names())
    assert not set(CROWDING_CONTEXT_FEATURES) & set(columns)
    assert set(get_context_feature_names()) - set(CROWDING_CONTEXT_FEATURES) <= set(
        columns
    )

"""Unit tests for the peptidecentric workflow utils."""

from pathlib import Path

import yaml

import alphadia
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    get_feature_names,
)
from alphadia.workflow.peptidecentric.utils import (
    DECOY_SCHEME_FEATURES,
    feature_columns,
    get_classifier_feature_columns,
)

DEFAULT_CONFIG_PATH = Path(alphadia.__file__).parent / "constants" / "default.yaml"


def test_get_classifier_feature_columns_for_rust_backend():
    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert columns == [
        name
        for name in get_feature_names() + get_context_feature_names()
        if name not in DECOY_SCHEME_FEATURES
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


def test_default_prefilter_features_are_classifier_features():
    # given
    with open(DEFAULT_CONFIG_PATH) as f:
        feature_subset = yaml.safe_load(f)["fdr"]["prefilter"]["feature_subset"]

    # when
    columns = get_classifier_feature_columns("rust")

    # then
    assert set(feature_subset) <= set(columns)

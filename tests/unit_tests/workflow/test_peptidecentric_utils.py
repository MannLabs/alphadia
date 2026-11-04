"""Unit tests for the peptidecentric workflow utils."""

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.outputtransform.quantification.quant_builder import (
    INTENSITY_LIBRARY_COLUMN,
)
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_claim_feature_names,
    get_context_feature_names,
    get_feature_names,
)
from alphadia.workflow.peptidecentric.utils import (
    IDF_FEATURE_COLUMNS,
    add_library_fragment_intensity,
    feature_columns,
    get_classifier_feature_columns,
)


def test_get_classifier_feature_columns_for_rust_backend():
    # when
    columns = get_classifier_feature_columns("rust", use_idf_features=True)

    # then
    assert (
        columns
        == get_feature_names() + get_context_feature_names() + get_claim_feature_names()
    )


def test_get_classifier_feature_columns_leaves_out_the_idf_features():
    # when
    columns = get_classifier_feature_columns("rust", use_idf_features=False)

    # then
    assert set(IDF_FEATURE_COLUMNS) <= set(get_feature_names())
    assert columns == [
        name
        for name in get_feature_names()
        + get_context_feature_names()
        + get_claim_feature_names()
        if name not in IDF_FEATURE_COLUMNS
    ]


def test_get_classifier_feature_columns_for_python_backend():
    # when
    columns = get_classifier_feature_columns("python", use_idf_features=False)

    # then
    assert columns == feature_columns


def _flat_library():
    library = SpecLibFlat()
    library._precursor_df = pd.DataFrame(
        {
            "precursor_idx": [5, 7],
            "flat_frag_start_idx": [0, 2],
            "flat_frag_stop_idx": [2, 5],
        }
    )
    library._fragment_df = pd.DataFrame(
        {
            "number": [1, 2, 1, 2, 3],
            "type": [121, 121, 98, 121, 121],
            "charge": [1, 1, 1, 1, 2],
            "loss_type": [0, 0, 0, 0, 0],
            "intensity": [1.0, 0.5, 0.2, 1.0, 0.7],
        }
    )
    return library


def test_add_library_fragment_intensity_matches_fragments_by_ion():
    # given quantified fragments in a different order than the library, one of them unknown to it
    fragments_df = pd.DataFrame(
        {
            "precursor_idx": np.array([7, 5, 7, 5], dtype=np.uint32),
            "number": np.array([3, 2, 1, 9], dtype=np.uint8),
            "type": np.array([121, 121, 98, 121], dtype=np.uint8),
            "charge": np.array([2, 1, 1, 1], dtype=np.uint8),
            "loss_type": np.array([0, 0, 0, 0], dtype=np.uint8),
            "intensity": [10.0, 20.0, 30.0, 40.0],
        }
    )

    # when
    result = add_library_fragment_intensity(fragments_df, _flat_library())

    # then
    np.testing.assert_allclose(result[INTENSITY_LIBRARY_COLUMN], [0.7, 0.5, 0.2, 0.0])
    np.testing.assert_allclose(result["intensity"], fragments_df["intensity"])

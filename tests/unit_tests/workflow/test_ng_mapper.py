"""Unit tests for the NG mapper helpers around the context features and the decoy IDF."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from conftest import mock_context_features

from alphadia.libtransform.base import ProcessingPipeline
from alphadia.libtransform.decoy import DIANN_INNER, DecoyGenerator
from alphadia.libtransform.flatten import FlattenLibrary, InitFlatColumns
from alphadia.libtransform.harmonize import PrecursorInitializer
from alphadia.libtransform.loader import DynamicLoader
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    get_context_feature_names,
    merge_context_features,
    target_fragment_mz_for_decoys,
)

_LIBRARY_TSV = """PrecursorMz	ProductMz	Annotation	ProteinId	GeneName	PeptideSequence	ModifiedPeptideSequence	PrecursorCharge	LibraryIntensity	NormalizedRetentionTime	PrecursorIonMobility	FragmentType	FragmentCharge	FragmentSeriesNumber	FragmentLossType
300.156968	333.188096	y3^1	Q9CX84	Rgs19	LMHSPTGR	LMHSPTGR	3	4311.4	-25.67		y	1	3
300.156968	430.24086	y4^1	Q9CX84	Rgs19	LMHSPTGR	LMHSPTGR	3	7684.9	-25.67		y	1	4
300.156968	517.27289	y5^1	Q9CX84	Rgs19	LMHSPTGR	LMHSPTGR	3	10000.0	-25.67		y	1	5
300.159143	313.187033	y5^2	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	4817.8	29.42		y	2	5
300.159143	375.223813	y3^1	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	8740.7	29.42		y	1	3
300.159143	488.307878	y4^1	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	10000.0	29.42		y	1	4
300.159143	639.273285	b6^1	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	1844.4	29.42		b	1	6
"""


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"precursor_idx": [1, 1, 2], "rank": [0, 1, 0], "score": [1.0, 0.5, 2.0]}
    )


def test_merge_context_features_aligns_on_candidate():
    # given: the context rows come in a different order than the features
    features_df = _features_df()
    context_features = mock_context_features([2, 1, 1], [0, 1, 0])

    # when
    merged_df = merge_context_features(features_df, context_features)

    # then
    assert merged_df["precursor_idx"].tolist() == [1, 1, 2]
    assert merged_df["rank"].tolist() == [0, 1, 0]
    for name in get_context_feature_names():
        assert merged_df[name].tolist() == [2.0, 1.0, 0.0]


def test_merge_context_features_raises_on_missing_candidate():
    # given: candidate (2, 0) has no context features
    features_df = _features_df()
    context_features = mock_context_features([1, 1], [0, 1])

    # when / then
    with pytest.raises(ValueError, match="missing for 1 candidates"):
        merge_context_features(features_df, context_features)


def test_merge_context_features_raises_on_duplicate_candidate():
    # given: candidate (1, 0) appears twice
    features_df = _features_df()
    context_features = mock_context_features([1, 1, 2, 1], [0, 1, 0, 0])

    # when / then
    with pytest.raises(ValueError, match="duplicate candidates") as error:
        merge_context_features(features_df, context_features)
    assert "1" in str(error.value)


def _flat_library_with_decoys():
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as library_file:
        library_file.write(_LIBRARY_TSV.encode())
    speclib = ProcessingPipeline([DynamicLoader(), PrecursorInitializer()])(
        library_file.name
    )
    return ProcessingPipeline(
        [DecoyGenerator(decoy_type=DIANN_INNER), FlattenLibrary(), InitFlatColumns()]
    )(speclib)


def test_target_fragment_mz_for_decoys_gives_each_decoy_fragment_its_target_fragment_mz():
    # given: a library whose decoys are generated and flattened as in a search
    speclib = _flat_library_with_decoys()
    precursor_df, fragment_df = speclib.precursor_df, speclib.fragment_df
    owner = np.repeat(
        precursor_df.index.to_numpy(),
        (precursor_df["flat_frag_stop_idx"] - precursor_df["flat_frag_start_idx"]),
    )
    fragments = fragment_df.iloc[
        np.concatenate(
            [
                np.arange(start, stop)
                for start, stop in zip(
                    precursor_df["flat_frag_start_idx"],
                    precursor_df["flat_frag_stop_idx"],
                )
            ]
        )
    ].assign(
        **{
            col: precursor_df.loc[owner, col].to_numpy()
            for col in ["decoy", "elution_group_idx", "channel"]
        },
        precursor_charge=precursor_df.loc[owner, "charge"].to_numpy(),
    )

    # when
    fragment_mz = target_fragment_mz_for_decoys(precursor_df, fragment_df)

    # then: decoy fragments carry the m/z of the target fragment with the same annotation, targets keep theirs
    fragments["new_mz"] = fragment_mz[fragments.index]
    key = [
        "elution_group_idx",
        "channel",
        "precursor_charge",
        "type",
        "number",
        "charge",
        "loss_type",
    ]
    decoys = fragments[fragments["decoy"] == 1]
    targets = fragments[fragments["decoy"] == 0]
    paired = decoys.merge(targets, on=key, suffixes=("_decoy", "_target"))
    assert len(decoys) > 0
    assert len(paired) == len(decoys)
    assert not np.allclose(paired["mz_library_decoy"], paired["mz_library_target"])
    np.testing.assert_array_equal(paired["new_mz_decoy"], paired["mz_library_target"])
    np.testing.assert_array_equal(targets["new_mz"], targets["mz_library"])

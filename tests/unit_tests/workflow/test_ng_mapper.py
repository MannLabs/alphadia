"""Unit tests for the NG mapper helpers around the context features and the decoy IDF."""

import numpy as np
import pandas as pd
import pytest
from conftest import mock_context_features

from alphadia.libtransform.base import ProcessingPipeline
from alphadia.libtransform.decoy import DecoyGenerator
from alphadia.libtransform.flatten import FlattenLibrary, InitFlatColumns
from alphadia.libtransform.harmonize import PrecursorInitializer
from alphadia.libtransform.loader import DynamicLoader
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    decoy_idf_from_targets,
    get_context_feature_names,
    merge_context_features,
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


def _flat_library_with_decoys(tmp_path):
    library_path = tmp_path / "library.tsv"
    library_path.write_text(_LIBRARY_TSV)
    speclib = ProcessingPipeline([DynamicLoader(), PrecursorInitializer()])(
        str(library_path)
    )
    return ProcessingPipeline(
        [DecoyGenerator(decoy_type="diann"), FlattenLibrary(), InitFlatColumns()]
    )(speclib)


def _fragments_with_precursors(speclib) -> pd.DataFrame:
    """One row per fragment with the decoy flag, elution group, channel and charge of its precursor."""
    precursor_df = speclib.precursor_df
    rows = np.concatenate(
        [
            np.arange(start, stop)
            for start, stop in zip(
                precursor_df["flat_frag_start_idx"], precursor_df["flat_frag_stop_idx"]
            )
        ]
    )
    owner = np.repeat(
        precursor_df.index.to_numpy(),
        precursor_df["flat_frag_stop_idx"] - precursor_df["flat_frag_start_idx"],
    )
    return speclib.fragment_df.iloc[rows].assign(
        row=rows,
        decoy=precursor_df.loc[owner, "decoy"].to_numpy(),
        elution_group_idx=precursor_df.loc[owner, "elution_group_idx"].to_numpy(),
        channel=precursor_df.loc[owner, "channel"].to_numpy(),
        precursor_charge=precursor_df.loc[owner, "charge"].to_numpy(),
    )


def test_decoy_idf_from_targets_looks_each_decoy_fragment_up_at_its_target_fragment(
    tmp_path,
):
    # given: a library whose decoys are generated and flattened as in a search
    speclib = _flat_library_with_decoys(tmp_path)
    fragments = _fragments_with_precursors(speclib)

    # when
    fragment_idf_mz, counted = decoy_idf_from_targets(
        speclib.precursor_df, speclib.fragment_df
    )

    # then: decoy fragments are looked up at the target fragment with the same annotation, targets at their own m/z,
    # and only target fragments are counted
    fragments["idf_mz"] = fragment_idf_mz[fragments["row"]]
    fragments["counted"] = counted[fragments["row"]]
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
    np.testing.assert_array_equal(paired["idf_mz_decoy"], paired["mz_library_target"])
    np.testing.assert_array_equal(targets["idf_mz"], targets["mz_library"])
    assert targets["counted"].all()
    assert not decoys["counted"].any()


def _synthetic_library(
    precursors: list[tuple[int, int, int, int]], fragment_numbers: list[list[int]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precursors given as (decoy, elution_group_idx, charge, first fragment m/z), each with fragments of the given
    numbers at m/z first, first + 1, ..."""
    lengths = [len(numbers) for numbers in fragment_numbers]
    stops = np.cumsum(lengths)
    precursor_df = pd.DataFrame(
        {
            "decoy": [p[0] for p in precursors],
            "elution_group_idx": [p[1] for p in precursors],
            "channel": 0,
            "charge": [p[2] for p in precursors],
            "flat_frag_start_idx": stops - lengths,
            "flat_frag_stop_idx": stops,
        }
    )
    fragment_df = pd.DataFrame(
        {
            "mz_library": np.concatenate(
                [
                    p[3] + np.arange(n, dtype=np.float32)
                    for p, n in zip(precursors, lengths)
                ]
            ).astype(np.float32),
            "type": 98,
            "number": np.concatenate(fragment_numbers),
            "charge": 1,
            "loss_type": 0,
        }
    )
    return precursor_df, fragment_df


def test_decoy_idf_from_targets_keeps_the_own_mz_where_there_is_no_single_matching_target():
    # given:
    # - group 0: target and decoy pair; the decoy's last fragment has another annotation
    # - group 1: two targets of the same charge, so their decoy is ambiguous
    # - group 2: the decoy has one fragment more than its target
    # - group 3: a decoy without a target
    precursor_df, fragment_df = _synthetic_library(
        [
            (0, 0, 2, 100),
            (1, 0, 2, 200),
            (0, 1, 2, 300),
            (0, 1, 2, 400),
            (1, 1, 2, 500),
            (0, 2, 2, 600),
            (1, 2, 2, 700),
            (1, 3, 2, 800),
        ],
        [[1, 2, 3], [1, 2, 4], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2, 3], [1, 2]],
    )

    # when
    fragment_idf_mz, counted = decoy_idf_from_targets(precursor_df, fragment_df)

    # then
    expected_mz = [100, 101, 102, 100, 101, 202]
    expected_mz += [300, 301, 400, 401, 500, 501]
    expected_mz += [600, 601, 700, 701, 702]
    expected_mz += [800, 801]
    np.testing.assert_array_equal(
        fragment_idf_mz, np.array(expected_mz, dtype=np.float32)
    )
    expected_counted = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(counted, np.array(expected_counted, dtype=bool))

import numpy as np
import pandas as pd
import pytest

from alphadia.search.plexscoring.utils import (
    calculate_score_groups,
    fragment_correlation,
    fragment_correlation_different,
    merge_missing_columns,
    multiplex_candidates,
    save_corrcoeff,
)


def test_multiplex_candidates():
    test_candidate_df = pd.DataFrame(
        {
            "elution_group_idx": [0, 0, 1],
            "precursor_idx": [0, 1, 3],
            "proba": [0.1, 0.4, 0.3],
            "rank": [0, 0, 0],
            "frame_start": [0, 0, 0],
            "frame_center": [0, 0, 0],
            "frame_stop": [0, 0, 0],
            "scan_start": [0, 0, 0],
            "scan_stop": [0, 0, 0],
            "scan_center": [0, 0, 0],
        }
    )

    test_precursor_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4, 5],
            "elution_group_idx": [0, 0, 0, 1, 1, 1],
            "decoy": [0, 0, 0, 0, 0, 0],
            "channel": [0, 4, 8, 0, 4, 8],
            "flat_frag_start_idx": [0, 0, 0, 0, 0, 0],
            "flat_frag_stop_idx": [0, 0, 0, 0, 0, 0],
            "charge": [2, 2, 2, 2, 2, 2],
            "rt_library": [0, 0, 0, 0, 0, 0],
            "mobility_library": [0, 0, 0, 0, 0, 0],
            "mz_library": [0, 0, 0, 0, 0, 0],
            "proteins": ["A", "A", "A", "A", "A", "A"],
            "genes": ["A", "A", "A", "A", "A", "A"],
        }
    )

    multiplexed_candidates = multiplex_candidates(
        test_candidate_df, test_precursor_df, channels=[]
    )
    assert len(multiplexed_candidates) == 0

    multiplexed_candidates = multiplex_candidates(
        test_candidate_df, test_precursor_df, channels=[0, 4, 8]
    )
    assert multiplexed_candidates["precursor_idx"].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.all(
        np.isclose(
            multiplexed_candidates["proba"].tolist(), [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]
        )
    )


def test_score_groups():
    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "channel": np.array([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]),
            "decoy": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        }
    )

    sample_df = calculate_score_groups(sample_df)

    assert np.allclose(sample_df["score_group_idx"].values, np.arange(10))

    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "channel": np.array([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]),
            "decoy": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        }
    )

    sample_df = calculate_score_groups(sample_df, group_channels=True)

    assert np.allclose(
        sample_df["score_group_idx"].values, np.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 3])
    )

    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "channel": np.array([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]),
            "decoy": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
            "rank": np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
        }
    )

    sample_df = calculate_score_groups(sample_df, group_channels=True)
    assert np.allclose(sample_df["score_group_idx"].values, np.arange(10))

    sample_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(10),
            "elution_group_idx": np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            "channel": np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
            "decoy": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            "rank": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        }
    )

    sample_df = calculate_score_groups(sample_df, group_channels=True)

    assert np.allclose(
        sample_df["score_group_idx"].values, np.array([0, 0, 1, 1, 2, 3, 4, 4, 5, 5])
    )


@pytest.fixture()
def left_and_right_df():
    left_df = pd.DataFrame([{"idx": 1, "col_1": 0, "col_2": 0}])

    right_df = pd.DataFrame([{"idx": 1, "col_3": 0, "col_4": 0}])
    return left_df, right_df


def test_merge_missing_fail_on(left_and_right_df):
    # given:
    left_df, right_df = left_and_right_df

    # when, then
    with pytest.raises(ValueError):
        merge_missing_columns(left_df, right_df, ["col_3"], on="idx_doesnt_exist")


def test_merge_missing_fail_right(left_and_right_df):
    # given:
    left_df, right_df = left_and_right_df

    # when, then
    with pytest.raises(ValueError):
        merge_missing_columns(left_df, right_df, ["col_5"], on="idx")


def test_merge_missing(left_and_right_df):
    # given
    left_df, right_df = left_and_right_df
    # when
    df = merge_missing_columns(left_df, right_df, ["col_3"], on="idx")
    # then
    assert np.all(df.columns == ["idx", "col_1", "col_2", "col_3"])


def test_save_corrcoeff():
    p = save_corrcoeff(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
        np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32),
    )
    assert np.isclose(p, -1.0)

    p = save_corrcoeff(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
    )
    assert np.isclose(p, 1.0)

    p = save_corrcoeff(
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
    )
    assert np.isclose(p, 0.0)


def test_fragment_correlation():
    a = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[3, 2, 1], [1, 2, 3]], [[0, 0, 0], [0, 0, 0]]]
    )
    corr = fragment_correlation(a)
    assert corr.shape == (2, 3, 3)

    test_a = np.array(
        [
            [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert np.allclose(corr, test_a)

    b = np.zeros((10, 10, 10))
    corr = fragment_correlation(b)
    assert corr.shape == (10, 10, 10)
    assert np.allclose(corr, b)


def test_fragment_correlation_different():
    a = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[3, 2, 1], [1, 2, 3]], [[0, 0, 0], [0, 0, 0]]]
    )
    corr = fragment_correlation_different(a, a)
    assert corr.shape == (2, 3, 3)

    test_a = np.array(
        [
            [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert np.allclose(corr, test_a)

    b = np.zeros((10, 10, 10))
    corr = fragment_correlation_different(b, b)
    assert corr.shape == (10, 10, 10)
    assert np.allclose(corr, b)

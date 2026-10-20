import numpy as np
import pandas as pd

from alphadia.fragcomp.fragcomp import compete_for_fragments
from alphadia.fragcomp.utils import add_frag_start_stop_idx, candidate_hash


def test_fragment_competition():
    cycle = np.array([[[[90, 110]], [[190, 210]]]])

    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(6, dtype=np.uint32),
            "rt_observed": np.array([10.0, 20.0, 20.0, 10.0, 10.0, 20]),
            "mz_observed": np.array([100, 100, 100, 200, 200, 200]),
            "proba": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "rank": np.zeros(6, dtype=np.uint8),
        }
    )

    frag_df = pd.DataFrame(
        {
            "precursor_idx": np.repeat(np.arange(6, dtype=np.uint32), 10),
            "mz_observed": np.tile(np.arange(100, 110), 6),
            "rank": np.zeros(60, dtype=np.uint8),
        }
    )

    # when
    psm_df = compete_for_fragments(psm_df, frag_df, cycle)

    pd.testing.assert_frame_equal(
        psm_df.reset_index(drop=True),
        pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1, 3, 5], dtype=np.uint32),
                "rt_observed": np.array([10.0, 20.0, 10.0, 20]),
                "mz_observed": np.array([100, 100, 200, 200]),
                "proba": np.array([0.1, 0.2, 0.4, 0.6]),
                "rank": np.array([0, 0, 0, 0], dtype=np.uint8),
                "_candidate_idx": np.array([0, 1, 3, 5], dtype=np.uint64),
            }
        ),
    )


def test_candidate_hash():
    """Test the candidate_hash function to ensure it generates correct hashes."""
    # last element required 64 bit representation
    precursor_idx = np.array([1, 2, 1000000])
    rank = np.array([0, 1, 2])

    # Test the candidate_hash function
    hash_values = candidate_hash(precursor_idx, rank)

    assert all(hash_values == np.array([1, 4294967298, 8590934592]))
    assert hash_values.dtype == np.uint64


def test_add_frag_start_stop_idx_keeps_psms_without_fragments():
    # given: PSM 7 has no fragments
    psm_df = pd.DataFrame({"_candidate_idx": np.array([3, 7, 5], dtype=np.uint64)})
    frag_df = pd.DataFrame(
        {"_candidate_idx": np.array([3, 3, 5], dtype=np.uint64), "mz_observed": 1.0}
    )

    # when
    psm_df = add_frag_start_stop_idx(psm_df, frag_df)

    # then: it stays, with an empty fragment range, so it cannot compete
    assert psm_df["_candidate_idx"].tolist() == [3, 7, 5]
    assert psm_df["_frag_start_idx"].tolist() == [0, 0, 2]
    assert psm_df["_frag_stop_idx"].tolist() == [2, 0, 3]


def test_fragment_competition_keeps_psm_without_fragments():
    # given: PSM 1 co-elutes with the better PSM 0 but has no quantified fragments
    cycle = np.array([[[[90, 110]]]])
    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(2, dtype=np.uint32),
            "rt_observed": np.array([10.0, 10.0]),
            "mz_observed": np.array([100, 100]),
            "proba": np.array([0.1, 0.2]),
            "rank": np.zeros(2, dtype=np.uint8),
        }
    )
    frag_df = pd.DataFrame(
        {
            "precursor_idx": np.zeros(3, dtype=np.uint32),
            "mz_observed": np.array([300.0, 400.0, 500.0]),
            "rank": np.zeros(3, dtype=np.uint8),
        }
    )

    # when
    psm_df = compete_for_fragments(psm_df, frag_df, cycle)

    # then: both PSMs survive
    assert psm_df["precursor_idx"].tolist() == [0, 1]

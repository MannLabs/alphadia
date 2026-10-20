import numpy as np
import pandas as pd

from alphadia.fragcomp.fragcomp import compete_for_fragments
from alphadia.fragcomp.utils import candidate_hash


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


def test_fragment_competition_min_overlapping_fragments():
    # Given: two co-eluting PSMs in one window that share exactly two ions
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
            "precursor_idx": np.repeat(np.arange(2, dtype=np.uint32), 3),
            "mz_observed": np.array([300.0, 400.0, 500.0, 300.0, 400.0, 900.0]),
            "rank": np.zeros(6, dtype=np.uint8),
        }
    )

    # When: they compete at the default and at a minimum of three shared ions
    by_default = compete_for_fragments(psm_df.copy(), frag_df.copy(), cycle)
    with_three = compete_for_fragments(
        psm_df.copy(), frag_df.copy(), cycle, min_overlapping_fragments=3
    )

    # Then: two shared ions are a conflict by default, a coincidence at three
    assert by_default["precursor_idx"].tolist() == [0]
    assert with_three["precursor_idx"].tolist() == [0, 1]

import numpy as np
import pandas as pd

from alphadia.fragcomp.fragcomp import (
    FragmentCompetition,
    _compete_for_fragments,
    _get_fragment_overlap,
)
from alphadia.fragcomp.utils import candidate_hash


def test_fragment_overlap():
    frag_mz_1 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    frag_mz_2 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    assert _get_fragment_overlap(frag_mz_1, frag_mz_2) == 10

    frag_mz_1 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    frag_mz_2 = np.array([100])
    assert _get_fragment_overlap(frag_mz_1, frag_mz_2) == 1

    frag_mz_1 = np.array([])
    frag_mz_2 = np.array([])
    assert _get_fragment_overlap(frag_mz_1, frag_mz_2) == 0

    frag_mz_1 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    frag_mz_2 = np.array([])
    assert _get_fragment_overlap(frag_mz_1, frag_mz_2) == 0

    frag_mz_1 = np.array([])
    frag_mz_2 = np.array([100, 200, 300, 400, 500, 600, 700, 801, 901, 1001])
    assert _get_fragment_overlap(frag_mz_1, frag_mz_2) == 0

    frag_mz_1 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    frag_mz_2 = np.array([101, 201, 301, 401, 501, 601, 701, 801, 901, 1001])
    assert _get_fragment_overlap(frag_mz_1, frag_mz_2) == 0


def test_compete_for_fragments():
    rt = np.array([10.0, 20.0, 20.0, 10.0, 10.0, 20])
    valid = np.array([True] * 6)
    frag_start_idx = np.array([0, 10, 20, 30, 40, 50])
    frag_stop_idx = np.array([10, 20, 30, 40, 50, 60])
    fragment_mz = np.tile(np.arange(100, 110), 6)

    _compete_for_fragments(
        np.array([0, 1]),
        np.array([0, 3]),
        np.array([3, 6]),
        rt,
        valid,
        frag_start_idx,
        frag_stop_idx,
        fragment_mz,
        3,
        15,
    )

    assert np.all(valid == np.array([True, True, False, True, False, True]))


def test_fragment_competition():
    cycle = np.array([[[[90, 110]], [[190, 210]]]])

    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(6, dtype=np.uint32),
            "rt_observed": np.array([10.0, 20.0, 20.0, 10.0, 10.0, 20]),
            "valid": np.array([True] * 6),
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
    fragment_competition = FragmentCompetition()
    psm_df = fragment_competition(psm_df, frag_df, cycle)

    pd.testing.assert_frame_equal(
        psm_df.reset_index(drop=True),
        pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1, 3, 5], dtype=np.uint32),
                "rt_observed": np.array([10.0, 20.0, 10.0, 20]),
                "valid": np.array([True] * 4),
                "mz_observed": np.array([100, 100, 200, 200]),
                "proba": np.array([0.1, 0.2, 0.4, 0.6]),
                "rank": np.array([0, 0, 0, 0], dtype=np.uint8),
                "_candidate_idx": np.array([0, 1, 3, 5], dtype=np.int64),
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
    assert hash_values.dtype == np.int64

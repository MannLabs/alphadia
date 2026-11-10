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
            "decoy": np.zeros(6, dtype=np.uint8),
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
                "decoy": np.zeros(4, dtype=np.uint8),
                "_candidate_idx": np.array([0, 1, 3, 5], dtype=np.uint64),
            }
        ),
    )


def test_fragment_competition_within_class():
    """A decoy and a target sharing all ions both keep them; candidates of one class still compete."""
    cycle = np.array([[[[90, 110]]]])
    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(4, dtype=np.uint32),
            "rt_observed": np.full(4, 10.0),
            "mz_observed": np.full(4, 100.0),
            "proba": np.array([0.1, 0.2, 0.3, 0.4]),
            "rank": np.zeros(4, dtype=np.uint8),
            "decoy": np.array([0, 1, 0, 1], dtype=np.uint8),
        }
    )
    frag_df = pd.DataFrame(
        {
            "precursor_idx": np.repeat(np.arange(4, dtype=np.uint32), 10),
            "mz_observed": np.tile(np.arange(100, 110), 4),
            "rank": np.zeros(40, dtype=np.uint8),
        }
    )

    # when
    psm_df = compete_for_fragments(psm_df, frag_df, cycle)

    # then: the best target and the best decoy survive, each knocks out the other of its class
    assert psm_df["precursor_idx"].tolist() == [0, 1]


def test_candidate_hash():
    """Test the candidate_hash function to ensure it generates correct hashes."""
    # last element required 64 bit representation
    precursor_idx = np.array([1, 2, 1000000])
    rank = np.array([0, 1, 2])

    # Test the candidate_hash function
    hash_values = candidate_hash(precursor_idx, rank)

    assert all(hash_values == np.array([1, 4294967298, 8590934592]))
    assert hash_values.dtype == np.uint64

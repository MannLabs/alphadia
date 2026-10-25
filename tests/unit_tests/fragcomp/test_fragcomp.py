import numpy as np
import pandas as pd

from alphadia.fragcomp.fragcomp import compete_for_fragments, competition_mz
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


def test_competition_mz_is_the_window_most_fragments_are_read_from():
    # two adjacent 2 Th windows; both PSMs have M0 at 499.7 in the lower one and M1.. in the upper one
    cycle = np.array([[[[498.0, 500.0]], [[500.0, 502.0]]]], dtype=np.float32)
    light_isotopes = [0.5, 0.3, 0.15, 0.05]
    heavy_isotopes = [0.2, 0.35, 0.3, 0.15]
    psm_df = pd.DataFrame(
        {
            "mz_observed": np.array([499.7, 499.7], dtype=np.float32),
            "charge": np.array([2, 2], dtype=np.uint8),
            **{
                f"i_{k}": np.array([light_isotopes[k], heavy_isotopes[k]])
                for k in range(4)
            },
            "_frag_start_idx": np.array([0, 2], dtype=np.uint64),
            "_frag_stop_idx": np.array([2, 4], dtype=np.uint64),
        }
    )
    frag_df = pd.DataFrame(
        {
            "mz_library": np.array([200.0, 600.0, 200.0, 600.0], dtype=np.float32),
            "charge": np.ones(4, dtype=np.uint8),
            "intensity": np.ones(4, dtype=np.float32),
        }
    )

    # the light precursor's fragments stay in the window of its m/z, the heavy one's move up
    np.testing.assert_array_equal(
        competition_mz(psm_df, frag_df, cycle), np.array([499.0, 501.0])
    )
    np.testing.assert_array_equal(
        competition_mz(
            psm_df.drop(columns=[f"i_{k}" for k in range(4)]), frag_df, cycle
        ),
        np.array([499.7, 499.7], dtype=np.float32),
    )

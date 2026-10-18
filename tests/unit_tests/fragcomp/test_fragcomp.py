import numpy as np
import pandas as pd
import pytest

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


def _two_psms_sharing(n_shared: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cycle = np.array([[[[90, 110]]]])
    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(2, dtype=np.uint32),
            "rt_observed": np.array([10.0, 11.0]),
            "mz_observed": np.array([100.0, 100.0]),
            "proba": np.array([0.1, 0.2]),
            "rank": np.zeros(2, dtype=np.uint8),
        }
    )
    other = np.concatenate(
        [np.arange(300, 300 + n_shared), np.arange(500, 510 - n_shared)]
    )
    frag_df = pd.DataFrame(
        {
            "precursor_idx": np.repeat(np.arange(2, dtype=np.uint32), 10),
            "mz_observed": np.concatenate([np.arange(300, 310), other]).astype(float),
            "rank": np.zeros(20, dtype=np.uint8),
        }
    )
    return psm_df, frag_df, cycle


def test_fragment_competition_two_shared_ions():
    psm_df, frag_df, cycle = _two_psms_sharing(2)

    kept_two = compete_for_fragments(
        psm_df.copy(), frag_df.copy(), cycle, min_overlap=2
    )
    kept_default = compete_for_fragments(psm_df.copy(), frag_df.copy(), cycle)

    assert kept_two["precursor_idx"].tolist() == [0]
    assert kept_default["precursor_idx"].tolist() == [0, 1]


def test_fragment_competition_only_within_targets_and_within_decoys():
    # Given: PSM 1 shares all its ions with the better PSM 0
    psm_df, frag_df, cycle = _two_psms_sharing(10)

    def kept(decoy_flags):
        out = compete_for_fragments(
            psm_df.assign(_decoy=decoy_flags), frag_df.copy(), cycle
        )
        return out["precursor_idx"].tolist()

    # Then: PSM 1 loses only to a PSM of its own kind
    assert kept([0, 0]) == [0]
    assert kept([1, 1]) == [0]
    assert kept([0, 1]) == [0, 1]
    assert kept([1, 0]) == [0, 1]


def test_fragment_competition_removes_the_decoy_of_a_target_that_lost():
    # Given: target 1 shares all its ions with the better target 0; decoy 2 is in the elution group of target 1
    psm_df, frag_df, cycle = _two_psms_sharing(10)
    decoy = psm_df.iloc[[1]].assign(
        precursor_idx=np.uint32(2), rt_observed=100.0, proba=0.3
    )
    psm_df = pd.concat([psm_df, decoy], ignore_index=True).assign(
        _decoy=[0, 0, 1], elution_group_idx=[10, 11, 11]
    )
    frag_df = pd.concat(
        [
            frag_df,
            pd.DataFrame(
                {
                    "precursor_idx": np.uint32(2),
                    "mz_observed": np.arange(700, 710.0),
                    "rank": np.uint8(0),
                }
            ),
        ],
        ignore_index=True,
    )

    # When: fragment competition runs
    kept = compete_for_fragments(psm_df, frag_df, cycle)

    # Then: target 1 and its decoy leave together
    assert kept["precursor_idx"].tolist() == [0]


def test_fragment_competition_matches_rust_at_three_ions():
    rust = pytest.importorskip("alphadia_search_rs")
    rng = np.random.default_rng(0)
    n_psm, n_frag = 400, 8
    cycle = np.array([[[[400, 500]], [[500, 600]], [[600, 700]]]], dtype=np.float32)
    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_psm, dtype=np.uint32),
            "rt_observed": rng.uniform(0, 20, n_psm),
            "mz_observed": rng.uniform(400, 700, n_psm),
            "proba": rng.uniform(0, 1, n_psm),
            "rank": np.zeros(n_psm, dtype=np.uint8),
        }
    )
    frag_df = pd.DataFrame(
        {
            "precursor_idx": np.repeat(np.arange(n_psm, dtype=np.uint32), n_frag),
            "mz_observed": rng.integers(200, 230, n_psm * n_frag).astype(float),
            "rank": np.zeros(n_psm * n_frag, dtype=np.uint8),
        }
    )

    kept = compete_for_fragments(psm_df.copy(), frag_df.copy(), cycle)

    frag_df["_candidate_idx"] = candidate_hash(
        frag_df["precursor_idx"].values, frag_df["rank"].values
    )
    starts = np.arange(n_psm, dtype=np.int64) * n_frag
    valid = rust.FragmentCompetition(3.0, 15.0).compete(
        psm_df["mz_observed"].to_numpy(dtype=np.float32),
        psm_df["precursor_idx"].to_numpy(dtype=np.int64),
        psm_df["proba"].to_numpy(dtype=np.float64),
        psm_df["rt_observed"].to_numpy(dtype=np.float32),
        starts,
        starts + n_frag,
        frag_df["mz_observed"].to_numpy(dtype=np.float32),
        cycle,
    )
    assert 0 < valid.sum() < n_psm
    assert kept["precursor_idx"].tolist() == psm_df["precursor_idx"][valid].tolist()

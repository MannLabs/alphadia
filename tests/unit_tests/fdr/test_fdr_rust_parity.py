"""Parity tests: Rust-accelerated FDR kernels vs the pandas reference."""

import numpy as np
import pandas as pd
import pytest

from alphadia.fdr import fdr

if not fdr._RUST_FDR_AVAILABLE:
    pytest.skip("alphadia_search_rs FDR kernels not available", allow_module_level=True)


def _make_psms(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic PSMs with distinct probas, target/decoy labels, and group ids."""
    rng = np.random.default_rng(seed)
    # distinct probas to avoid tie-ordering ambiguity between sort and counting
    proba = rng.permutation(np.linspace(0.0, 1.0, n, endpoint=False))
    decoy = (rng.random(n) < 0.4).astype(float)
    return pd.DataFrame(
        {
            "proba": proba,
            "_decoy": decoy,
            "precursor_idx": np.arange(n),
            "elution_group_idx": rng.integers(0, n // 2 + 1, n),
            "channel": rng.integers(0, 2, n),
        }
    )


def _run(monkeypatch, use_rust: bool, fn, *args, **kwargs):
    monkeypatch.setattr(fdr, "_USE_RUST_FDR", use_rust and fdr._RUST_FDR_AVAILABLE)
    return fn(*args, **kwargs)


def test_get_q_values_matches_reference(monkeypatch):
    df = _make_psms(50_000)

    rust = _run(monkeypatch, True, fdr.get_q_values, df.copy(), "proba", "_decoy")
    ref = _run(monkeypatch, False, fdr.get_q_values, df.copy(), "proba", "_decoy")

    rust = rust.sort_values("precursor_idx").reset_index(drop=True)
    ref = ref.sort_values("precursor_idx").reset_index(drop=True)

    np.testing.assert_allclose(
        rust["qval"].to_numpy(), ref["qval"].to_numpy(), atol=1e-12
    )


def test_get_q_values_string_tiebreak_matches_reference(monkeypatch):
    """Protein FDR breaks ties on the protein group accession, a string column."""
    rng = np.random.default_rng(1)
    n = 20_000
    df = pd.DataFrame(
        {
            # deliberate proba ties, so the tie-break column decides the ordering
            "proba": rng.choice(np.linspace(0.0, 1.0, n // 10), n),
            "decoy": (rng.random(n) < 0.4).astype(float),
            "pg": [f"P{i:06d}" for i in rng.permutation(n)],
        }
    )

    rust = _run(
        monkeypatch,
        True,
        fdr.get_q_values,
        df.copy(),
        "proba",
        "decoy",
        extra_sort_columns=["pg"],
    )
    ref = _run(
        monkeypatch,
        False,
        fdr.get_q_values,
        df.copy(),
        "proba",
        "decoy",
        extra_sort_columns=["pg"],
    )

    rust = rust.sort_values("pg").reset_index(drop=True)
    ref = ref.sort_values("pg").reset_index(drop=True)

    np.testing.assert_allclose(
        rust["qval"].to_numpy(), ref["qval"].to_numpy(), atol=1e-12
    )


@pytest.mark.parametrize(
    "group_columns",
    [["precursor_idx"], ["channel", "elution_group_idx"], ["elution_group_idx"]],
)
def test_keep_best_matches_reference(monkeypatch, group_columns):
    df = _make_psms(50_000)

    rust = _run(monkeypatch, True, fdr.keep_best, df.copy(), "proba", group_columns)
    ref = _run(monkeypatch, False, fdr.keep_best, df.copy(), "proba", group_columns)

    rust_idx = set(rust["precursor_idx"]) if "precursor_idx" in rust else None
    # compare the set of retained rows by their unique precursor_idx
    assert sorted(rust["precursor_idx"]) == sorted(ref["precursor_idx"])
    assert rust_idx is not None


def test_fused_finalize_threshold_counts_match_reference(monkeypatch):
    """The fused histogram path should agree with the exact pandas chain on the
    number of PSMs passing common FDR thresholds (within quantization noise)."""
    df = _make_psms(200_000)
    group_columns = ["elution_group_idx", "channel"]

    # exact pandas reference: keep_best -> q-values
    ref = _run(monkeypatch, False, fdr.keep_best, df.copy(), "proba", group_columns)
    ref = _run(monkeypatch, False, fdr.get_q_values, ref, "proba", "_decoy")

    # fused sort-free rust path
    group_id = df.groupby(group_columns, sort=False).ngroup().to_numpy(dtype=np.int64)
    order, qvalues = fdr._rs_finalize(
        df["proba"].to_numpy(dtype=np.float64),
        df["_decoy"].to_numpy(dtype=np.float64),
        group_id,
        fdr._FDR_QVALUE_BINS,
    )
    fused = df.iloc[order].copy()
    fused["qval"] = qvalues

    # same survivors after keep-best
    assert sorted(fused["precursor_idx"]) == sorted(ref["precursor_idx"])

    for thresh in (0.001, 0.01, 0.05):
        n_ref = int((ref["qval"] < thresh).sum())
        n_fused = int((fused["qval"] < thresh).sum())
        # quantization tolerance: at most a handful of rows near the boundary
        assert abs(n_ref - n_fused) <= max(
            2, int(0.001 * n_ref)
        ), f"thresh={thresh}: ref={n_ref} fused={n_fused}"

import numpy as np
import pandas as pd

from alphadia.libtransform.next_ions import NEXT_ION_SLOTS, next_ion_mz

FRAGMENT_COLUMNS = ["b_z1", "b_z2", "y_z1", "y_z2"]


def _library(
    n_aa: int, intensity: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One precursor of `n_aa` residues; fragment m/z = 1000 * row + column + 1 so each ion is identifiable."""
    rows = n_aa - 1
    mz = 1000.0 * np.arange(rows)[:, None] + np.arange(1, 5)[None, :]
    precursor_df = pd.DataFrame(
        {"frag_start_idx": [0], "frag_stop_idx": [rows], "nAA": [n_aa]}
    )
    return (
        precursor_df,
        pd.DataFrame(mz, columns=FRAGMENT_COLUMNS),
        pd.DataFrame(intensity, columns=FRAGMENT_COLUMNS),
    )


def test_next_ions_are_the_ranks_after_the_top_k():
    # given 9 rows x 4 columns with distinct intensities, rank r at row r // 4, column r % 4
    intensity = (1.0 - np.arange(36) / 36.0).reshape(9, 4)
    precursor_df, mz_df, intensity_df = _library(10, intensity)

    # when
    out = next_ion_mz(
        precursor_df,
        mz_df,
        intensity_df,
        top_k_fragments=4,
        min_fragment_intensity=0.01,
    )

    # then ranks 4 .. 15 are rows 1 .. 3; row 1 holds b2 (kept) and y8 (number n-2 = 8, kept)
    assert out.shape == (1, NEXT_ION_SLOTS)
    expected = [1000.0 * r + c for r in (1, 2, 3) for c in (1, 2, 3, 4)]
    np.testing.assert_allclose(out[0], expected)


def test_next_ions_skip_terminal_ions_weak_ions_and_empty_mz():
    # given
    intensity = np.zeros((9, 4))
    intensity[0, 0] = 1.0  # b1, the top 1
    intensity[0, 2] = 0.9  # y9 = y_(n-1): left out
    intensity[8, 2] = 0.8  # y1: left out
    intensity[4, 0] = 0.7  # b5 with m/z 0 below: left out
    intensity[3, 3] = 0.6  # y6 2+: kept
    intensity[5, 1] = 0.005  # below the minimum intensity
    precursor_df, mz_df, intensity_df = _library(10, intensity)
    mz_df.iloc[4, 0] = 0.0

    # when
    out = next_ion_mz(
        precursor_df,
        mz_df,
        intensity_df,
        top_k_fragments=1,
        min_fragment_intensity=0.01,
    )

    # then
    np.testing.assert_allclose(out[0, :2], [3004.0, 0.0])

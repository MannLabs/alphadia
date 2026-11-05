import numpy as np
import pandas as pd
import pytest

from alphadia.raw_data.alpharaw_wrapper import AlphaRaw
from alphadia.raw_data.dia_cycle import determine_dia_cycle
from alphadia.raw_data.ms1_alignment import align_ms1_to_ms2_cycle

WINDOWS = [(400.0, 402.0), (402.0, 404.0), (404.0, 406.0), (406.0, 408.0)]
MS2_SCAN_TIME = 0.01
PEAKS_PER_SPECTRUM = 2


def acquisition(
    n_cycles: int, ms1_interval: float, gap: tuple[float, float] | None = None
) -> pd.DataFrame:
    """MS2 scans cycling through WINDOWS back to back, MS1 scans every ms1_interval on their own clock."""
    n_ms2 = n_cycles * len(WINDOWS)
    ms2_rt = np.arange(n_ms2) * MS2_SCAN_TIME
    ms1_rt = np.arange(0.003, ms2_rt[-1], ms1_interval)
    if gap is not None:
        ms1_rt = ms1_rt[(ms1_rt < gap[0]) | (ms1_rt > gap[1])]
    ms2 = pd.DataFrame(
        {
            "rt": ms2_rt,
            "ms_level": 2,
            "isolation_lower_mz": [WINDOWS[i % len(WINDOWS)][0] for i in range(n_ms2)],
            "isolation_upper_mz": [WINDOWS[i % len(WINDOWS)][1] for i in range(n_ms2)],
        }
    )
    ms1 = pd.DataFrame(
        {
            "rt": ms1_rt,
            "ms_level": 1,
            "isolation_lower_mz": -1.0,
            "isolation_upper_mz": -1.0,
        }
    )
    df = pd.concat([ms1, ms2]).sort_values("rt", kind="stable").reset_index(drop=True)
    df["precursor_mz"] = np.where(
        df["ms_level"] == 1, -1.0, df["isolation_lower_mz"] + 1
    )
    df["spec_idx"] = np.arange(len(df))
    df["peak_start_idx"] = df["spec_idx"] * PEAKS_PER_SPECTRUM
    df["peak_stop_idx"] = df["peak_start_idx"] + PEAKS_PER_SPECTRUM
    return df


@pytest.mark.parametrize(("ms1_interval", "slots"), [(0.037, 1), (0.021, 2)])
def test_aligned_table_has_a_regular_cycle_with_ms1_slots_first(ms1_interval, slots):
    aligned = align_ms1_to_ms2_cycle(acquisition(50, ms1_interval))

    cycle, cycle_start, cycle_length = determine_dia_cycle(aligned)

    assert cycle_start == 0
    assert cycle_length == len(WINDOWS) + slots
    np.testing.assert_array_equal(cycle[0, :slots, 0], -1.0)
    np.testing.assert_array_equal(cycle[0, slots:, 0], WINDOWS)
    assert len(aligned) % cycle_length == 0
    assert (aligned["spec_idx"].to_numpy() == np.arange(len(aligned))).all()


def test_slots_hold_the_nearest_ms1_scan_and_the_cycle_rt():
    df = acquisition(50, 0.037)
    ms1 = df[df["ms_level"] == 1]
    rt_of_peaks = dict(zip(ms1["peak_start_idx"], ms1["rt"], strict=True))
    cycle_time = len(WINDOWS) * MS2_SCAN_TIME

    aligned = align_ms1_to_ms2_cycle(df)

    slots = aligned[aligned["ms_level"] == 1]
    cycle_rt = slots["rt"].to_numpy()
    np.testing.assert_allclose(cycle_rt, np.arange(len(slots)) * cycle_time)
    original_rt = slots["peak_start_idx"].map(rt_of_peaks).to_numpy()
    target = cycle_rt + cycle_time / 2
    nearest_possible = np.abs(ms1["rt"].to_numpy()[None, :] - target[:, None]).min(1)
    np.testing.assert_allclose(np.abs(original_rt - target), nearest_possible)


def test_slots_in_an_acquisition_gap_stay_empty():
    aligned = align_ms1_to_ms2_cycle(acquisition(50, 0.037, gap=(0.8, 1.2)))

    slots = aligned[aligned["ms_level"] == 1]
    empty = slots["peak_stop_idx"] == slots["peak_start_idx"]
    in_gap = (slots["rt"] > 0.85) & (slots["rt"] < 1.1)
    assert empty[in_gap].all()
    assert not empty[~in_gap & (slots["rt"] < 0.7)].any()


def test_too_few_ms1_scans_cannot_be_aligned():
    df = acquisition(50, 0.037)
    first_ms1 = df.index[df["ms_level"] == 1][0]
    df = df[(df["ms_level"] == 2) | (df.index == first_ms1)]

    assert align_ms1_to_ms2_cycle(df) is None


class _Raw(AlphaRaw):
    pass


def test_irregular_ms1_is_kept_by_the_raw_reader():
    raw = _Raw()
    raw.spectrum_df = acquisition(50, 0.037)
    n_peaks = int(raw.spectrum_df["peak_stop_idx"].max())
    raw.peak_df = pd.DataFrame(
        {"mz": np.linspace(400, 900, n_peaks), "intensity": np.ones(n_peaks)}
    )

    raw._preprocess_raw_data()

    assert raw.has_ms1
    assert raw.cycle.shape[1] == len(WINDOWS) + 1
    assert raw.cycle[0, 0, 0, 0] == -1.0

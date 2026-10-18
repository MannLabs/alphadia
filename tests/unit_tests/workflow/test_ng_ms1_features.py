"""The rust scoring computes MS1 isotope features once alphaDIA passes the library isotopes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from alphadia_search_rs import PeakGroupScoring, ScoringParameters

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    candidates_to_ng,
    dia_data_to_ng,
    speclib_to_ng,
)

C13 = 1.0033548
PRECURSOR_MZ = 500.0
CHARGE = 2
ISOTOPES = [1.0, 0.6, 0.2]
FRAGMENT_MZ = [300.0, 400.0, 600.0]
PROFILE = [1.0, 5.0, 20.0, 5.0, 1.0]
MS1_TOLERANCE = 4.0


def _speclib(with_isotopes: bool) -> SimpleNamespace:
    precursor_df = pd.DataFrame(
        {
            "precursor_idx": [0],
            "mz_library": [PRECURSOR_MZ],
            "rt_library": [2.0],
            "nAA": [10],
            "charge": [CHARGE],
            "flat_frag_start_idx": [0],
            "flat_frag_stop_idx": [3],
        }
    )
    if with_isotopes:
        for k, abundance in enumerate(ISOTOPES):
            precursor_df[f"i_{k}"] = abundance
    fragment_df = pd.DataFrame(
        {
            "mz_library": FRAGMENT_MZ,
            "intensity": [1.0, 0.8, 0.5],
            "cardinality": [1, 1, 1],
            "charge": [1, 1, 1],
            "loss_type": [0, 0, 0],
            "number": [3, 4, 5],
            "position": [2, 3, 4],
            "type": [ord("b"), ord("y"), ord("y")],
        }
    )
    return SimpleNamespace(precursor_df=precursor_df, fragment_df=fragment_df)


def _dia_data():
    isotope_mz = [PRECURSOR_MZ + k * C13 / CHARGE for k in range(len(ISOTOPES))]
    spectra, peaks_mz, peaks_intensity = [], [], []
    for cycle, height in enumerate(PROFILE):
        for ms_level, mzs, abundances in [
            (1, isotope_mz, ISOTOPES),
            (2, FRAGMENT_MZ, [1.0, 0.8, 0.5]),
        ]:
            start = len(peaks_mz)
            peaks_mz += mzs
            peaks_intensity += [height * 100 * a for a in abundances]
            spectra.append(
                {
                    "isolation_lower_mz": -1.0 if ms_level == 1 else 400.0,
                    "isolation_upper_mz": -1.0 if ms_level == 1 else 600.0,
                    "peak_start_idx": start,
                    "peak_stop_idx": len(peaks_mz),
                    "rt": (cycle + 1) / 60,
                }
            )
    return SimpleNamespace(
        spectrum_df=pd.DataFrame(spectra),
        peak_df=pd.DataFrame({"mz": peaks_mz, "intensity": peaks_intensity}),
        cycle=np.array([[[[-1.0, -1.0]], [[400.0, 600.0]]]], dtype=np.float32),
    )


def _score(with_isotopes: bool) -> dict:
    dia_data = dia_data_to_ng(_dia_data())
    speclib = speclib_to_ng(
        _speclib(with_isotopes),
        rt_column="rt_library",
        precursor_mz_column="mz_library",
        fragment_mz_column="mz_library",
    )
    cycle_len = 2
    candidates = candidates_to_ng(
        pd.DataFrame(
            {
                "precursor_idx": [0],
                "rank": [0],
                "score": [1.0],
                "scan_center": [0],
                "scan_start": [0],
                "scan_stop": [1],
                "frame_center": [2 * cycle_len],
                "frame_start": [0],
                "frame_stop": [len(PROFILE) * cycle_len],
            }
        ),
        dia_data,
    )
    params = ScoringParameters()
    params.update({"mass_tolerance": 7.0, "ms1_mass_tolerance": MS1_TOLERANCE})
    features = PeakGroupScoring(params).score(dia_data, speclib, candidates)
    return {k: np.asarray(v) for k, v in features.to_dict_arrays().items()}


def test_ms1_features_from_library_isotopes():
    features = _score(with_isotopes=True)

    assert features["ms1_isotope_correlation"][0] > 0.99
    assert features["ms1_profile_correlation"][0] > 0.9
    assert features["ms1_isotope_coelution"][0] > 0.99
    assert features["ms1_mass_error"][0] < 1.0
    assert features["log10_ms1_intensity"][0] > 3.0


def test_ms1_features_missing_without_isotopes():
    features = _score(with_isotopes=False)

    assert features["ms1_isotope_correlation"][0] == 0.0
    assert features["ms1_mass_error"][0] == MS1_TOLERANCE
    assert features["log10_ms1_intensity"][0] == 0.0

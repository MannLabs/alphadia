import numpy as np
import pandas as pd

from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    MS1_MISSING_PPM_ERROR,
    MS1_OFFSET_MIN_CANDIDATES,
    ms1_reference,
    recenter_ms1_mass_error,
)


def test_ms1_reference_gives_a_decoy_its_targets_mass_and_envelope():
    precursor_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "elution_group_idx": [5, 5, 9],
            "decoy": [0, 1, 1],
            "charge": [2, 2, 3],
            "mz_library": [500.0, 507.0, 610.0],
            "i_0": [1.0, 1.0, 1.0],
            "i_1": [0.6, 0.7, 0.8],
            "i_2": [0.2, 0.3, 0.4],
            "i_3": [0.05, 0.1, 0.1],
        }
    )

    reference = ms1_reference(precursor_df, "mz_library")

    assert reference["mz_library"].tolist() == [500.0, 500.0, 610.0]
    assert reference["i_1"].tolist() == [0.6, 0.6, 0.8]
    assert reference["charge"].tolist() == [2, 2, 3]


def test_recenter_ms1_mass_error_removes_the_run_offset():
    rng = np.random.default_rng(0)
    n = 10 * MS1_OFFSET_MIN_CANDIDATES
    features_df = pd.DataFrame(
        {
            "score": rng.uniform(0, 1, n),
            "ms1_n_isotopes": 4.0,
            "ms1_log_m0_apex": 3.0,
            "ms1_ppm_signed": 2.0 + rng.normal(0, 0.1, n),
        }
    )
    features_df.loc[0, "ms1_log_m0_apex"] = 0.0

    error = recenter_ms1_mass_error(features_df)

    assert error.iloc[0] == MS1_MISSING_PPM_ERROR
    assert error.iloc[1:].median() < 0.2

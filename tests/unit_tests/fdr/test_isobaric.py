import pandas as pd

from alphadia.fdr.fdr import isobaric_losing_groups


def _psms(**columns) -> pd.DataFrame:
    base = {
        "elution_group_idx": [0, 1, 2, 3],
        "_decoy": [0, 0, 0, 0],
        "qval": [0.0, 0.1, 0.1, 0.1],
        "mz_library": [500.0, 500.002, 500.0, 600.0],
        "rt_observed": [100.0, 101.0, 100.0, 100.0],
        "fwhm_rt": [3.0, 3.0, 3.0, 3.0],
        "charge": [2, 2, 3, 2],
        "proba": [0.99, 0.6, 0.6, 0.6],
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_the_weaker_of_two_isobaric_coeluting_targets_loses():
    assert isobaric_losing_groups(_psms()).tolist() == [1]


def test_separate_peaks_and_decoys_do_not_compete():
    assert (
        isobaric_losing_groups(_psms(rt_observed=[100.0, 110.0, 100.0, 100.0])).tolist()
        == []
    )
    assert isobaric_losing_groups(_psms(_decoy=[0, 1, 0, 0])).tolist() == []


def test_targets_outside_the_pool_do_not_compete():
    assert isobaric_losing_groups(_psms(qval=[0.0, 0.9, 0.1, 0.1])).tolist() == []


def test_candidates_of_the_same_elution_group_do_not_compete():
    psms = _psms(elution_group_idx=[0, 0, 2, 3])
    assert isobaric_losing_groups(psms).tolist() == []


def test_only_a_confident_target_removes_another():
    psms = _psms(qval=[0.05, 0.1, 0.1, 0.1])
    assert isobaric_losing_groups(psms).tolist() == []

import pandas as pd

from alphadia.fdr import gap_trace
from alphadia.fdr.gap_trace import GapTrace, precursor_key


def test_precursor_key_matches_diann_layout():
    assert (
        precursor_key("ACMK", "Carbamidomethyl@C;Oxidation@M", "2;3", 2)
        == "ACMK|2:4,3:35|2"
    )
    assert precursor_key("PEPK", "", "", 3) == "PEPK||3"


def test_init_traces_matched_groups_and_background(tmp_path, monkeypatch):
    keys_path = tmp_path / "gap_keys.parquet"
    pd.DataFrame(
        {
            "key": ["PEPK||2"],
            "diann_rt": [60.0],
            "diann_q": [0.001],
            "diann_only": [True],
        }
    ).to_parquet(keys_path)
    monkeypatch.setattr(gap_trace, "GAP_KEYS_PATH", keys_path)
    monkeypatch.setattr(gap_trace, "SAMPLE_GROUP_MODULUS", 100)
    library = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4],
            "elution_group_idx": [7, 7, 8, 8, 100],
            "decoy": [0, 1, 0, 1, 0],
            "sequence": ["PEPK", "PAPK", "OTHERK", "OTHAK", "BGK"],
            "mods": ["", "", "", "", ""],
            "mod_sites": ["", "", "", "", ""],
            "charge": [2, 2, 2, 2, 2],
        }
    )
    trace = GapTrace()

    traced = trace.init(library)

    assert sorted(traced["precursor_idx"]) == [0, 1, 4]
    assert traced.set_index("precursor_idx").loc[0, "diann_only"]

    trace.record("final", pd.DataFrame({"precursor_idx": [0, 2], "proba": [0.1, 0.2]}))
    trace.write(str(tmp_path))
    assert pd.read_parquet(tmp_path / "gap_final.parquet")[
        "precursor_idx"
    ].tolist() == [0]


def test_record_is_noop_without_init():
    trace = GapTrace()
    trace.record("final", pd.DataFrame({"precursor_idx": [0]}))
    assert trace.stages == {}

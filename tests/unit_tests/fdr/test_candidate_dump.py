"""Unit tests for the diagnostic files of the final extraction round."""

import numpy as np
import pandas as pd
import pytest

from alphadia.constants.keys import CalibCols
from alphadia.fdr import candidate_dump, fdr, riding


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precursor_idx": np.array([3, 3, 7, 9], dtype=np.uint64),
            "rank": np.array([0, 1, 0, 0], dtype=np.uint64),
            "score": np.array([12.0, 8.0, 8.0, 3.0], dtype=np.float32),
            "frame_center": [0, 4, 8, 100],
            "frame_start": [0, 0, 4, 96],
            "selection_det_share": np.array([0.1, 0.9, 0.5, 0.2], dtype=np.float32),
            "selection_n_detected": np.array([2, 9, 5, 1], dtype=np.float32),
            "elution_group_idx": [1, 1, 2, 3],
            "decoy": [0, 0, 1, 0],
        }
    )


def test_write_candidates_keeps_every_candidate_with_its_cutoff_decision(tmp_path):
    # given: four candidates, the first above the cutoff, the second admitted by the detection share
    candidates_df = _candidates()
    above_cutoff = np.array([True, False, False, False])
    kept = np.array([True, True, False, False])
    rt_values = np.arange(12, dtype=np.float64) * 1.5

    # when
    candidate_dump.write_candidates(
        str(tmp_path),
        candidates_df,
        above_cutoff,
        kept,
        rt_values,
        {
            candidate_dump.SCORE_CUTOFF_COLUMN: 10.0,
            candidate_dump.NUM_CANDIDATES_COLUMN: 3,
        },
    )

    # then: every candidate is written in its order, with compact identity columns
    table = pd.read_parquet(tmp_path / candidate_dump.CANDIDATES_FILE_NAME)
    assert table["precursor_idx"].tolist() == [3, 3, 7, 9]
    assert table["precursor_idx"].dtype == np.uint32
    assert table["rank"].dtype == np.uint8
    assert table["decoy"].tolist() == [0, 0, 1, 0]
    assert table["selection_det_share"].tolist() == pytest.approx([0.1, 0.9, 0.5, 0.2])
    assert "frame_start" not in table.columns

    # and: the apex RT is the RT of the apex frame, clipped to the last spectrum
    assert table[candidate_dump.RT_OBSERVED_COLUMN].tolist() == [0.0, 6.0, 12.0, 16.5]

    # and: the cutoff decision and the settings
    assert table[candidate_dump.ABOVE_CUTOFF_COLUMN].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert table[candidate_dump.GATE_ADMITTED_COLUMN].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert (table[candidate_dump.SCORE_CUTOFF_COLUMN] == 10.0).all()
    assert (table[candidate_dump.NUM_CANDIDATES_COLUMN] == 3).all()


def test_write_riding_without_a_removal_marks_no_row(tmp_path):
    # given: final-round rows that the riding removal did not touch
    psm_df = pd.DataFrame(
        {"precursor_idx": [5, 6], "rank": [0, 2], "decoy": [0, 1], "extra": [1.0, 2.0]}
    )

    # when
    candidate_dump.write_riding(str(tmp_path), psm_df, np.array([0.2, 0.7]), None)

    # then: identity, pre-score and an empty removal result
    table = pd.read_parquet(tmp_path / candidate_dump.RIDING_FILE_NAME)
    assert table.columns.tolist() == [
        "precursor_idx",
        "rank",
        "decoy",
        candidate_dump.PRESCORE_COLUMN,
        *candidate_dump.RIDING_COLUMNS,
    ]
    assert table[candidate_dump.PRESCORE_COLUMN].tolist() == pytest.approx([0.2, 0.7])
    assert not table[candidate_dump.RIDING_REMOVED_COLUMN].any()
    assert not table[candidate_dump.RIDING_PARTNER_COLUMN].any()
    assert table[candidate_dump.RIDING_SHARE_COLUMN].isna().all()


def test_write_library_precursors_writes_the_available_identity_columns(tmp_path):
    # given: a library precursor table with an unwanted column and without genes
    precursor_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1],
            "elution_group_idx": [0, 0],
            "decoy": [0, 1],
            "sequence": ["PEPTIDEK", "PETPIDEK"],
            "mods": ["", ""],
            "charge": [2, 2],
            "proteins": ["P1", "P1"],
            "flat_frag_start_idx": [0, 10],
        }
    )

    # when
    candidate_dump.write_library_precursors(str(tmp_path), precursor_df)

    # then
    table = pd.read_parquet(tmp_path / candidate_dump.LIBRARY_PRECURSORS_FILE_NAME)
    assert table.columns.tolist() == [
        "precursor_idx",
        "elution_group_idx",
        "decoy",
        "sequence",
        "mods",
        "charge",
        "proteins",
    ]
    assert table["sequence"].tolist() == ["PEPTIDEK", "PETPIDEK"]


def _riding_psms() -> pd.DataFrame:
    # three elution groups of one target each and a decoy group; rows 1 and 2 will ride
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3],
            riding.SEQUENCE_ID_COLUMN: [0, 1, 2, 3],
            "rank": [0, 0, 0, 0],
            "elution_group_idx": [0, 1, 2, 3],
            "channel": [0, 0, 0, 0],
            "fwhm_rt": [2.0] * 4,
            CalibCols.RT_OBSERVED: [10.0] * 4,
            CalibCols.MZ_OBSERVED: [500.0] * 4,
            "decoy": [0, 0, 0, 1],
            "_decoy": [0, 0, 0, 1],
            fdr.DECOY_WEIGHT_COLUMN: [0.0, 0.0, 0.0, 1.0],
        }
    )


def test_riding_rows_report_share_and_removal_of_every_row(monkeypatch):
    # given: the riding kernel reports a large share for rows 1 and 2, but row 2 has full own support
    psm_df = _riding_psms()
    proba = np.array([0.01, 0.4, 0.5, 0.9])
    monkeypatch.setattr(
        fdr.riding,
        "find_riding",
        lambda rows, fragments, partner, cycle, tol: (
            np.array([0.0, 0.5, 0.5, 0.0]),
            np.array([1.0, 0.2, 2.0, 1.0]),
        ),
    )

    # when
    rows = fdr._riding_rows(
        psm_df, proba, ["elution_group_idx"], lambda df: None, np.zeros(1), 7.0
    )

    # then: only row 1 is removed, and the kept positions follow from it
    assert rows.columns.tolist() == candidate_dump.RIDING_COLUMNS
    assert rows[candidate_dump.RIDING_REMOVED_COLUMN].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert rows[candidate_dump.RIDING_SHARE_COLUMN].tolist() == pytest.approx(
        [0.0, 0.5, 0.5, 0.0]
    )
    assert rows[candidate_dump.RIDING_PARTNER_COLUMN].iloc[0]
    assert fdr._kept_rows(rows).tolist() == [0, 2, 3]
    assert fdr._without_riding(
        psm_df, proba, ["elution_group_idx"], lambda df: None, np.zeros(1), 7.0
    ).tolist() == [0, 2, 3]


def test_kept_rows_is_none_when_nothing_is_removed():
    # given
    rows = candidate_dump.empty_riding_rows(3)

    # then
    assert fdr._kept_rows(rows) is None
    assert fdr._kept_rows(None) is None


def test_dump_riding_writes_the_features_of_the_removed_rows(tmp_path):
    # given: two final-round rows, the second removed
    psm_df = pd.DataFrame(
        {"precursor_idx": [0, 1], "rank": [0, 1], "decoy": [0, 0], "channel": [0, 0]}
    )
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # noqa: N806
    riding_rows = candidate_dump.empty_riding_rows(2)
    riding_rows[candidate_dump.RIDING_REMOVED_COLUMN] = [False, True]

    # when
    fdr._dump_riding(
        str(tmp_path),
        psm_df,
        X,
        ["a", "b"],
        np.array([2, 3]),
        np.array([0.1, 0.6]),
        riding_rows,
    )

    # then: every row in the riding file, only the removed one with its features and pre-score
    table = pd.read_parquet(tmp_path / candidate_dump.RIDING_FILE_NAME)
    assert table[candidate_dump.RIDING_REMOVED_COLUMN].tolist() == [False, True]
    removed = pd.read_parquet(candidate_dump.removed_features_path(str(tmp_path)))
    assert removed["precursor_idx"].tolist() == [1]
    assert removed[["a", "b"]].to_numpy().tolist() == [[3.0, 4.0]]
    assert removed["fold"].tolist() == [3]
    assert removed["proba"].tolist() == pytest.approx([0.6])

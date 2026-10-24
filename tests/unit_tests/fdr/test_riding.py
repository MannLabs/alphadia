import numpy as np
import pandas as pd
import pytest

from alphadia.fdr import riding

WINNER_FRAGMENTS = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
CYCLE = np.array([[[[-1.0, -1.0]], [[499.0, 501.0]], [[501.0, 503.0]]]])


def _rows(spec: list[dict]) -> pd.DataFrame:
    defaults = {
        "rank": 0,
        "fwhm_rt": 2.0,
        "rt_observed": 100.0,
        "mz_observed": 500.0,
        "_decoy": 0,
    }
    rows = pd.DataFrame([defaults | s for s in spec])
    rows["elution_group_idx"] = rows["precursor_idx"]
    sequence = rows.get(riding.SEQUENCE_ID_COLUMN, rows["precursor_idx"])
    rows[riding.SEQUENCE_ID_COLUMN] = sequence.fillna(rows["precursor_idx"]).astype(int)
    return rows


def _fragments(mz, intensity, correlation) -> list[np.ndarray]:
    """One row of dense fragment arrays (library intensity 1), padded to TOP_K_FRAGMENTS."""
    out = []
    for values in (mz, np.ones(len(mz)), intensity, correlation):
        row = np.zeros(riding.TOP_K_FRAGMENTS)
        row[: len(values)] = values
        out.append(row)
    return out


def _winner_and(other: dict, other_fragments: list[np.ndarray]):
    rows = _rows([{"precursor_idx": 0, "proba": 0.01}, other])
    winner = _fragments(WINNER_FRAGMENTS, np.full(6, 1000.0), np.ones(6))
    fragments = tuple(np.stack([w, o]) for w, o in zip(winner, other_fragments))
    partner = np.array([True, False])
    return riding.find_riding(rows, fragments, partner, CYCLE, 10.0)


def test_fragment_evidence_caps_at_library_prediction():
    # given: three coherent fragments at scale 100 and one with a foreign peak, one not co-eluting
    observed = np.array([[100.0, 100.0, 100.0, 5000.0, 100.0]])
    correlation = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]])
    library = np.ones((1, 5))

    # when
    evidence = riding.fragment_evidence(observed, correlation, library)

    # then: the foreign peak is capped at the library prediction, the non co-eluting fragment counts nothing
    scale = (100.0 * 3 + 5000.0) / 4
    np.testing.assert_allclose(evidence[0], [100.0, 100.0, 100.0, scale, 0.0])


def test_rider_is_explained_by_the_better_partner():
    # given: a candidate whose only signal is on two fragments of the winner, its own fragments are noise
    rider = _fragments(
        [400.0, 600.0, 450.0, 550.0],
        [800.0, 800.0, 10.0, 10.0],
        [1.0, 1.0, 0.0, 0.0],
    )

    # when
    share, own = _winner_and({"precursor_idx": 1, "proba": 0.4}, rider)

    # then
    assert share[1] == pytest.approx(1.0)
    assert own[1] == pytest.approx(0.0)
    assert share[0] == 0


def test_real_co_eluter_keeps_its_own_evidence():
    # given: a candidate sharing one fragment with the winner but with five coherent fragments of its own
    real = _fragments(
        [400.0, 410.0, 420.0, 430.0, 440.0, 450.0],
        [300.0, 300.0, 300.0, 300.0, 300.0, 300.0],
        np.ones(6),
    )

    # when
    share, own = _winner_and({"precursor_idx": 1, "proba": 0.4}, real)

    # then
    assert share[1] == pytest.approx(1 / 6)
    assert own[1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "other",
    [
        {"precursor_idx": 1, "proba": 0.4, "rt_observed": 120.0},
        {"precursor_idx": 1, "proba": 0.4, riding.SEQUENCE_ID_COLUMN: 0},
        {"precursor_idx": 1, "proba": 0.001},
        {"precursor_idx": 1, "proba": 0.4, "mz_observed": 502.0},
    ],
    ids=["not co-eluting", "same sequence", "better than the partner", "other window"],
)
def test_no_partner_explains(other):
    # given
    rider = _fragments([400.0, 600.0], [800.0, 800.0], [1.0, 1.0])

    # when
    share, _ = _winner_and(other, rider)

    # then
    assert share[1] == 0


def test_within_class_keeps_decoys_from_target_partners(monkeypatch):
    # given: a decoy riding on a target
    rider = _fragments([400.0, 600.0], [800.0, 800.0], [1.0, 1.0])
    other = {"precursor_idx": 1, "proba": 0.4, "_decoy": 1}

    # when
    blind, _ = _winner_and(other, rider)
    monkeypatch.setattr(riding, "CLASS_BLIND", False)
    within, _ = _winner_and(other, rider)

    # then
    assert blind[1] == pytest.approx(1.0)
    assert within[1] == 0


def test_isolation_window_skips_ms1_and_marks_outside():
    # when
    window = riding.isolation_window(np.array([500.0, 502.0, 600.0]), CYCLE)

    # then
    assert window.tolist() == [0, 1, -1]

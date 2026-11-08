import os

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from alphadia.fdr import fdr, joint_features, joint_rescoring

CONFIG_FDR = {
    "fdr": 0.01,
    "enable_nn_hyperparameter_tuning": True,
    "feature_transform": "quantile",
}
GROUP_COLUMNS = ["elution_group_idx", "channel"]
N_GROUPS = 600
N_REAL = 350
FEATURES = ["signal", "noise", "naa"]


def _run_rows(rng: np.random.Generator) -> pd.DataFrame:
    """A target and a decoy per elution group, the first N_REAL targets real, and a per-run probability that ranks
    them noisily."""
    groups = np.arange(N_GROUPS)
    real = groups < N_REAL
    target = pd.DataFrame(
        {
            "precursor_idx": groups * 2,
            "elution_group_idx": groups,
            "channel": 0,
            "decoy": 0,
            "rank": 0,
            "naa": 7 + groups % 6,
            "signal": np.where(
                real, rng.normal(4, 1, N_GROUPS), rng.normal(0, 1, N_GROUPS)
            ),
            "noise": rng.normal(0, 1, N_GROUPS),
        }
    )
    decoy = target.assign(
        precursor_idx=groups * 2 + 1, decoy=1, signal=rng.normal(0, 1, N_GROUPS)
    )
    rows = pd.concat([target, decoy], ignore_index=True)
    rows["proba"] = expit(-rows["signal"] + rng.normal(0, 1.5, len(rows)))
    return rows


def _write_run(folder, rng: np.random.Generator, lost: np.ndarray | None = None):
    """Write the run's joint feature files and return its PSM table: the competition winners up to q 5 %."""
    os.makedirs(folder)
    rows = _run_rows(rng)
    joint_features.write_run_features(
        str(folder),
        rows,
        rows[FEATURES].to_numpy(),
        FEATURES,
        rows["proba"].to_numpy(),
    )
    if lost is not None:
        joint_features.write_fragcomp_lost(str(folder), lost)
    best = fdr.get_q_values_by_length(
        fdr.keep_best(rows, group_columns=GROUP_COLUMNS), "proba", "decoy"
    )
    return best[best["qval"] <= joint_rescoring.KEEP_QVAL].assign(
        run=os.path.basename(folder), intensity=1.0
    )


def test_run_features_round_trip(tmp_path):
    # given: rows whose feature naa is also an identity column
    rows = _run_rows(np.random.default_rng(0))

    # when
    joint_features.write_run_features(
        str(tmp_path), rows, rows[FEATURES].to_numpy(), FEATURES, rows["proba"]
    )
    joint_features.write_fragcomp_lost(str(tmp_path), np.arange(len(rows)) == 3)

    # then: the feature columns come back in order as float32, the identity columns with the probability and the
    # fragment competition outcome
    assert joint_features.feature_columns(str(tmp_path)) == FEATURES
    x = joint_features.read_features(str(tmp_path), FEATURES, np.array([1, 5]))
    assert x.dtype == np.float32
    np.testing.assert_allclose(
        x, rows[FEATURES].to_numpy(np.float32)[[1, 5]], rtol=1e-6
    )
    ids = joint_features.read_ids(str(tmp_path))
    np.testing.assert_array_equal(ids["precursor_idx"], rows["precursor_idx"])
    np.testing.assert_array_equal(ids["naa"], rows["naa"])
    np.testing.assert_allclose(ids["proba"], rows["proba"])
    assert ids["lost"].tolist() == (np.arange(len(rows)) == 3).tolist()


def test_read_ids_without_fragment_competition_marks_no_row_lost(tmp_path):
    # given
    rows = _run_rows(np.random.default_rng(0))
    joint_features.write_run_features(
        str(tmp_path), rows, rows[FEATURES].to_numpy(), FEATURES, rows["proba"]
    )

    # when
    ids = joint_features.read_ids(str(tmp_path))

    # then
    assert not ids["lost"].any()


@pytest.mark.parametrize(
    ("n_runs", "n_groups"), [(2, 2), (8, 8), (9, joint_rescoring.MAX_LOFO_RUNS)]
)
def test_run_groups_leave_one_run_out_up_to_the_maximum(n_runs, n_groups):
    # when
    groups = joint_rescoring.run_groups(n_runs)

    # then
    assert len(groups) == n_runs
    assert len(np.unique(groups)) == n_groups


def test_training_rows_keep_every_row_below_the_maximum():
    # given
    competitions = [np.repeat(np.arange(50), 2) for _ in range(3)]

    # when
    rows = joint_rescoring.training_rows(
        competitions, [0, 2], np.random.default_rng(0), max_rows=200
    )

    # then
    assert sorted(rows) == [0, 2]
    assert all(len(r) == 100 for r in rows.values())


def test_training_rows_sample_whole_competition_groups_above_the_maximum():
    # given: two runs of 5000 groups of a target and a decoy
    competitions = [np.repeat(np.arange(5000), 2) for _ in range(2)]

    # when: at most half of the rows may train
    rows = joint_rescoring.training_rows(
        competitions, [0, 1], np.random.default_rng(0), max_rows=10_000
    )

    # then: about half of every run's groups, each with both of its rows
    for run, r in rows.items():
        drawn = competitions[run][r]
        assert 0.45 < len(np.unique(drawn)) / 5000 < 0.55
        assert (np.bincount(drawn) % 2 == 0).all()


def test_ensemble_proba_is_the_sigmoid_of_the_mean_logit():
    # given
    run_proba = np.array([0.1, 0.5, 0.9])
    joint_proba = np.array([0.3, 0.2, 0.9])

    # when
    proba = joint_rescoring.ensemble_proba(run_proba, joint_proba)

    # then
    def logit(p):
        return np.log(p / (1 - p))

    np.testing.assert_allclose(
        proba, expit((logit(run_proba) + logit(joint_proba)) / 2)
    )


def test_rescore_leaves_every_run_out_of_its_own_model(tmp_path, monkeypatch):
    # given: three runs with joint features
    rng = np.random.default_rng(0)
    folders = [str(tmp_path / f"run_{i}") for i in range(3)]
    psm_df = pd.concat([_write_run(f, rng) for f in folders], ignore_index=True)
    calls = []

    def fit_predict(folders, ids, rows, left_out, columns, config_fdr, random_state):
        calls.append((set(rows), set(left_out)))
        return {i: ids[i]["proba"].to_numpy() for i in left_out}

    monkeypatch.setattr(joint_rescoring, "_fit_predict_left_out", fit_predict)

    # when
    joint_rescoring.rescore(psm_df, folders, CONFIG_FDR, GROUP_COLUMNS, 0)

    # then: one model per run, trained on the other runs only
    assert [left_out for _, left_out in calls] == [{0}, {1}, {2}]
    for train, left_out in calls:
        assert train == {0, 1, 2} - left_out


def test_rescore_accepts_the_real_precursors_and_keeps_runs_without_features(
    tmp_path,
):
    # given: three runs with joint features, the first with the targets of groups 0-19 lost to fragment competition,
    # and a run without joint features whose PSMs must stay as they are
    rng = np.random.default_rng(0)
    lost = np.zeros(2 * N_GROUPS, dtype=bool)
    lost[:20] = True
    folders = [str(tmp_path / f"run_{i}") for i in range(3)]
    psm_df = pd.concat(
        [_write_run(f, rng, lost if i == 0 else None) for i, f in enumerate(folders)],
        ignore_index=True,
    )
    other = pd.DataFrame(
        {
            "precursor_idx": [10_000],
            "elution_group_idx": [10_000],
            "channel": [0],
            "decoy": [0],
            "rank": [0],
            "run": ["other"],
            "proba": [0.4],
            "qval": [0.5],
        }
    )
    os.makedirs(tmp_path / "other")
    psm_df = pd.concat([psm_df, other], ignore_index=True)

    # when
    out, rescored = joint_rescoring.rescore(
        psm_df, [*folders, str(tmp_path / "other")], CONFIG_FDR, GROUP_COLUMNS, 0
    )

    # then: the run without features is kept as it was, the others are rescored
    assert rescored == folders
    assert out[out["run"] == "other"]["qval"].tolist() == [0.5]
    for folder in folders:
        run = os.path.basename(folder)
        run_out = out[out["run"] == run]
        assert (run_out["qval"] <= CONFIG_FDR["fdr"]).all()
        # only PSMs the run quantified are reported, with their other columns
        assert set(run_out["precursor_idx"]) <= set(
            psm_df[psm_df["run"] == run]["precursor_idx"]
        )
        assert (run_out["intensity"] == 1.0).all()
        # the joint model sees the signal the noisy per-run score blurs
        run_psm = psm_df[psm_df["run"] == run]
        per_run = (run_psm["qval"] <= CONFIG_FDR["fdr"]) & (run_psm["decoy"] == 0)
        real = (run_out["decoy"] == 0) & (run_out["elution_group_idx"] < N_REAL)
        assert real.sum() > 1.3 * per_run.sum()
        assert (run_out["decoy"] == 0).sum() - real.sum() <= 0.05 * real.sum()
    # and a target that lost its fragments is not reported
    run_0 = out[out["run"] == "run_0"]
    assert not run_0["precursor_idx"].isin(np.arange(20) * 2).any()


def test_rescore_of_a_single_run_keeps_its_per_run_q_values(tmp_path):
    # given: one run with joint features, its PSMs kept up to q 5 %
    rng = np.random.default_rng(0)
    folder = str(tmp_path / "run_0")
    psm_df = _write_run(folder, rng)
    assert (psm_df["qval"] > CONFIG_FDR["fdr"]).any()

    # when
    out, rescored = joint_rescoring.rescore(
        psm_df, [folder], CONFIG_FDR, GROUP_COLUMNS, 0
    )

    # then: filtered at the configured FDR by their own q-values
    assert rescored == [folder]
    expected = psm_df[psm_df["qval"] <= CONFIG_FDR["fdr"]]
    assert sorted(out["precursor_idx"]) == sorted(expected["precursor_idx"])


def test_rescore_without_joint_features_returns_the_psms_unchanged(tmp_path):
    # given
    psm_df = pd.DataFrame({"run": ["a"], "qval": [0.5], "precursor_idx": [1]})
    os.makedirs(tmp_path / "a")

    # when
    out, rescored = joint_rescoring.rescore(
        psm_df, [str(tmp_path / "a")], CONFIG_FDR, GROUP_COLUMNS, 0
    )

    # then
    assert rescored == []
    pd.testing.assert_frame_equal(out, psm_df)


def test_rescored_run_computes_q_values_within_length_strata():
    # given: a run whose short stratum (naa <= 8) has its decoy at the top, its long stratum none
    ids = pd.DataFrame(
        {
            "precursor_idx": np.arange(6),
            "elution_group_idx": np.arange(6),
            "channel": 0,
            "decoy": [1, 0, 0, 0, 0, 0],
            "rank": 0,
            "naa": [7, 7, 7, 12, 12, 12],
            "proba": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "lost": False,
        }
    )
    competition = np.arange(6)

    # when: the joint model ranks the rows as the run did
    best = joint_rescoring._rescored_run(
        ids, competition, ids["proba"].to_numpy(), GROUP_COLUMNS
    )

    # then: the long stratum is not charged the short stratum's decoy
    q = best.set_index("precursor_idx")["qval"]
    assert (q[[3, 4, 5]] == 0).all()
    assert (q[[1, 2]] > 0).all()


def test_is_data_limited_counts_competed_targets(monkeypatch):
    # given: 5 groups, each with a target ranked before its decoy, so all 5 targets pass at q 0
    groups = np.repeat(np.arange(5), 2)
    decoy = np.tile([False, True], 5)
    proba = np.where(decoy, 0.9, 0.1) + np.repeat(np.arange(5), 2) * 1e-3

    # when / then: at or below the target limit the run is data-limited, above it not
    monkeypatch.setattr(joint_features, "DATA_LIMITED_MAX_TARGETS", 5)
    assert joint_features.is_data_limited(groups, decoy, proba)
    monkeypatch.setattr(joint_features, "DATA_LIMITED_MAX_TARGETS", 4)
    assert not joint_features.is_data_limited(groups, decoy, proba)


def test_is_data_limited_ignores_targets_beyond_the_qvalue(monkeypatch):
    # given: 100 groups whose decoys outrank their targets, so no target passes q 1 %
    groups = np.repeat(np.arange(100), 2)
    decoy = np.tile([False, True], 100)
    proba = np.where(decoy, 0.1, 0.9)

    # when / then: even a limit of 0 targets is met
    monkeypatch.setattr(joint_features, "DATA_LIMITED_MAX_TARGETS", 0)
    assert joint_features.is_data_limited(groups, decoy, proba)

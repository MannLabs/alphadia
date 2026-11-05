import numpy as np
import pandas as pd

from alphadia.fdr import fdr, mbr_transfer

CONFIG_FDR = {
    "fdr": 0.01,
    "enable_nn_hyperparameter_tuning": True,
    "feature_transform": "quantile",
}
N_GROUPS = 400
N_REAL = 250


def _write_run(folder, run_index: int, rng: np.random.Generator) -> pd.DataFrame:
    """Transfer features of one run (a target and a decoy per group, the first N_REAL targets real) and its
    competition winners as the PSM table holds them."""
    groups = np.arange(N_GROUPS)
    real = groups < N_REAL
    target = pd.DataFrame(
        {
            "precursor_idx": groups * 2,
            "elution_group_idx": groups,
            "rank": 0,
            "decoy": 0,
            "naa": 12,
            "signal": np.where(
                real, rng.normal(4, 1, N_GROUPS), rng.normal(0, 1, N_GROUPS)
            ),
            "noise": rng.normal(0, 1, N_GROUPS),
        }
    )
    decoy = target.assign(
        precursor_idx=groups * 2 + 1,
        decoy=1,
        signal=rng.normal(0, 1, N_GROUPS),
    )
    features = pd.concat([target, decoy], ignore_index=True)
    features[fdr.KEPT_DECOY_COLUMN] = np.int8(0)
    features[fdr.TRANSFER_COLUMN] = np.int8(1)
    folder.mkdir()
    features.to_parquet(folder / mbr_transfer.TRANSFER_FEATURES_FILE_NAME)
    winners = features.loc[
        features.groupby("elution_group_idx")["signal"].idxmax(),
        ["precursor_idx", "elution_group_idx", "rank", "decoy", "naa"],
    ]
    return winners.assign(run=folder.name, proba=0.5, qval=0.5)


def test_rescore_transfers_accepts_the_real_transfers_and_keeps_re_found_rows(tmp_path):
    # given: two runs of transfers, of which N_REAL groups are real, and one re-found PSM
    rng = np.random.default_rng(0)
    folders = [tmp_path / f"run_{i}" for i in range(2)]
    transfers = pd.concat(
        [_write_run(folder, i, rng) for i, folder in enumerate(folders)],
        ignore_index=True,
    )
    refound = pd.DataFrame(
        {
            "precursor_idx": [10_000],
            "elution_group_idx": [10_000],
            "rank": [0],
            "decoy": [0],
            "naa": [12],
            "run": ["run_0"],
            "proba": [0.001],
            "qval": [0.0],
        }
    )
    psm_df = pd.concat([refound, transfers], ignore_index=True)
    transfer = np.r_[False, np.ones(len(transfers), dtype=bool)]

    # when
    out = mbr_transfer.rescore_transfers(
        psm_df,
        transfer,
        [str(f) for f in folders],
        CONFIG_FDR,
        decoy_multiplicity=1,
        random_state=0,
    )

    # then: the re-found PSM is kept as it was, the accepted transfers are real and pass the FDR
    assert (out["precursor_idx"] == 10_000).sum() == 1
    accepted = out[out["precursor_idx"] != 10_000]
    assert (accepted["qval"] <= CONFIG_FDR["fdr"]).all()
    real_targets = (accepted["decoy"] == 0) & (accepted["elution_group_idx"] < N_REAL)
    assert real_targets.sum() > N_REAL
    assert (
        accepted["decoy"] == 0
    ).sum() - real_targets.sum() <= 0.05 * real_targets.sum()


def test_restrict_run_outputs_keeps_the_fragments_of_passing_psms(tmp_path):
    # given: fragments of three precursors, two of which passed
    pd.DataFrame(
        {"precursor_idx": [1, 1, 2, 3], "intensity": [1.0, 2.0, 3.0, 4.0]}
    ).to_parquet(tmp_path / "frag.parquet")
    run_psm_df = pd.DataFrame({"precursor_idx": [1, 3], "qval": [0.0, 0.0]})

    # when
    mbr_transfer.restrict_run_outputs(
        str(tmp_path), run_psm_df, "psm.parquet", "frag.parquet"
    )

    # then
    frag = pd.read_parquet(tmp_path / "frag.parquet")
    assert sorted(frag["precursor_idx"]) == [1, 1, 3]
    assert len(pd.read_parquet(tmp_path / "psm.parquet")) == 2

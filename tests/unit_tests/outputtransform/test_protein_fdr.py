import numpy as np
import pandas as pd

from alphadia.outputtransform.protein_fdr import perform_protein_fdr

N_TRUE_TARGET_GROUPS = 1800
# the false targets and the decoys are drawn from one distribution, so the decoys stand in for
# them one for one and the target-decoy ratio is a meaningful estimate
N_FALSE_TARGET_GROUPS = 200
N_DECOY_GROUPS = 200
FDR_THRESHOLD = 0.05
RANDOM_STATE = 7


def _groups(
    prefix: str, n: int, decoy: int, rng: np.random.Generator, true_protein: bool
):
    rows = []
    for i in range(n):
        n_precursors = rng.integers(3, 9) if true_protein else rng.integers(1, 3)
        proba = (
            rng.uniform(0.0, 0.05, n_precursors)
            if true_protein
            else rng.uniform(0.2, 1.0, n_precursors)
        )
        for j in range(n_precursors):
            rows.append(
                {
                    "pg": f"{prefix}{i}",
                    "genes": f"{prefix}{i}",
                    "proteins": f"{prefix}{i}",
                    "decoy": decoy,
                    "precursor_idx": len(rows),
                    "sequence": f"{prefix}{i}_{j}",
                    "run": "run_0",
                    "proba": proba[j],
                }
            )
    return rows


def test_protein_q_values_are_the_plain_target_decoy_ratio():
    # given
    rng = np.random.default_rng(RANDOM_STATE)
    psm_df = pd.DataFrame(
        _groups("T", N_TRUE_TARGET_GROUPS, 0, rng, True)
        + _groups("F", N_FALSE_TARGET_GROUPS, 0, rng, False)
        + _groups("D", N_DECOY_GROUPS, 1, rng, False)
    )

    # when
    scored = perform_protein_fdr(psm_df, figure_path=None)

    # then
    groups = scored.drop_duplicates(["pg", "decoy"])
    accepted = groups[groups["pg_qval"] <= FDR_THRESHOLD]
    n_targets = int((accepted["decoy"] == 0).sum())
    n_decoys = int((accepted["decoy"] == 1).sum())

    assert 0 < len(accepted) < len(groups)
    # the q-value of the last accepted group is the decoy-to-target ratio at that cut, so the
    # accepted set can never exceed the threshold
    assert n_decoys / n_targets <= FDR_THRESHOLD
    # and it is the largest such set: one more decoy would push the ratio over the threshold,
    # so the realised ratio sits within one decoy of it. Any factor applied to the q-values after
    # the fact breaks this.
    assert n_decoys / n_targets >= FDR_THRESHOLD - 2 / n_targets

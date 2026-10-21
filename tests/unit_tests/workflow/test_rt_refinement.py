import numpy as np
import pandas as pd

from alphadia.workflow.peptidecentric.rt_refinement import refine_calibrated_rt

N_PEPTIDES = 2000
SECONDS_PER_TRYPTOPHAN = 3.0


def _library(rng: np.random.Generator) -> pd.DataFrame:
    residues = np.array(list("ACDEFGHIKLMNPQRSTVY"))
    sequences = [
        "".join(rng.choice(residues, rng.integers(7, 15)))
        + ("W" * rng.integers(0, 3))
        + "K"
        for _ in range(N_PEPTIDES)
    ]
    targets = pd.DataFrame(
        {
            "precursor_idx": np.arange(N_PEPTIDES),
            "elution_group_idx": np.arange(N_PEPTIDES),
            "decoy": 0,
            "sequence": sequences,
            "mods": "",
            "charge": 2,
            "rt_calibrated": rng.uniform(0, 600, N_PEPTIDES),
        }
    )
    decoys = targets.assign(
        precursor_idx=targets["precursor_idx"] + N_PEPTIDES,
        decoy=1,
        sequence=targets["sequence"].str[::-1],
    )
    return pd.concat([targets, decoys], ignore_index=True)


def test_refine_calibrated_rt_learns_a_sequence_dependent_error():
    rng = np.random.default_rng(0)
    library = _library(rng)
    targets = library[library["decoy"] == 0]
    true_rt = targets["rt_calibrated"] + SECONDS_PER_TRYPTOPHAN * targets[
        "sequence"
    ].str.count("W")
    ids = pd.DataFrame(
        {"precursor_idx": targets["precursor_idx"], "rt_observed": true_rt}
    )
    before = np.median(np.abs(true_rt - targets["rt_calibrated"]))

    refine_calibrated_rt(library, ids)

    refined = library.loc[library["decoy"] == 0, "rt_calibrated"]
    after = np.median(np.abs(true_rt.to_numpy() - refined.to_numpy()))
    assert after < 0.2 * before


def test_refine_calibrated_rt_gives_a_decoy_the_correction_of_its_target():
    rng = np.random.default_rng(1)
    library = _library(rng)
    targets = library[library["decoy"] == 0]
    ids = pd.DataFrame(
        {
            "precursor_idx": targets["precursor_idx"],
            "rt_observed": targets["rt_calibrated"]
            + SECONDS_PER_TRYPTOPHAN * targets["sequence"].str.count("W"),
        }
    )

    refine_calibrated_rt(library, ids)

    by_group = library.pivot(
        index="elution_group_idx", columns="decoy", values="rt_calibrated"
    )
    np.testing.assert_allclose(by_group[0], by_group[1])


def test_refine_calibrated_rt_skips_too_few_identifications():
    rng = np.random.default_rng(2)
    library = _library(rng)
    original = library["rt_calibrated"].copy()
    ids = pd.DataFrame({"precursor_idx": [0, 1, 2], "rt_observed": [1.0, 2.0, 3.0]})

    refine_calibrated_rt(library, ids)

    pd.testing.assert_series_equal(library["rt_calibrated"], original)


def test_refine_calibrated_rt_skips_a_residual_without_sequence_dependence():
    rng = np.random.default_rng(3)
    library = _library(rng)
    original = library["rt_calibrated"].copy()
    targets = library[library["decoy"] == 0]
    ids = pd.DataFrame(
        {
            "precursor_idx": targets["precursor_idx"],
            "rt_observed": targets["rt_calibrated"] + rng.normal(0, 5, len(targets)),
        }
    )

    refine_calibrated_rt(library, ids)

    pd.testing.assert_series_equal(library["rt_calibrated"], original)

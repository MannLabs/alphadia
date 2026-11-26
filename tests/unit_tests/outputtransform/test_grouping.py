import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform import grouping
from alphadia.outputtransform.protein_fdr import perform_protein_fdr


# helper function to construct test cases
def construct_test_cases():
    # define test cases as outlined in Nesvizhskii, Alexey I., and Ruedi Aebersold. "Interpretation of shotgun proteomic data." Molecular & cellular proteomics 4.10 (2005): 1419-1440. Figure 5
    # additional cases: complex case with multiple proteins, circular case with multiple proteins

    distinct_proteins_input = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A", "B", "B"],
        "decoy": [0, 0, 0, 0],
    }
    distinct_proteins_expected = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A", "B", "B"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["A", "A", "B", "B"],
        "pg": ["A", "A", "B", "B"],  # heuristic grouping
    }

    differentiable_proteins_input = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A;B", "A;B", "B"],
        "decoy": [0, 0, 0, 0],
    }
    differentiable_proteins_expected = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A;B", "A;B", "B"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["A", "A", "A", "B"],
        "pg": ["A", "A;B", "A;B", "B"],  # heuristic grouping
    }

    indistinguishable_proteins_input = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A;B", "A;B", "A;B", "A;B"],
        "decoy": [0, 0, 0, 0],
    }
    indistinguishable_proteins_expected = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A;B", "A;B", "A;B", "A;B"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["A", "A", "A", "A"],
        "pg": ["A", "A", "A", "A"],  # heuristic grouping
    }

    subset_proteins_input = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A;B", "A;B", "A;B"],
        "decoy": [0, 0, 0, 0],
    }
    subset_proteins_expected = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A;B", "A;B", "A;B"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["A", "A", "A", "A"],
        "pg": ["A", "A", "A", "A"],  # heuristic grouping
    }

    subsumable_proteins_input = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A;B", "B;C", "C"],
        "decoy": [0, 0, 0, 0],  # heuristic grouping
    }
    subsumable_proteins_expected = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A", "A;B", "B;C", "C"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["A", "A", "C", "C"],
        "pg": ["A", "A", "C", "C"],  # heuristic grouping
    }

    shared_only_proteins_input = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A;B", "A;B;C", "A;B;C", "A;C"],
        "decoy": [0, 0, 0, 0],  # heuristic grouping
    }
    shared_only_proteins_expected = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A;B", "A;B;C", "A;B;C", "A;C"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["A", "A", "A", "A"],
        "pg": ["A", "A", "A", "A"],  # heuristic grouping
    }

    circular_proteins_input = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A;B;C", "B;C;D", "C;D;E", "D;E;A"],
        "decoy": [0, 0, 0, 0],
    }
    circular_proteins_expected = {
        "precursor_idx": [1, 2, 3, 4],
        "proteins": ["A;B;C", "B;C;D", "C;D;E", "D;E;A"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["C", "C", "C", "A"],
        "pg": ["A;C", "C", "C", "A"],  # heuristic grouping
    }

    complex_example_proteins_input = {
        "precursor_idx": [0, 1, 2, 3],
        "proteins": ["P1;P2;P3;P4", "P1;P4", "P2", "P2;P5"],
        "decoy": [0, 0, 0, 0],
    }
    complex_example_proteins_expected = {
        "precursor_idx": [0, 1, 2, 3],
        "proteins": ["P1;P2;P3;P4", "P1;P4", "P2", "P2;P5"],
        "decoy": [0, 0, 0, 0],
        "pg_master": ["P2", "P1", "P2", "P2"],
        "pg": ["P1;P2", "P1", "P2", "P2"],  # heuristic grouping
    }

    test_cases = [
        ("distinct_proteins", distinct_proteins_input, distinct_proteins_expected),
        (
            "differentiable proteins",
            differentiable_proteins_input,
            differentiable_proteins_expected,
        ),
        (
            "indistinguishable proteins",
            indistinguishable_proteins_input,
            indistinguishable_proteins_expected,
        ),
        ("subset proteins", subset_proteins_input, subset_proteins_expected),
        (
            "subsumable proteins",
            subsumable_proteins_input,
            subsumable_proteins_expected,
        ),
        ("shared only", shared_only_proteins_input, shared_only_proteins_expected),
        ("circular", circular_proteins_input, circular_proteins_expected),
        (
            "complex example",
            complex_example_proteins_input,
            complex_example_proteins_expected,
        ),
    ]

    return test_cases


# parametrized function to evaluate test case correctness
@pytest.mark.parametrize("type, input_dict, expected_dict", construct_test_cases())
def test_grouping(
    type: str,
    input_dict: dict,
    expected_dict: dict,
):
    assert (
        grouping.perform_grouping(
            pd.DataFrame(input_dict), genes_or_proteins="proteins"
        ).to_dict(orient="list")
        == expected_dict
    )


# Perform grouping on a large dataset
def test_grouping_fuzz(expected_time: int = 10):
    # test grouping performance with dummy dataset
    n_precursors = 4000

    # generate precursor index and randomize sequence
    precursor_idx = np.random.choice(
        np.array(range(n_precursors)), n_precursors, replace=False
    )

    # generate 7500 random uniprot-like protein IDs
    fake_ids = np.array(
        [
            "P" + str(i)
            for i in np.random.choice(
                np.array(range(10000, 100000, 1)), 5000, replace=False
            )
        ]
    )

    # simulate distribution of IDs per precursor, roughly equivalent to exponential distribution observed in real data
    counts = np.int64(np.ceil(np.random.exponential(scale=10.0, size=n_precursors)))
    proteins = [";".join(np.random.choice(fake_ids, i, replace=True)) for i in counts]

    # decoys
    decoys = [
        np.random.choice(np.array([0, 1]), 1, replace=True)[0]
        for _ in range(n_precursors)
    ]

    # build dummy dataframe
    simulated_psm_data = pd.DataFrame(
        {"precursor_idx": precursor_idx, "proteins": proteins, "decoy": decoys}
    )

    _ = grouping.perform_grouping(simulated_psm_data, genes_or_proteins="proteins")
    assert True  # TODO fix this test


def test_heuristic_grouping_no_decoy_to_target_leakage():
    """Decoy master proteins must not leak into target heuristic protein groups."""
    psm_df = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3],
            "proteins": ["P1", "P1;P2", "P2"],
            "decoy": [0, 0, 1],
        }
    )

    combined = grouping.perform_grouping(
        psm_df.copy(), genes_or_proteins="proteins", group=True
    )
    targets_only = grouping.perform_grouping(
        psm_df[psm_df["decoy"] == 0].copy(), genes_or_proteins="proteins", group=True
    )

    combined_targets = combined[combined["decoy"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(combined_targets, targets_only.reset_index(drop=True))
    assert combined_targets["pg"].tolist() == ["P1", "P1"]


def test_heuristic_grouping_no_target_to_decoy_leakage():
    """Target master proteins must not leak into decoy heuristic protein groups."""
    psm_df = pd.DataFrame(
        {
            "precursor_idx": [1, 4, 5],
            "proteins": ["P1", "P3", "P3;P1"],
            "decoy": [0, 1, 1],
        }
    )

    combined = grouping.perform_grouping(
        psm_df.copy(), genes_or_proteins="proteins", group=True
    )
    decoys_only = grouping.perform_grouping(
        psm_df[psm_df["decoy"] == 1].copy(), genes_or_proteins="proteins", group=True
    )

    combined_decoys = combined[combined["decoy"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(combined_decoys, decoys_only.reset_index(drop=True))
    assert combined_decoys["pg"].tolist() == ["P3", "P3"]


@pytest.mark.parametrize("genes_or_proteins", ["proteins", "genes"])
@pytest.mark.parametrize("return_parsimony_groups", [False, True])
def test_heuristic_grouping_target_decoy_independence(
    genes_or_proteins: str, return_parsimony_groups: bool
):
    """Heuristic grouping on targets+decoys must match grouping each class alone."""
    psm_df = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3, 4, 5],
            genes_or_proteins: ["A", "A;B", "B", "C", "C;A"],
            "decoy": [0, 0, 1, 1, 1],
        }
    )

    combined = grouping.perform_grouping(
        psm_df.copy(),
        genes_or_proteins=genes_or_proteins,
        group=True,
        return_parsimony_groups=return_parsimony_groups,
    )
    targets_only = grouping.perform_grouping(
        psm_df[psm_df["decoy"] == 0].copy(),
        genes_or_proteins=genes_or_proteins,
        group=True,
        return_parsimony_groups=return_parsimony_groups,
    )
    decoys_only = grouping.perform_grouping(
        psm_df[psm_df["decoy"] == 1].copy(),
        genes_or_proteins=genes_or_proteins,
        group=True,
        return_parsimony_groups=return_parsimony_groups,
    )

    assert combined["pg"].tolist() == ["A", "A", "B", "C", "C"]
    pd.testing.assert_frame_equal(
        combined[combined["decoy"] == 0].reset_index(drop=True),
        targets_only.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        combined[combined["decoy"] == 1].reset_index(drop=True),
        decoys_only.reset_index(drop=True),
    )


def test_heuristic_grouping_preserves_protein_fdr_with_shared_peptides():
    """Shared peptides between decoy and target masters must not shatter decoy groups or bias protein FDR."""
    rows = []
    pid = 0
    for i in range(100):
        rows.append(
            {
                "precursor_idx": pid,
                "proteins": f"A_{i}",
                "decoy": 0,
                "proba": 1e-4,
                "qval": 0.001,
            }
        )
        pid += 1
        rows.append(
            {
                "precursor_idx": pid,
                "proteins": f"A_{i};B_{i}",
                "decoy": 0,
                "proba": 1e-4,
                "qval": 0.001,
            }
        )
        pid += 1

    for i in range(10):
        rows.append(
            {
                "precursor_idx": pid,
                "proteins": f"BORDER_{i}",
                "decoy": 0,
                "proba": 0.05,
                "qval": 0.005,
            }
        )
        pid += 1

    for proteins in ["D_0", "D_0;A_0", "D_0;A_1"]:
        rows.append(
            {
                "precursor_idx": pid,
                "proteins": proteins,
                "decoy": 1,
                "proba": 0.03,
                "qval": 0.004,
            }
        )
        pid += 1

    rows.append(
        {
            "precursor_idx": pid,
            "proteins": "B_0",
            "decoy": 1,
            "proba": 0.40,
            "qval": 0.009,
        }
    )

    grouped = grouping.perform_grouping(
        pd.DataFrame(rows), genes_or_proteins="proteins", group=True
    )
    scored = perform_protein_fdr(grouped, figure_path=None)
    groups = scored.drop_duplicates(["pg", "decoy"])

    assert int((groups["decoy"] == 0).sum()) == 110
    assert sorted(groups.loc[groups["decoy"] == 1, "pg"].tolist()) == ["B_0", "D_0"]
    assert int(((groups["decoy"] == 0) & (groups["pg_qval"] <= 0.01)).sum()) == 110
    assert np.isclose(
        groups.loc[groups["pg"] == "BORDER_0", "pg_qval"].iloc[0], 1.0 / 110.0
    )

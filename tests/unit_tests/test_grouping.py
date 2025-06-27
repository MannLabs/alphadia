import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform import grouping


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

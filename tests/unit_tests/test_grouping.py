import time
import pytest
import numpy as np
import pandas as pd
from alphadia import grouping

#helper function to construct test cases
def construct_test_cases():

    #define test cases as outlined in Nesvizhskii, Alexey I., and Ruedi Aebersold. "Interpretation of shotgun proteomic data." Molecular & cellular proteomics 4.10 (2005): 1419-1440. Figure 5
    #additional cases: complex case with multiple proteins, circular case with multiple proteins

    distinct_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A", "B", "B"], "_decoy": [0,0,0,0]}
    distinct_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A", "B", "B"], "_decoy": [0,0,0,0], "pg_master":["A","A","B","B"], "pg":["A","A","B","B"]}

    differentiable_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A;B", "A;B", "B"], "_decoy": [0,0,0,0]}
    differentiable_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A;B", "A;B", "B"], "_decoy": [0,0,0,0], "pg_master":["A","A","A","B"], "pg":["A","A","A","B"]}

    indistinguishable_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A;B", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0]}
    indistinguishable_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A;B", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0], "pg_master":["A","A","A","A"], "pg":["A;B","A;B","A;B","A;B"]}

    subset_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0]}
    subset_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0], "pg_master":["A","A","A","A"], "pg":["A;B","A;B","A;B","A;B"]}

    subsumable_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A;B", "B;C", "C"], "_decoy": [0,0,0,0]}
    subsumable_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A;B", "B;C", "C"], "_decoy": [0,0,0,0], "pg_master":["A","A","C","C"], "pg":["A","A","C;B","C;B"]}

    shared_only_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A;B", "A;B;C", "A;B;C", "A;C"], "_decoy": [0,0,0,0]}
    shared_only_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A;B", "A;B;C", "A;B;C", "A;C"], "_decoy": [0,0,0,0], "pg_master":["A","A","A","A"], "pg":["A;B;C","A;B;C","A;B;C","A;B;C"]}

    circular_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A;B;C", "B;C;D", "C;D;E", "D;E;A"], "_decoy": [0,0,0,0]}
    circular_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A;B;C", "B;C;D", "C;D;E", "D;E;A"], "_decoy": [0,0,0,0], "pg_master":["C","C","C","A"], "pg":["C;B", "C;B", "C;B", "A;D;E"]}

    complex_example_proteins_input = {"precursor_idx": [0,1,2,3], "proteins": ["P1;P2;P3;P4", "P1;P4", "P2", "P2;P5"], "_decoy": [0,0,0,0]}
    complex_example_proteins_expected = {"precursor_idx":[0,1,2,3], "proteins":["P1;P2;P3;P4", "P1;P4", "P2", "P2;P5"], "_decoy": [0,0,0,0], "pg_master":["P2", "P1", "P2", "P2"], "pg":["P2;P3;P5","P1;P4","P2;P3;P5","P2;P3;P5"]}

    test_cases = [
        ("distinct_proteins", distinct_proteins_input, distinct_proteins_expected),
        ("differentiable proteins", differentiable_proteins_input, differentiable_proteins_expected),
        ("indistinguishable proteins", indistinguishable_proteins_input, indistinguishable_proteins_expected),
        ("subset proteins", subset_proteins_input, subset_proteins_expected),
        ("subsumable proteins", subsumable_proteins_input, subsumable_proteins_expected),
        ("shared only", shared_only_proteins_input, shared_only_proteins_expected),
        ("circular", circular_proteins_input, circular_proteins_expected),
        ("complex example", complex_example_proteins_input, complex_example_proteins_expected)
    ]

    return test_cases

#parametrized function to evaluate test case correctness
@pytest.mark.parametrize('type, input_dict, expected_dict', construct_test_cases())
def test_grouping(
    type: str,
    input_dict: dict,
    expected_dict: dict,
):
    assert grouping.perform_grouping(pd.DataFrame(input_dict), genes_or_proteins = "proteins").to_dict(orient = 'list') == expected_dict

#timing test on (seeded) random generated data to monitor grouping performance
def test_grouping_performance(
        expected_time : int = 35
):
    
    #test grouping performance with dummy dataset
    np.random.seed(42)
    n_precursors = 40000

    #generate precursor index and randomize sequence
    precursor_idx = np.random.choice(np.array(range(n_precursors)), n_precursors, replace=False)

    #generate 7500 random uniprot-like protein IDs
    fake_ids = np.array(['P' + str(i) for i in np.random.choice(np.array(range(10000, 100000, 1)), 5000, replace=False)])

    #simulate distribution of IDs per precursor, roughly equivalent to exponential distribution observed in real data
    counts = np.int64(np.ceil(np.random.exponential(scale = 10.0, size = n_precursors)))
    proteins = [";".join(np.random.choice(fake_ids, i, replace = True)) for i in counts]

    #decoys
    decoys = [np.random.choice(np.array([0,1]),1,replace=True)[0] for _ in range(n_precursors)]

    #build dummy dataframe
    simulated_psm_data = pd.DataFrame({
        "precursor_idx": precursor_idx,
        "proteins": proteins,
        "_decoy": decoys
    })

    grouping_start_time = time.time()
    _ = grouping.perform_grouping(simulated_psm_data, genes_or_proteins="proteins")
    grouping_end_time = time.time()
    elapsed_time = grouping_end_time - grouping_start_time
    assert elapsed_time < expected_time
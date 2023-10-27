import pytest
import numpy as np
import pandas as pd
from alphadia import grouping

#define test cases as outlined in Nesvizhskii, Alexey I., and Ruedi Aebersold. "Interpretation of shotgun proteomic data." Molecular & cellular proteomics 4.10 (2005): 1419-1440. Figure 5
#additional cases: complex case with multiple proteins, circular case with multiple proteins

distinct_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A", "B", "B"], "_decoy": [0,0,0,0]}
distinct_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A", "B", "B"], "_decoy": [0,0,0,0], "parsimony_proteins":["A","A","B","B"], "parsimony_proteins_groups":["A","A","B","B"]}

differentiable_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A;B", "A;B", "B"], "_decoy": [0,0,0,0]}
differentiable_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A;B", "A;B", "B"], "_decoy": [0,0,0,0], "parsimony_proteins":["A","A","A","B"], "parsimony_proteins_groups":["A","A","A","B"]}

indistinguishable_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A;B", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0]}
indistinguishable_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A;B", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0], "parsimony_proteins":["A","A","A","A"], "parsimony_proteins_groups":["A;B","A;B","A;B","A;B"]}

subset_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0]}
subset_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A;B", "A;B", "A;B"], "_decoy": [0,0,0,0], "parsimony_proteins":["A","A","A","A"], "parsimony_proteins_groups":["A;B","A;B","A;B","A;B"]}

subsumable_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A", "A;B", "B;C", "C"], "_decoy": [0,0,0,0]}
subsumable_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A", "A;B", "B;C", "C"], "_decoy": [0,0,0,0], "parsimony_proteins":["A","A","C","C"], "parsimony_proteins_groups":["A","A","C;B","C;B"]}

shared_only_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A;B", "A;B;C", "A;B;C", "A;C"], "_decoy": [0,0,0,0]}
shared_only_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A;B", "A;B;C", "A;B;C", "A;C"], "_decoy": [0,0,0,0], "parsimony_proteins":["A","A","A","A"], "parsimony_proteins_groups":["A;B;C","A;B;C","A;B;C","A;B;C"]}

circular_proteins_input = {"precursor_idx": [1,2,3,4], "proteins": ["A;B;C", "B;C;D", "C;D;E", "D;E;A"], "_decoy": [0,0,0,0]}
circular_proteins_expected = {"precursor_idx":[1,2,3,4], "proteins":["A;B;C", "B;C;D", "C;D;E", "D;E;A"], "_decoy": [0,0,0,0], "parsimony_proteins":["C","C","C","A"], "parsimony_proteins_groups":["C;B", "C;B", "C;B", "A;D;E"]}

complex_example_proteins_input = {"precursor_idx": [0,1,2,3], "proteins": ["P1;P2;P3;P4", "P1;P4", "P2", "P2;P5"], "_decoy": [0,0,0,0]}
complex_example_proteins_expected = {"precursor_idx":[0,1,2,3], "proteins":["P1;P2;P3;P4", "P1;P4", "P2", "P2;P5"], "_decoy": [0,0,0,0], "parsimony_proteins":["P2", "P1", "P2", "P2"], "parsimony_proteins_groups":["P2;P3;P5","P1;P4","P2;P3;P5","P2;P3;P5"]}

@pytest.mark.parametrize(
    ('type', 'input_dict', 'expected_dict'),
    (
        ("distinct_proteins", distinct_proteins_input, distinct_proteins_expected),
        ("differentiable proteins", differentiable_proteins_input, differentiable_proteins_expected),
        ("indistinguishable proteins", indistinguishable_proteins_input, indistinguishable_proteins_expected),
        ("subset proteins", subset_proteins_input, subset_proteins_expected),
        ("subsumable proteins", subsumable_proteins_input, subsumable_proteins_expected),
        ("shared only", shared_only_proteins_input, shared_only_proteins_expected),
        ("circular", circular_proteins_input, circular_proteins_expected),
        ("complex example", complex_example_proteins_input, complex_example_proteins_expected)
    )
)

def test_grouping(
    type: str,
    input_dict: dict,
    expected_dict: dict,
):
    assert grouping.perform_grouping(pd.DataFrame(input_dict), genes_or_proteins = "proteins").to_dict(orient = 'list') == expected_dict
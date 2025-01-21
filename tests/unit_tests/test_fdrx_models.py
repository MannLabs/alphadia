import pandas as pd
import pytest
from collections import Counter

from alphadia.fdrx.models.two_step_classifier import apply_absolute_transformations, get_target_decoy_partners, compute_and_filter_q_values


def test_apply_absolute_transformations():
    data = {
        'delta_rt': [-1, -2, 3],
        'top_3_ms2_mass_error': [-1, -2, -3],
        'mean_ms2_mass_error': [1, -2, 3],
        'extra_column': [-1, -2, -3]
    }
    df = pd.DataFrame(data)

    transformed_df = apply_absolute_transformations(df)

    assert (transformed_df['delta_rt'] >= 0).all(), "delta_rt contains negative values"
    assert (transformed_df['top_3_ms2_mass_error'] >= 0).all(), "top_3_ms2_mass_error contains negative values"
    assert (transformed_df['mean_ms2_mass_error'] >= 0).all(), "mean_ms2_mass_error contains negative values"

    assert (transformed_df['extra_column'] == df['extra_column']).all(), "extra_column should not be transformed"


@pytest.fixture
def setup_data():
    
    reference_df = pd.DataFrame({
        'decoy': [0, 1], 
        'rank': [1, 0],
        'elution_group_idx': [100, 101]
    })
    
    full_df = pd.DataFrame({
        'decoy':             [  0,   0,   1,   1,   0], 
        'rank':              [  1,   0,   2,   1,   2],
        'elution_group_idx': [100, 101, 102, 100, 102],
        'intensity':         [200, 150, 120, 130,  95],
        'peptide':          ['pepA', 'pepB', 'pepC', 'pepD', 'pepE']
        
    })
    
    return reference_df, full_df


def test_get_target_decoy_partners_correct_extraction(setup_data):
    reference_df, full_df = setup_data
    group_columns = ['elution_group_idx', 'rank']
    result_df = get_target_decoy_partners(reference_df, full_df, group_by=group_columns)
    
    assert len(result_df) == 3  # should match rows with ("rank", "elution_group_idx")=(1,100) and (2,101)
    assert all(col in result_df.columns for col in full_df.columns)
    
    assert Counter(result_df['decoy']) == Counter([0, 0, 1])
    assert Counter(result_df['peptide']) == Counter(['pepA', 'pepB', 'pepD'])


def test_handling_nonexistent_partners_in_get_target_decoy_partners_(setup_data):
    reference_df, full_df = setup_data
    
    reference_df.loc[1] = [0, 3, 104]
    result_df = get_target_decoy_partners(reference_df, full_df)
        
    assert len(result_df) == 3
    assert not result_df[(result_df['rank'] == 3) & (result_df['elution_group_idx'] == 104)].empty == True 



@pytest.mark.parametrize(
    ["fdr", "remove_decoys", "expected_length", "expected_decoy_count"],
    [
        (0.5, True, 3, 0),
        (0.01, True, 1, 0),
        (0.5, False, 4, 1),
        (0.01, False, 1, 0),
    ]
)
def test_compute_and_filter_q_values(fdr, remove_decoys, expected_length, expected_decoy_count):
    df = pd.DataFrame({
        'proba': [0.1, 0.3, 0.8, 0.9, 0.9, 0.2, 0.4, 0.5],
        'decoy': [0, 1, 0, 1, 0, 1, 0, 1],
        'group': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    })
    result = compute_and_filter_q_values(df, fdr=fdr, group_columns=['group'], remove_decoys=remove_decoys)
    print(result)
    assert len(result) == expected_length
    assert len(result[result['decoy'] == 1]) == expected_decoy_count

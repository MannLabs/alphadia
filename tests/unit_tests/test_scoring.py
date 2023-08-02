from alphadia.extraction.scoring import keep_best

import pandas as pd


def test_keep_best():
    test_df = pd.DataFrame({
        'channel': [0,0,0,4,4,4,8,8,8],
        'elution_group_idx': [0,1,2,0,1,2,0,1,2],
        'proba': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3]
    })

    result_df = keep_best(test_df)
    pd.testing.assert_frame_equal(result_df, test_df)

    test_df = pd.DataFrame({
        'channel': [0,0,0,4,4,4,8,8,8],
        'elution_group_idx': [0,0,1,0,0,1,0,0,1],
        'proba': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3]
    })
    result_df = keep_best(test_df)
    result_expected = pd.DataFrame({
        'channel': [0,0,4,4,8,8],
        'elution_group_idx': [0,1,0,1,0,1],
        'proba': [0.1, 0.3, 0.4, 0.6, 0.1, 0.3]
    })
    pd.testing.assert_frame_equal(result_df, result_expected)

    test_df = pd.DataFrame({
        'channel': [0,0,0,4,4,4,8,8,8],
        'precursor_idx': [0,0,1,0,0,1,0,0,1],
        'proba': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3]
    })
    result_df = keep_best(test_df, group_columns = ['channel', 'precursor_idx'])
    result_expected = pd.DataFrame({
        'channel': [0,0,4,4,8,8],
        'precursor_idx': [0,1,0,1,0,1],
        'proba': [0.1, 0.3, 0.4, 0.6, 0.1, 0.3]
    })
    pd.testing.assert_frame_equal(result_df, result_expected)
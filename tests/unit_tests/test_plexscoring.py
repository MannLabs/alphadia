import numpy as np
import pandas as pd

from alphadia.extraction import plexscoring

def test_multiplex_candidates():

    test_candidate_df = pd.DataFrame(
        {
            'elution_group_idx': [0,0,1],
            'precursor_idx': [0,1,3],
            'proba': [0.1, 0.4, 0.3],
            'rank': [0,0,0],
            'frame_start': [0,0,0],
            'frame_center': [0,0,0],
            'frame_stop': [0,0,0],
            'scan_start': [0,0,0],
            'scan_stop': [0,0,0],
            'scan_center': [0,0,0],
        }
    )

    test_precursor_df = pd.DataFrame(
        {
            'precursor_idx': [0,1,2,3,4,5],
            'elution_group_idx': [0,0,0,1,1,1],
            'decoy': [0,0,0,0,0,0],
            'channel': [0,4,8,0,4,8],
            'flat_frag_start_idx': [0,0,0,0,0,0],
            'flat_frag_stop_idx': [0,0,0,0,0,0],
            'charge': [2,2,2,2,2,2],
            'rt_library': [0,0,0,0,0,0],
            'mobility_library': [0,0,0,0,0,0],
            'mz_library': [0,0,0,0,0,0],
        }
    )

    multiplexed_candidates = plexscoring.multiplex_candidates(test_candidate_df, test_precursor_df, channels=[])
    pd.testing.assert_frame_equal(multiplexed_candidates.sort_index(axis=1), test_candidate_df.sort_index(axis=1))

    multiplexed_candidates = plexscoring.multiplex_candidates(test_candidate_df, test_precursor_df, channels=[0,4,8])
    assert multiplexed_candidates['precursor_idx'].tolist() == [0,1,2,3,4,5]
    assert np.all(np.isclose(multiplexed_candidates['proba'].tolist(),[0.1, 0.4, 0.1, 0.3, 0.3, 0.3]))
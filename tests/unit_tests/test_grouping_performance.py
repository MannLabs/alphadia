import time
import pytest
import numpy as np
import pandas as pd
from alphadia import grouping
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

def test_grouping_performance(
        simulated_psm_data : pd.DataFrame = simulated_psm_data,
        expected_time : int = 10
):
    
    grouping_start_time = time.time()
    _ = grouping.perform_grouping(simulated_psm_data, genes_or_proteins="proteins")
    grouping_end_time = time.time()
    elapsed_time = grouping_end_time - grouping_start_time
    assert elapsed_time < expected_time
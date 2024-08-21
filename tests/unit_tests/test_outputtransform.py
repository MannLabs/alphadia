import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from conftest import mock_fragment_df, mock_precursor_df

from alphadia import outputtransform
from alphadia.workflow import manager, peptidecentric


def test_output_transform():
    run_columns = ["run_0", "run_1", "run_2"]

    config = {
        "general": {
            "thread_count": 8,
        },
        "search": {"channel_filter": "0"},
        "fdr": {
            "fdr": 0.01,
            "inference_strategy": "heuristic",
            "group_level": "proteins",
            "keep_decoys": False,
        },
        "search_output": {
            "min_k_fragments": 3,
            "min_correlation": 0.25,
            "num_samples_quadratic": 50,
            "min_nonnan": 1,
            "normalize_lfq": True,
            "peptide_level_lfq": False,
            "precursor_level_lfq": False,
            "file_format": "parquet",
        },
        "multiplexing": {
            "enabled": False,
        },
        "search_initial": {
            "initial_ms1_tolerance": 4,
            "initial_ms2_tolerance": 7,
            "initial_rt_tolerance": 200,
            "initial_mobility_tolerance": 0.04,
            "initial_num_candidates": 1,
        },
        "optimization_manager": {
            "fwhm_rt": 2.75,
            "fwhm_mobility": 2,
            "score_cutoff": 50,
        },
    }

    temp_folder = os.path.join(tempfile.gettempdir(), "alphadia")
    os.makedirs(temp_folder, exist_ok=True)

    progress_folder = os.path.join(temp_folder, "progress")
    os.makedirs(progress_folder, exist_ok=True)

    # setup raw folders
    raw_folders = [os.path.join(progress_folder, run) for run in run_columns]

    psm_base_df = mock_precursor_df(n_precursor=100)
    fragment_base_df = mock_fragment_df(n_precursor=200)

    for raw_folder in raw_folders:
        os.makedirs(raw_folder, exist_ok=True)

        psm_df = psm_base_df.sample(50)
        psm_df["run"] = os.path.basename(raw_folder)
        frag_df = fragment_base_df[
            fragment_base_df["precursor_idx"].isin(psm_df["precursor_idx"])
        ]

        frag_df.to_parquet(os.path.join(raw_folder, "frag.parquet"), index=False)
        psm_df.to_parquet(os.path.join(raw_folder, "psm.parquet"), index=False)

        optimization_manager = manager.OptimizationManager(
            config,
            path=os.path.join(
                raw_folder,
                peptidecentric.PeptideCentricWorkflow.OPTIMIZATION_MANAGER_PATH,
            ),
        )
        optimization_manager.fit({"ms2_error": 6})
        optimization_manager.save()
        timing_manager = manager.TimingManager(
            path=os.path.join(
                raw_folder, peptidecentric.PeptideCentricWorkflow.TIMING_MANAGER_PATH
            )
        )

        timing_manager.set_start_time("extraction")
        timing_manager.set_end_time("extraction")
        timing_manager.save()

    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_precursor_table(raw_folders, save=True)
    _ = output.build_stat_df(raw_folders, save=True)
    _ = output.build_internal_df(raw_folders, save=True)
    _ = output.build_lfq_tables(raw_folders, save=True)

    # validate psm_df output
    psm_df = pd.read_parquet(
        os.path.join(temp_folder, f"{output.PRECURSOR_OUTPUT}.parquet"),
    )
    assert all(
        [
            col in psm_df.columns
            for col in [
                "pg",
                "precursor_idx",
                "decoy",
                "mz_library",
                "charge",
                "proteins",
                "genes",
                "proba",
                "qval",
                "run",
            ]
        ]
    )
    assert psm_df["run"].nunique() == 3

    # validate stat_df output
    stat_df = pd.read_csv(
        os.path.join(temp_folder, f"{output.STAT_OUTPUT}.tsv"), sep="\t"
    )
    assert len(stat_df) == 3
    assert stat_df["ms2_error"][0] == 6
    assert stat_df["rt_error"][0] == 200

    assert all(
        [
            col in stat_df.columns
            for col in [
                "run",
                "precursors",
                "proteins",
            ]
        ]
    )

    internal_df = pd.read_csv(
        os.path.join(temp_folder, f"{output.INTERNAL_OUTPUT}.tsv"), sep="\t"
    )
    assert isinstance(internal_df["duration_extraction"][0], float)
    # validate protein_df output
    protein_df = pd.read_parquet(os.path.join(temp_folder, "pg.matrix.parquet"))
    assert all([col in protein_df.columns for col in ["run_0", "run_1", "run_2"]])

    for i in run_columns:
        for j in run_columns:
            if i == j:
                continue
            assert np.corrcoef(protein_df[i], protein_df[j])[0, 0] > 0.5

    shutil.rmtree(temp_folder)

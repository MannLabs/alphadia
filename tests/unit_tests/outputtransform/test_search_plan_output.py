import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from conftest import mock_fragment_df, mock_precursor_df

from alphadia.outputtransform.search_plan_output import SearchPlanOutput
from alphadia.workflow.base import QUANT_FOLDER_NAME
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.managers.timing_manager import TimingManager
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow


def test_output_transform():
    run_columns = ["run_0", "run_1", "run_2"]

    config = {
        "general": {"thread_count": 8, "save_figures": True, "save_mbr_library": False},
        "transfer_library": {"enabled": False},
        "transfer_learning": {"enabled": False},
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
            "save_fragment_quant_matrix": False,
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

    quant_path = os.path.join(temp_folder, QUANT_FOLDER_NAME)
    os.makedirs(quant_path, exist_ok=True)

    # setup raw folders
    raw_folders = [os.path.join(quant_path, run) for run in run_columns]

    psm_base_df = mock_precursor_df(n_precursor=100)
    fragment_base_df = mock_fragment_df(n_precursor=200)

    for i, raw_folder in enumerate(raw_folders):
        os.makedirs(raw_folder, exist_ok=True)

        psm_df = psm_base_df.sample(50)
        psm_df["run"] = os.path.basename(raw_folder)
        frag_df = fragment_base_df[
            fragment_base_df["precursor_idx"].isin(psm_df["precursor_idx"])
        ]

        frag_df.to_parquet(os.path.join(raw_folder, "frag.parquet"), index=False)
        psm_df.to_parquet(os.path.join(raw_folder, "psm.parquet"), index=False)

        optimization_manager = OptimizationManager(
            config,
            path=os.path.join(
                raw_folder,
                PeptideCentricWorkflow.OPTIMIZATION_MANAGER_PKL_NAME,
            ),
        )

        timing_manager = TimingManager(
            path=os.path.join(
                raw_folder,
                PeptideCentricWorkflow.TIMING_MANAGER_PKL_NAME,
            )
        )

        if (
            i == 2
        ):  # simulate the case that the search fails such that the optimization and timing managers are not saved
            pass
        else:
            optimization_manager.update(ms2_error=6)
            optimization_manager.save()
            timing_manager.set_start_time("extraction")
            timing_manager.set_end_time("extraction")
            timing_manager.save()

    SearchPlanOutput(config, temp_folder).build(raw_folders, None)

    # validate psm_df output
    psm_df = pd.read_parquet(
        os.path.join(temp_folder, f"{SearchPlanOutput.PRECURSOR_OUTPUT}.parquet"),
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
        os.path.join(temp_folder, f"{SearchPlanOutput.STAT_OUTPUT}.tsv"), sep="\t"
    )
    assert len(stat_df) == 3

    assert stat_df["optimization.ms2_error"][0] == 6
    assert stat_df["optimization.rt_error"][0] == 200

    assert all(
        [
            col in stat_df.columns
            for col in [
                "run",
                "channel",
                "precursors",
                "proteins",
                "fwhm_rt",
                "fwhm_mobility",
                "optimization.ms2_error",
                "optimization.ms1_error",
                "optimization.rt_error",
                "optimization.mobility_error",
                "calibration.ms2_median_accuracy",
                "calibration.ms2_median_precision",
                "calibration.ms1_median_accuracy",
                "calibration.ms1_median_precision",
                "raw.gradient_min_m",
                "raw.gradient_max_m",
                "raw.gradient_length_m",
                "raw.cycle_length",
                "raw.cycle_duration",
                "raw.cycle_number",
                "raw.msms_range_min",
                "raw.msms_range_max",
            ]
        ]
    )

    internal_df = pd.read_csv(
        os.path.join(temp_folder, f"{SearchPlanOutput.INTERNAL_OUTPUT}.tsv"), sep="\t"
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


def test_merge_quant_levels_to_psm_merges_precursor_level():
    """Test that precursor level quantification is merged correctly."""
    from alphadia.outputtransform.search_plan_output import (
        LFQOutputConfig,
        SearchPlanOutput,
    )

    spo = SearchPlanOutput({"general": {"save_figures": False}}, "/tmp")
    psm_df = pd.DataFrame({"mod_seq_charge_hash": ["A1"], "run": ["run1"]})
    lfq_results = {
        "precursor": pd.DataFrame({"mod_seq_charge_hash": ["A1"], "run1": [100.0]})
    }
    configs = [LFQOutputConfig("mod_seq_charge_hash", "precursor")]

    result = spo._merge_quant_levels_to_psm(psm_df, lfq_results, configs)

    assert "precursor_intensity" in result.columns
    assert result["precursor_intensity"].iloc[0] == 100.0


def test_merge_quant_levels_to_psm_merges_peptide_level():
    """Test that peptide level quantification is merged correctly."""
    from alphadia.outputtransform.search_plan_output import (
        LFQOutputConfig,
        SearchPlanOutput,
    )

    spo = SearchPlanOutput({"general": {"save_figures": False}}, "/tmp")
    psm_df = pd.DataFrame({"mod_seq_hash": ["A"], "run": ["run1"]})
    lfq_results = {"peptide": pd.DataFrame({"mod_seq_hash": ["A"], "run1": [400.0]})}
    configs = [LFQOutputConfig("mod_seq_hash", "peptide")]

    result = spo._merge_quant_levels_to_psm(psm_df, lfq_results, configs)

    assert "peptide_intensity" in result.columns
    assert result["peptide_intensity"].iloc[0] == 400.0


def test_merge_quant_levels_to_psm_merges_protein_group_level():
    """Test that protein group level quantification is merged correctly."""
    from alphadia.outputtransform.search_plan_output import (
        LFQOutputConfig,
        SearchPlanOutput,
    )

    spo = SearchPlanOutput({"general": {"save_figures": False}}, "/tmp")
    psm_df = pd.DataFrame({"pg": ["PG1"], "run": ["run1"]})
    lfq_results = {"pg": pd.DataFrame({"pg": ["PG1"], "run1": [700.0]})}
    configs = [LFQOutputConfig("pg", "pg")]

    result = spo._merge_quant_levels_to_psm(psm_df, lfq_results, configs)

    assert "intensity" in result.columns
    assert result["intensity"].iloc[0] == 700.0


def test_merge_quant_levels_to_psm_handles_empty_lfq_results():
    """Test that empty LFQ results are handled gracefully."""
    from alphadia.outputtransform.search_plan_output import (
        LFQOutputConfig,
        SearchPlanOutput,
    )

    spo = SearchPlanOutput({"general": {"save_figures": False}}, "/tmp")
    psm_df = pd.DataFrame({"mod_seq_charge_hash": ["A1"], "run": ["run1"]})
    lfq_results = {"precursor": pd.DataFrame()}
    configs = [LFQOutputConfig("mod_seq_charge_hash", "precursor")]

    result = spo._merge_quant_levels_to_psm(psm_df, lfq_results, configs)

    assert len(result) == 1
    assert "precursor_intensity" not in result.columns


def test_merge_quant_levels_to_psm_merges_all_levels():
    """Test that all quantification levels are merged in one call."""
    from alphadia.outputtransform.search_plan_output import (
        LFQOutputConfig,
        SearchPlanOutput,
    )

    spo = SearchPlanOutput({"general": {"save_figures": False}}, "/tmp")
    psm_df = pd.DataFrame(
        {
            "mod_seq_charge_hash": ["A1"],
            "mod_seq_hash": ["A"],
            "pg": ["PG1"],
            "run": ["run1"],
        }
    )
    lfq_results = {
        "precursor": pd.DataFrame({"mod_seq_charge_hash": ["A1"], "run1": [100.0]}),
        "peptide": pd.DataFrame({"mod_seq_hash": ["A"], "run1": [400.0]}),
        "pg": pd.DataFrame({"pg": ["PG1"], "run1": [700.0]}),
    }
    configs = [
        LFQOutputConfig("mod_seq_charge_hash", "precursor"),
        LFQOutputConfig("mod_seq_hash", "peptide"),
        LFQOutputConfig("pg", "pg"),
    ]

    result = spo._merge_quant_levels_to_psm(psm_df, lfq_results, configs)

    assert all(
        col in result.columns
        for col in ["precursor_intensity", "peptide_intensity", "intensity"]
    )
    assert result["precursor_intensity"].iloc[0] == 100.0
    assert result["peptide_intensity"].iloc[0] == 400.0
    assert result["intensity"].iloc[0] == 700.0

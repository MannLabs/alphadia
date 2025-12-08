import os
import shutil
import tempfile

import pandas as pd
from conftest import mock_fragment_df, mock_precursor_df

from alphadia.constants.keys import InferenceStrategy, NormalizationMethods
from alphadia.outputtransform.quantification.quant_output_builder import (
    LFQOutputConfig,
)
from alphadia.outputtransform.search_plan_output import SearchPlanOutput
from alphadia.outputtransform.utils import merge_quant_levels_to_psm
from alphadia.workflow.base import QUANT_FOLDER_NAME
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.managers.timing_manager import TimingManager
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow


def test_search_plan_output_integration():
    """Integration test for SearchPlanOutput.build() covering end-to-end workflow.

    Tests that SearchPlanOutput.build() correctly orchestrates:
    - Protein grouping and FDR
    - Label-free quantification
    - Statistics collection from manager files
    - Output file generation (precursors, proteins, stat, internal)
    """
    # given
    run_columns = ["run_0", "run_1", "run_2"]

    config = {
        "general": {
            "thread_count": 8,
            "save_figures": False,
            "save_mbr_library": False,
        },
        "transfer_library": {"enabled": False},
        "transfer_learning": {"enabled": False},
        "search": {"channel_filter": "0"},
        "fdr": {
            "fdr": 0.01,
            "inference_strategy": InferenceStrategy.HEURISTIC,
            "group_level": "proteins",
            "keep_decoys": False,
        },
        "search_output": {
            "precursor_level_lfq": True,
            "peptide_level_lfq": True,
            "min_k_fragments": 3,
            "min_correlation": 0.25,
            "num_samples_quadratic": 50,
            "min_nonnan": 1,
            "save_fragment_quant_matrix": False,
            "file_format": "parquet",
            "normalization_method": NormalizationMethods.NORMALIZE_DIRECTLFQ,
            "normalize_directlfq": True,
        },
        "multiplexing": {"enabled": False},
        "search_initial": {
            "ms1_tolerance": 4,
            "ms2_tolerance": 7,
            "rt_tolerance": 200,
            "mobility_tolerance": 0.04,
            "num_candidates": 1,
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
                raw_folder, PeptideCentricWorkflow.OPTIMIZATION_MANAGER_PKL_NAME
            ),
        )
        timing_manager = TimingManager(
            path=os.path.join(
                raw_folder, PeptideCentricWorkflow.TIMING_MANAGER_PKL_NAME
            )
        )

        if i == 2:
            pass
        else:
            optimization_manager.update(ms2_error=6)
            optimization_manager.save()
            timing_manager.set_start_time("extraction")
            timing_manager.set_end_time("extraction")
            timing_manager.save()

    # when
    SearchPlanOutput(config, temp_folder).build(raw_folders, None)

    # then
    psm_df = pd.read_parquet(
        os.path.join(temp_folder, f"{SearchPlanOutput.PRECURSOR_OUTPUT}.parquet")
    )
    assert psm_df["raw.name"].nunique() == 3
    assert all(
        col in psm_df.columns
        for col in [
            "pg.name",
            "precursor.idx",
            "precursor.decoy",
            "precursor.mz.library",
            "precursor.charge",
            "pg.proteins",
            "pg.genes",
            "precursor.proba",
            "precursor.qval",
            "raw.name",
        ]
    )

    stat_df = pd.read_csv(
        os.path.join(temp_folder, f"{SearchPlanOutput.STAT_OUTPUT}.tsv"), sep="\t"
    )
    assert len(stat_df) == 3
    assert stat_df["optimization.ms2_error"][0] == 6
    assert stat_df["optimization.rt_error"][0] == 200
    assert all(
        col in stat_df.columns
        for col in [
            "raw.name",
            "search.channel",
            "search.precursors",
            "search.proteins",
            "search.fwhm_rt",
            "search.fwhm_mobility",
            "optimization.ms2_error",
            "optimization.ms1_error",
            "optimization.rt_error",
            "optimization.mobility_error",
            "calibration.ms2_bias",
            "calibration.ms2_variance",
            "calibration.ms1_bias",
            "calibration.ms1_variance",
            "raw.gradient_length",
            "raw.cycle_length",
            "raw.cycle_duration",
            "raw.cycle_number",
            "raw.ms2_range_min",
            "raw.ms2_range_max",
        ]
    )

    internal_df = pd.read_csv(
        os.path.join(temp_folder, f"{SearchPlanOutput.INTERNAL_OUTPUT}.tsv"), sep="\t"
    )
    assert isinstance(internal_df["duration_extraction"][0], float)

    protein_df = pd.read_parquet(os.path.join(temp_folder, "pg.matrix.parquet"))
    assert all(col in protein_df.columns for col in run_columns)

    shutil.rmtree(temp_folder)


def test_merge_quant_levels_to_psm_handles_empty_lfq():
    """Test merge_quant_levels_to_psm with empty LFQ results."""
    # given
    psm_df = pd.DataFrame({"mod_seq_charge_hash": ["A1"], "run": ["run1"]})
    lfq_results = {"precursor": pd.DataFrame()}
    configs = [
        LFQOutputConfig(
            "mod_seq_charge_hash",
            "precursor",
            "precursor.intensity",
            ["pg", "sequence", "mods", "charge"],
        )
    ]

    # when
    result = merge_quant_levels_to_psm(psm_df, lfq_results, configs)

    # then
    expected = pd.DataFrame({"mod_seq_charge_hash": ["A1"], "run": ["run1"]})
    pd.testing.assert_frame_equal(result, expected)


def test_merge_quant_levels_to_psm_merges_all_levels():
    """Test merge_quant_levels_to_psm with all three quantification levels."""
    # given
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
        LFQOutputConfig(
            "mod_seq_charge_hash",
            "precursor",
            "precursor.intensity",
            ["pg", "sequence", "mods", "charge"],
        ),
        LFQOutputConfig(
            "mod_seq_hash", "peptide", "peptide.intensity", ["pg", "sequence", "mods"]
        ),
        LFQOutputConfig("pg", "pg", "pg.intensity", ["pg"]),
    ]

    # when
    result = merge_quant_levels_to_psm(psm_df, lfq_results, configs)

    # then
    expected = pd.DataFrame(
        {
            "mod_seq_charge_hash": ["A1"],
            "mod_seq_hash": ["A"],
            "pg": ["PG1"],
            "run": ["run1"],
            "precursor.intensity": [100.0],
            "peptide.intensity": [400.0],
            "pg.intensity": [700.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)

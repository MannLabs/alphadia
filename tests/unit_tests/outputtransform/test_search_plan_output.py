import os

import numpy as np
import pandas as pd
import pytest
from conftest import mock_fragment_df, mock_precursor_df

from alphadia.constants.keys import (
    FragmentCols,
    InferenceStrategy,
    NormalizationMethods,
    PeptideOutputCols,
    PrecursorOutputCols,
    ProteinGroupOutputCols,
    QuantificationLevelName,
    SearchPlanFiles,
    StatSearchCols,
)
from alphadia.outputtransform.quantification.quant_output_builder import (
    LFQOutputConfig,
)
from alphadia.outputtransform.search_plan_output import SearchPlanOutput
from alphadia.outputtransform.utils import merge_quant_levels_to_psm
from alphadia.workflow.base import QUANT_FOLDER_NAME
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.managers.timing_manager import TimingManager
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow


@pytest.mark.parametrize(
    ("normalization_method", "library_anchored_quantification"),
    [
        (NormalizationMethods.DIRECTLFQ, True),
        (NormalizationMethods.DIRECTLFQ, False),
        (NormalizationMethods.NONE, True),
    ],
)
def test_search_plan_output_integration(
    normalization_method, library_anchored_quantification, tmp_path
):
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
            "file_format": "parquet",
            "normalization_method": normalization_method,
            "normalize_directlfq": True,
            "intensity_drift_correction": True,
            "library_anchored_quantification": library_anchored_quantification,
            "anchored_tau": None,
            "anchored_lambda": None,
            "anchored_sweeps": 20,
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

    temp_folder = str(tmp_path)
    quant_path = os.path.join(temp_folder, QUANT_FOLDER_NAME)
    os.makedirs(quant_path, exist_ok=True)
    raw_folders = [os.path.join(quant_path, run) for run in run_columns]

    psm_base_df = mock_precursor_df(n_precursor=100)
    fragment_base_df = mock_fragment_df(n_precursor=200)
    fragment_base_df[FragmentCols.INTENSITY_LIBRARY] = np.random.default_rng(0).uniform(
        0.1, 1.0, len(fragment_base_df)
    )

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
        os.path.join(temp_folder, f"{SearchPlanFiles.PRECURSORS_FILE_STEM}.parquet")
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
        os.path.join(temp_folder, SearchPlanFiles.STAT_FILE_NAME), sep="\t"
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
        os.path.join(temp_folder, SearchPlanFiles.INTERNAL_FILE_NAME), sep="\t"
    )
    assert isinstance(internal_df["duration_extraction"][0], float)

    lfq_enabled = normalization_method != NormalizationMethods.NONE
    for level_name in [
        QuantificationLevelName.PRECURSOR,
        QuantificationLevelName.PEPTIDE,
        QuantificationLevelName.PROTEIN,
    ]:
        matrix_path = os.path.join(temp_folder, f"{level_name}.matrix.parquet")
        assert os.path.exists(matrix_path) == lfq_enabled
        if lfq_enabled:
            matrix_df = pd.read_parquet(matrix_path)
            assert all(col in matrix_df.columns for col in run_columns)

    lfq_intensity_columns = [
        PrecursorOutputCols.INTENSITY,
        PeptideOutputCols.INTENSITY,
        ProteinGroupOutputCols.INTENSITY,
    ]
    assert all((col in psm_df.columns) == lfq_enabled for col in lfq_intensity_columns)


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


def test_build_stat_df_returns_stats_without_writing(tmp_path):
    # given
    config = {
        "general": {"save_figures": False},
        "search": {"channel_filter": "0"},
        "multiplexing": {"enabled": False},
    }
    raw_folder = tmp_path / "run_0"
    raw_folder.mkdir()
    psm_df = pd.DataFrame(
        {"run": ["run_0"] * 2, "decoy": [0, 1], "channel": [0, 0], "pg": ["A", "B"]}
    )

    # when
    stat_df = SearchPlanOutput(config, str(tmp_path))._build_stat_df(
        [str(raw_folder)], psm_df
    )

    # then
    assert stat_df[StatSearchCols.PRECURSORS].tolist() == [1]
    assert not (tmp_path / SearchPlanFiles.STAT_FILE_NAME).exists()

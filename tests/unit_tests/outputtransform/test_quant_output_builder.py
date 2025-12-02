from unittest.mock import patch

import pandas as pd
import pytest

from alphadia.constants.keys import NormalizationMethods
from alphadia.outputtransform.quantification import (
    QuantificationLevelName,
    QuantOutputBuilder,
)
from alphadia.outputtransform.quantification.quant_output_builder import (
    LFQOutputConfig,
)


@pytest.fixture
def config():
    """Configuration for quantification with all three quant levels enabled."""
    return {
        "general": {"thread_count": 4},
        "search_output": {
            "precursor_level_lfq": True,
            "peptide_level_lfq": True,
            "min_k_fragments": 3,
            "min_correlation": 0.5,
            "min_nonnan": 1,
            "num_samples_quadratic": 50,
            "normalize_lfq": True,
            "save_fragment_quant_matrix": False,
            "file_format": "parquet",
            "normalization_method": NormalizationMethods.NORMALIZE_DIRECTLFQ,
        },
    }


@pytest.fixture
def psm_df():
    """PSM dataframe with target and decoy precursors."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4],
            "decoy": [0, 0, 0, 0, 1],
            "pg": ["PG001", "PG002", "PG001", "PG003", "PG001"],
            "mod_seq_hash": [1, 2, 3, 4, 5],
            "mod_seq_charge_hash": [10, 20, 30, 40, 50],
            "sequence": ["PEPTIDE", "SEQUENCE", "PEPTIDE", "PROTEIN", "PEPTIDE"],
            "mods": ["", "Oxidation@M", "", "", ""],
            "mod_sites": ["", "5", "", "", ""],
            "charge": [2, 2, 3, 2, 2],
            "run": ["run1", "run1", "run2", "run2", "run1"],
        }
    )


@pytest.fixture
def feature_dfs():
    """Fragment intensity and correlation dataframes."""
    return {
        "intensity": pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2, 3],
                "ion": [100, 101, 102, 103],
                "run1": [1000.0, 2000.0, 3000.0, 4000.0],
                "run2": [1100.0, 2100.0, 3100.0, 4100.0],
                "pg": ["PG001", "PG002", "PG001", "PG003"],
                "mod_seq_hash": [1, 2, 3, 4],
                "mod_seq_charge_hash": [10, 20, 30, 40],
            }
        ),
        "correlation": pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2, 3],
                "ion": [100, 101, 102, 103],
                "run1": [0.9, 0.8, 0.9, 0.7],
                "run2": [0.8, 0.9, 0.8, 0.6],
                "pg": ["PG001", "PG002", "PG001", "PG003"],
                "mod_seq_hash": [1, 2, 3, 4],
                "mod_seq_charge_hash": [10, 20, 30, 40],
            }
        ),
    }


def lfq_side_effect(*args, **kwargs):
    # Extract the lfq_config instance from kwargs (it's passed as a keyword argument)
    lfq_config = kwargs.get("lfq_config")

    # Get the quant_level attribute from the LFQOutputConfig instance
    quant_level = getattr(lfq_config, "quant_level", "pg") if lfq_config else "pg"

    if quant_level == "mod_seq_charge_hash":
        return pd.DataFrame({"mod_seq_charge_hash": [10, 20], "run1": [1000.0, 2000.0]})
    elif quant_level == "mod_seq_hash":
        return pd.DataFrame({"mod_seq_hash": [1, 2], "run1": [1500.0, 2500.0]})
    return pd.DataFrame({"pg": ["PG001", "PG002"], "run1": [5000.0, 2000.0]})


class TestQuantOutputBuilder:
    """Test QuantOutputBuilder workflow."""

    def test_initialization_filters_decoys(self, psm_df, config):
        """Given PSM df with decoys, when initialized, then decoys are filtered for quantification."""
        # Given: psm_df contains 1 decoy (see fixture)

        # When
        builder = QuantOutputBuilder(psm_df, config)

        # Then
        assert len(builder.fragment_loader.psm_df) == 4
        assert all(builder.fragment_loader.psm_df["decoy"] == 0)

    @patch(
        "alphadia.outputtransform.quantification.fragment_accumulator.FragmentQuantLoader.accumulate_from_folders"
    )
    def test_build_returns_empty_when_no_fragments(
        self, mock_accumulate, psm_df, config
    ):
        """Given no fragment data, when build is called, then returns empty results."""
        # Given
        mock_accumulate.return_value = None
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, result_psm_df = builder.build(["folder1", "folder2"])

        # Then
        assert lfq_results == {}
        pd.testing.assert_frame_equal(result_psm_df, psm_df)

    @patch(
        "alphadia.outputtransform.quantification.quant_output_builder.QuantBuilder.direct_lfq"
    )
    @patch(
        "alphadia.outputtransform.quantification.quant_output_builder.QuantBuilder.filter_frag_df"
    )
    @patch(
        "alphadia.outputtransform.quantification.fragment_accumulator.FragmentQuantLoader.accumulate_from_folders"
    )
    def test_build_processes_all_levels_with_correct_annotations(
        self, mock_accumulate, mock_filter, mock_direct_lfq, psm_df, config, feature_dfs
    ):
        """Given all three quantification levels enabled, when build is called, then all levels return output with correct annotations."""
        # Given
        mock_accumulate.return_value = feature_dfs
        mock_filter.return_value = (
            feature_dfs["intensity"],
            feature_dfs["correlation"],
        )
        mock_direct_lfq.side_effect = lfq_side_effect
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, _ = builder.build(["folder1", "folder2"])

        # Then - verify all three levels present
        assert QuantificationLevelName.PRECURSOR in lfq_results
        assert QuantificationLevelName.PEPTIDE in lfq_results
        assert QuantificationLevelName.PROTEIN in lfq_results

        # Verify precursor annotations include charge
        precursor_result = lfq_results[QuantificationLevelName.PRECURSOR]
        assert all(
            col in precursor_result.columns
            for col in ["pg", "sequence", "mods", "charge"]
        )

        # Verify peptide annotations exclude charge
        peptide_result = lfq_results[QuantificationLevelName.PEPTIDE]
        assert all(col in peptide_result.columns for col in ["pg", "sequence", "mods"])
        assert "charge" not in peptide_result.columns

        # Verify protein has no sequence annotations
        protein_result = lfq_results[QuantificationLevelName.PROTEIN]
        assert "pg" in protein_result.columns
        assert all(
            col not in protein_result.columns for col in ["sequence", "mods", "charge"]
        )

    @patch("alphadia.outputtransform.utils.write_df")
    def test_save_results_writes_non_empty_results(self, mock_write_df, psm_df, config):
        """Given LFQ results with data, when save_results is called, then writes files to disk."""
        # Given
        lfq_results = {
            QuantificationLevelName.PRECURSOR: pd.DataFrame(
                {"mod_seq_charge_hash": [10], "run1": [1000.0]}
            ),
            QuantificationLevelName.PROTEIN: pd.DataFrame(
                {"pg": ["PG001"], "run1": [5000.0]}
            ),
        }
        builder = QuantOutputBuilder(psm_df, config)

        # When
        builder.save_results(lfq_results, "/output", file_format="parquet")

        # Then
        assert mock_write_df.call_count == 2

    @patch("alphadia.outputtransform.utils.write_df")
    def test_save_results_skips_empty_results(self, mock_write_df, psm_df, config):
        """Given LFQ results with empty dataframes, when save_results is called, then skips empty results."""
        # Given
        lfq_results = {
            QuantificationLevelName.PROTEIN: pd.DataFrame(
                {"pg": ["PG001"], "run1": [5000.0]}
            ),
        }
        builder = QuantOutputBuilder(psm_df, config)

        # When
        builder.save_results(lfq_results, "/output", file_format="parquet")

        # Then
        assert mock_write_df.call_count == 1

    def test_annotate_precursor(self, psm_df, config):
        """Given precursor-level LFQ dataframe, when annotated, then adds pg, sequence, mods, mod_sites, and charge."""
        # Given
        lfq_df = pd.DataFrame(
            {"mod_seq_charge_hash": [10, 20, 30], "run1": [1000.0, 2000.0, 3000.0]}
        )
        precursor_config = LFQOutputConfig(
            quant_level="mod_seq_charge_hash",
            level_name=QuantificationLevelName.PRECURSOR,
            intensity_column="precursor_lfq_intensity",
            aggregation_components=["pg", "sequence", "mods", "mod_sites", "charge"],
        )
        builder = QuantOutputBuilder(psm_df, config)

        # When
        annotated_df = builder._annotate_quant_df(lfq_df, psm_df, precursor_config)

        # Then
        expected_df = pd.DataFrame(
            {
                "mod_seq_charge_hash": [10, 20, 30],
                "run1": [1000.0, 2000.0, 3000.0],
                "pg": ["PG001", "PG002", "PG001"],
                "sequence": ["PEPTIDE", "SEQUENCE", "PEPTIDE"],
                "mods": ["", "Oxidation@M", ""],
                "mod_sites": ["", "5", ""],
                "charge": [2, 2, 3],
            }
        )
        pd.testing.assert_frame_equal(annotated_df, expected_df)

    def test_annotate_peptide(self, psm_df, config):
        """Given peptide-level LFQ dataframe, when annotated, then adds pg, sequence, mods, mod_sites but not charge."""
        # Given
        lfq_df = pd.DataFrame(
            {"mod_seq_hash": [1, 2, 3], "run1": [1500.0, 2500.0, 3500.0]}
        )
        peptide_config = LFQOutputConfig(
            quant_level="mod_seq_hash",
            level_name=QuantificationLevelName.PEPTIDE,
            intensity_column="peptide_lfq_intensity",
            aggregation_components=["pg", "sequence", "mods", "mod_sites"],
        )
        builder = QuantOutputBuilder(psm_df, config)

        # When
        annotated_df = builder._annotate_quant_df(lfq_df, psm_df, peptide_config)

        # Then
        expected_df = pd.DataFrame(
            {
                "mod_seq_hash": [1, 2, 3],
                "run1": [1500.0, 2500.0, 3500.0],
                "pg": ["PG001", "PG002", "PG001"],
                "sequence": ["PEPTIDE", "SEQUENCE", "PEPTIDE"],
                "mods": ["", "Oxidation@M", ""],
                "mod_sites": ["", "5", ""],
            }
        )
        pd.testing.assert_frame_equal(annotated_df, expected_df)

    def test_annotate_protein(self, psm_df, config):
        """Given protein-level LFQ dataframe, when annotated, then returns unchanged with no added annotations."""
        # Given
        lfq_df = pd.DataFrame(
            {
                "pg": ["PG001", "PG002"],
                "run1": [5000.0, 2000.0],
                "run2": [5100.0, 2100.0],
            }
        )
        pg_config = LFQOutputConfig(
            quant_level=QuantificationLevelName.PROTEIN,
            level_name=QuantificationLevelName.PROTEIN,
            intensity_column="pg_lfq_intensity",
            aggregation_components=[QuantificationLevelName.PROTEIN],
        )
        builder = QuantOutputBuilder(psm_df, config)

        # When
        annotated_df = builder._annotate_quant_df(lfq_df, psm_df, pg_config)

        # Then - dataframe should be completely unchanged
        expected_df = pd.DataFrame(
            {
                "pg": ["PG001", "PG002"],
                "run1": [5000.0, 2000.0],
                "run2": [5100.0, 2100.0],
            }
        )
        pd.testing.assert_frame_equal(annotated_df, expected_df)

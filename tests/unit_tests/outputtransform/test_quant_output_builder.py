from unittest.mock import patch

import pandas as pd
import pytest

from alphadia.outputtransform.quant_output_builder import (
    LFQOutputConfig,
    QuantOutputBuilder,
)


@pytest.fixture
def config():
    """Configuration for quantification."""
    return {
        "general": {"thread_count": 4},
        "search_output": {
            "precursor_level_lfq": True,
            "peptide_level_lfq": False,
            "min_k_fragments": 3,
            "min_correlation": 0.5,
            "min_nonnan": 1,
            "num_samples_quadratic": 50,
            "normalize_lfq": True,
            "save_fragment_quant_matrix": False,
            "file_format": "parquet",
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


class TestLFQOutputConfig:
    """Test LFQOutputConfig dataclass."""

    def test_default_values(self):
        """Given minimal config, when initialized, then uses default values."""
        # Given
        config_params = {
            "quant_level": "pg",
            "level_name": "protein",
            "intensity_column": "pg.intensity",
            "aggregation_components": ["pg"],
        }

        # When
        config = LFQOutputConfig(**config_params)

        # Then
        assert config.should_process is True
        assert config.save_fragments is False

    @pytest.mark.parametrize(
        "should_process,save_fragments",
        [(True, True), (False, False), (True, False), (False, True)],
    )
    def test_custom_values(self, should_process, save_fragments):
        """Given custom config values, when initialized, then uses provided values."""
        # Given / When
        config = LFQOutputConfig(
            quant_level="pg",
            level_name="protein",
            intensity_column="pg.intensity",
            aggregation_components=["pg"],
            should_process=should_process,
            save_fragments=save_fragments,
        )

        # Then
        assert config.should_process == should_process
        assert config.save_fragments == save_fragments


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
        "alphadia.outputtransform.fragment_accumulator.FragmentQuantLoader.accumulate_from_folders"
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

    @patch("alphadia.outputtransform.quant_output_builder.QuantBuilder.lfq")
    @patch("alphadia.outputtransform.quant_output_builder.QuantBuilder.filter_frag_df")
    @patch(
        "alphadia.outputtransform.fragment_accumulator.FragmentQuantLoader.accumulate_from_folders"
    )
    def test_build_respects_config_flags(
        self, mock_accumulate, mock_filter, mock_lfq, psm_df, config, feature_dfs
    ):
        """Given config with specific LFQ levels enabled, when build is called, then only processes enabled levels."""
        # Given
        config["search_output"]["precursor_level_lfq"] = True
        config["search_output"]["peptide_level_lfq"] = False

        mock_accumulate.return_value = feature_dfs
        mock_filter.return_value = (
            feature_dfs["intensity"],
            feature_dfs["correlation"],
        )

        def lfq_side_effect(*args, **kwargs):
            group_column = kwargs.get("group_column", "pg")
            if group_column == "mod_seq_charge_hash":
                return pd.DataFrame(
                    {"mod_seq_charge_hash": [10, 20], "run1": [1000.0, 2000.0]}
                )
            return pd.DataFrame({"pg": ["PG001", "PG002"], "run1": [5000.0, 2000.0]})

        mock_lfq.side_effect = lfq_side_effect
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, _ = builder.build(["folder1", "folder2"])

        # Then
        assert "precursor" in lfq_results
        assert "pg" in lfq_results
        assert "peptide" not in lfq_results

    @patch("alphadia.outputtransform.quant_output_builder.QuantBuilder.lfq")
    @patch("alphadia.outputtransform.quant_output_builder.QuantBuilder.filter_frag_df")
    @patch(
        "alphadia.outputtransform.fragment_accumulator.FragmentQuantLoader.accumulate_from_folders"
    )
    def test_build_annotates_non_protein_levels(
        self, mock_accumulate, mock_filter, mock_lfq, psm_df, config, feature_dfs
    ):
        """Given precursor-level quantification, when build is called, then results include sequence annotations."""
        # Given
        mock_accumulate.return_value = feature_dfs
        mock_filter.return_value = (
            feature_dfs["intensity"],
            feature_dfs["correlation"],
        )

        def lfq_side_effect(*args, **kwargs):
            group_column = kwargs.get("group_column", "pg")
            if group_column == "mod_seq_charge_hash":
                return pd.DataFrame(
                    {"mod_seq_charge_hash": [10, 20], "run1": [1000.0, 2000.0]}
                )
            return pd.DataFrame({"pg": ["PG001", "PG002"], "run1": [5000.0, 2000.0]})

        mock_lfq.side_effect = lfq_side_effect
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, _ = builder.build(["folder1", "folder2"])

        # Then
        precursor_result = lfq_results["precursor"]
        assert "pg" in precursor_result.columns
        assert "sequence" in precursor_result.columns
        assert "mods" in precursor_result.columns
        assert "charge" in precursor_result.columns

    @patch("alphadia.outputtransform.utils.write_df")
    def test_save_results_writes_non_empty_results(self, mock_write_df, psm_df, config):
        """Given LFQ results with data, when save_results is called, then writes files to disk."""
        # Given
        lfq_results = {
            "precursor": pd.DataFrame({"mod_seq_charge_hash": [10], "run1": [1000.0]}),
            "pg": pd.DataFrame({"pg": ["PG001"], "run1": [5000.0]}),
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
            "precursor": pd.DataFrame(),
            "pg": pd.DataFrame({"pg": ["PG001"], "run1": [5000.0]}),
        }
        builder = QuantOutputBuilder(psm_df, config)

        # When
        builder.save_results(lfq_results, "/output", file_format="parquet")

        # Then
        assert mock_write_df.call_count == 1

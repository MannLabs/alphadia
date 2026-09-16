from unittest.mock import patch

import numpy as np
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
            "save_fragment_quant_matrix": False,
            "file_format": "parquet",
            "normalization_method": NormalizationMethods.DIRECTLFQ,
            "normalize_directlfq": False,
        },
    }


@pytest.fixture
def psm_df():
    """PSM dataframe with target and decoy precursors; 5 and 6 share a precursor under two protein groups."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4, 5, 6],
            "decoy": [0, 0, 0, 0, 1, 0, 0],
            "pg": ["PG001", "PG002", "PG001", "PG003", "PG001", "PG004", "PG005"],
            "mod_seq_hash": [1, 2, 3, 4, 5, 6, 6],
            "mod_seq_charge_hash": [10, 20, 30, 40, 50, 60, 60],
            "sequence": [
                "PEPTIDE",
                "SEQUENCE",
                "PEPTIDE",
                "PROTEIN",
                "PEPTIDE",
                "SHARED",
                "SHARED",
            ],
            "mods": ["", "Oxidation@M", "", "", "", "", ""],
            "mod_sites": ["", "5", "", "", "", "", ""],
            "charge": [2, 2, 3, 2, 2, 2, 2],
            "run": ["run1", "run1", "run2", "run2", "run1", "run1", "run2"],
        }
    )


def write_fragment_files(tmp_path, precursor_idx, correlation, intensities):
    """Write one frag.parquet per run and return the run folders."""
    n_fragments = len(precursor_idx)
    fragments = pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "number": np.arange(1, n_fragments + 1, dtype=np.uint8),
            "type": np.full(n_fragments, ord("b"), dtype=np.uint8),
            "charge": np.ones(n_fragments, dtype=np.uint8),
            "loss_type": np.zeros(n_fragments, dtype=np.uint8),
            "correlation": correlation,
        }
    )

    folders = []
    for run, intensity in intensities.items():
        folder = tmp_path / run
        folder.mkdir()
        fragments.assign(intensity=intensity).to_parquet(
            folder / "frag.parquet", index=False
        )
        folders.append(str(folder))
    return folders


@pytest.fixture
def quant_folders(tmp_path):
    """Two runs of fragments for precursors 0 to 6.

    The second fragment of precursor 0 does not correlate, precursors 0 and 2 double
    from run1 to run2, precursor 4 is a decoy and 5 and 6 share a precursor hash.
    """
    return write_fragment_files(
        tmp_path,
        precursor_idx=[0, 0, 1, 1, 2, 3, 4, 5, 6],
        correlation=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        intensities={
            "run1": [100.0, 1000.0, 300.0, 500.0, 50.0, 10.0, 999.0, 70.0, 30.0],
            "run2": [200.0, 2000.0, 300.0, 500.0, 100.0, 20.0, 999.0, 70.0, 30.0],
        },
    )


class TestQuantOutputBuilder:
    """Test QuantOutputBuilder workflow."""

    def test_initialization_filters_decoys(self, psm_df, config):
        """Given PSM df with decoys, when initialized, then decoys are filtered for quantification."""
        # Given: psm_df contains 1 decoy (see fixture)

        # When
        builder = QuantOutputBuilder(psm_df, config)

        # Then
        assert len(builder.fragment_loader.psm_df) == 6
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

    def test_build_quantifies_all_levels_from_precursor_sums(
        self, psm_df, config, quant_folders
    ):
        """Given fragment files for two runs, when build is called, then precursors are correlation-weighted fragment sums and peptides and protein groups are estimated from them."""
        # Given
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, psm_df_with_quant = builder.build(quant_folders)

        # Then - the uncorrelated fragment of precursor 10 does not count
        precursor_df = lfq_results[QuantificationLevelName.PRECURSOR].set_index(
            "mod_seq_charge_hash"
        )
        assert precursor_df.loc[10, ["run1", "run2"]].tolist() == pytest.approx(
            [100.0, 200.0]
        )
        assert precursor_df.loc[20, ["run1", "run2"]].tolist() == pytest.approx(
            [800.0, 800.0]
        )
        assert all(
            col in precursor_df.columns
            for col in ["pg", "sequence", "mods", "mod_sites", "charge"]
        )

        # Then - a peptide with a single precursor carries that precursor's quantities
        peptide_df = lfq_results[QuantificationLevelName.PEPTIDE].set_index(
            "mod_seq_hash"
        )
        assert peptide_df.loc[1, ["run1", "run2"]].tolist() == pytest.approx(
            [100.0, 200.0]
        )
        assert "charge" not in peptide_df.columns

        # Then - PG001 combines precursors 10 and 30 and keeps their two-fold change
        pg_df = lfq_results[QuantificationLevelName.PROTEIN].set_index("pg")
        assert pg_df.loc["PG001"].tolist() == pytest.approx([150.0, 300.0])
        assert pg_df.loc["PG002"].tolist() == pytest.approx([800.0, 800.0])
        assert list(pg_df.columns) == ["run1", "run2"]

        assert "precursor_lfq_intensity" in psm_df_with_quant.columns

    def test_build_keeps_a_precursor_shared_by_two_protein_groups_in_both(
        self, psm_df, config, quant_folders
    ):
        """Given one precursor hash under two protein groups, when build is called, then both groups are quantified and the precursor is reported once."""
        # Given
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, _ = builder.build(quant_folders)

        # Then
        pg_df = lfq_results[QuantificationLevelName.PROTEIN].set_index("pg")
        assert pg_df.loc["PG004"].tolist() == pytest.approx([70.0, 70.0])
        assert pg_df.loc["PG005"].tolist() == pytest.approx([30.0, 30.0])

        precursor_df = lfq_results[QuantificationLevelName.PRECURSOR]
        shared = precursor_df[precursor_df["mod_seq_charge_hash"] == 60]
        assert len(shared) == 1
        assert shared[["run1", "run2"]].iloc[0].tolist() == pytest.approx(
            [100.0, 100.0]
        )

    def test_build_quantifies_precursors_even_when_precursor_output_is_disabled(
        self, psm_df, config, quant_folders
    ):
        """Given precursor level output disabled, when build is called, then peptides and protein groups are still derived from the precursor sums."""
        # Given
        config["search_output"]["precursor_level_lfq"] = False
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, _ = builder.build(quant_folders)

        # Then
        assert QuantificationLevelName.PRECURSOR not in lfq_results
        pg_df = lfq_results[QuantificationLevelName.PROTEIN].set_index("pg")
        assert pg_df.loc["PG002"].tolist() == pytest.approx([800.0, 800.0])

    def test_build_returns_empty_when_nothing_was_observed(
        self, psm_df, config, tmp_path
    ):
        """Given fragment files with zero intensities only, when build is called with normalization, then no level is quantified and nothing fails."""
        # Given
        config["search_output"]["normalize_directlfq"] = True
        folders = write_fragment_files(
            tmp_path,
            precursor_idx=[0, 1],
            correlation=[1.0, 1.0],
            intensities={"run1": [0.0, 0.0], "run2": [0.0, 0.0]},
        )
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, result_psm_df = builder.build(folders)

        # Then
        assert lfq_results == {}
        pd.testing.assert_frame_equal(result_psm_df, psm_df)

    def test_build_returns_empty_when_no_fragment_belongs_to_a_precursor(
        self, psm_df, config, tmp_path
    ):
        """Given fragment files of unknown precursors only, when build is called, then no level is quantified."""
        # Given
        folders = write_fragment_files(
            tmp_path,
            precursor_idx=[100],
            correlation=[1.0],
            intensities={"run1": [100.0]},
        )
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, result_psm_df = builder.build(folders)

        # Then
        assert lfq_results == {}
        pd.testing.assert_frame_equal(result_psm_df, psm_df)

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

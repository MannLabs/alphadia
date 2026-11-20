from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphadia.constants.keys import (
    NormalizationMethods,
)
from alphadia.outputtransform.quantification import (
    QuantificationLevelName,
    QuantOutputBuilder,
    quant_builder,
)
from alphadia.outputtransform.quantification.quant_builder import (
    INTENSITY_LIBRARY_COLUMN,
)
from alphadia.outputtransform.quantification.quant_output_builder import (
    LFQOutputConfig,
    _annotate_quant_df,
    merge_quant_levels_to_psm,
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
            "fragment_censoring_correction": False,
            "min_correlation": 0.5,
            "min_nonnan": 1,
            "num_samples_quadratic": 50,
            "file_format": "parquet",
            "normalization_method": NormalizationMethods.DIRECTLFQ,
            "normalize_directlfq": True,
            "intensity_drift_correction": True,
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
            "rt_observed": [100.0, 200.0, 300.0, 400.0, 500.0],
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

    @patch("alphadia.outputtransform.quantification.quant_output_builder.direct_lfq")
    @patch(
        "alphadia.outputtransform.quantification.quant_output_builder.filter_frag_df"
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
        mock_filter.return_value = feature_dfs["intensity"]
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

    @patch("alphadia.outputtransform.quantification.quant_output_builder.direct_lfq")
    @patch(
        "alphadia.outputtransform.quantification.quant_output_builder.filter_frag_df"
    )
    @patch(
        "alphadia.outputtransform.quantification.fragment_accumulator.FragmentQuantLoader.accumulate_from_folders"
    )
    def test_build_annotates_from_targets_when_decoy_shares_hash(
        self, mock_accumulate, mock_filter, mock_direct_lfq, psm_df, config, feature_dfs
    ):
        """Given a decoy listed first with its target's hashes, when build is called, then the target's precursor and peptide rows keep the target's sequence."""
        # Given
        decoy_row = psm_df.iloc[[4]].assign(
            mod_seq_hash=1, mod_seq_charge_hash=10, sequence="PDPTIDE"
        )
        psm_df = pd.concat([decoy_row, psm_df.iloc[:4]], ignore_index=True)
        mock_accumulate.return_value = feature_dfs
        mock_filter.return_value = feature_dfs["intensity"]
        mock_direct_lfq.side_effect = lfq_side_effect
        builder = QuantOutputBuilder(psm_df, config)

        # When
        lfq_results, _ = builder.build(["folder1", "folder2"])

        # Then
        precursor_result = lfq_results[QuantificationLevelName.PRECURSOR]
        peptide_result = lfq_results[QuantificationLevelName.PEPTIDE]
        assert (
            precursor_result.loc[
                precursor_result["mod_seq_charge_hash"] == 10, "sequence"
            ].item()
            == "PEPTIDE"
        )
        assert (
            peptide_result.loc[peptide_result["mod_seq_hash"] == 1, "sequence"].item()
            == "PEPTIDE"
        )

    @patch("alphadia.outputtransform.quantification.quant_output_builder.direct_lfq")
    @patch(
        "alphadia.outputtransform.quantification.quant_output_builder.filter_frag_df"
    )
    @patch(
        "alphadia.outputtransform.quantification.fragment_accumulator.FragmentQuantLoader.accumulate_from_folders"
    )
    def test_build_does_not_give_decoys_the_quantities_of_their_target(
        self, mock_accumulate, mock_filter, mock_direct_lfq, psm_df, config, feature_dfs
    ):
        """Given a decoy with the hashes and run of a target, when build is called, then only the target row gets the quantities."""
        # Given: the decoy shares the precursor, peptide and protein keys of target 0 in run1
        psm_df.loc[4, ["mod_seq_hash", "mod_seq_charge_hash"]] = [1, 10]
        mock_accumulate.return_value = feature_dfs
        mock_filter.return_value = feature_dfs["intensity"]
        mock_direct_lfq.side_effect = lfq_side_effect
        builder = QuantOutputBuilder(psm_df, config)

        # When
        _, psm_df_with_quant = builder.build(["folder1", "folder2"])

        # Then
        lfq_intensity_columns = [
            "precursor_lfq_intensity",
            "peptide_lfq_intensity",
            "pg_lfq_intensity",
        ]
        target_row = psm_df_with_quant[psm_df_with_quant["precursor_idx"] == 0]
        decoy_row = psm_df_with_quant[psm_df_with_quant["precursor_idx"] == 4]
        assert target_row[lfq_intensity_columns].notna().all(axis=None)
        assert decoy_row[lfq_intensity_columns].isna().all(axis=None)

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

        # When
        annotated_df = _annotate_quant_df(lfq_df, psm_df, precursor_config)

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

        # When
        annotated_df = _annotate_quant_df(lfq_df, psm_df, peptide_config)

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

        # When
        annotated_df = _annotate_quant_df(lfq_df, psm_df, pg_config)

        # Then - dataframe should be completely unchanged
        expected_df = pd.DataFrame(
            {
                "pg": ["PG001", "PG002"],
                "run1": [5000.0, 2000.0],
                "run2": [5100.0, 2100.0],
            }
        )
        pd.testing.assert_frame_equal(annotated_df, expected_df)


class TestMergeQuantLevelsToPsm:
    """Quantities of every level are merged onto the target PSMs."""

    @pytest.fixture
    def level_dfs(self):
        return {
            QuantificationLevelName.PRECURSOR: pd.DataFrame(
                {"mod_seq_charge_hash": ["A1"], "run1": [100.0]}
            ),
            QuantificationLevelName.PEPTIDE: pd.DataFrame(
                {"mod_seq_hash": ["A"], "run1": [400.0]}
            ),
            QuantificationLevelName.PROTEIN: pd.DataFrame(
                {"pg": ["PG1"], "run1": [700.0]}
            ),
        }

    def test_merges_all_levels(self, level_dfs):
        """Given quantities on all levels, when merged, then the target gets one intensity column per level."""
        # Given
        psm_df = pd.DataFrame(
            {
                "mod_seq_charge_hash": ["A1"],
                "mod_seq_hash": ["A"],
                "pg": ["PG1"],
                "run": ["run1"],
                "decoy": [0],
            }
        )

        # When
        result = merge_quant_levels_to_psm(psm_df, level_dfs)

        # Then
        expected = psm_df.assign(
            precursor_lfq_intensity=[100.0],
            peptide_lfq_intensity=[400.0],
            pg_lfq_intensity=[700.0],
        )
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("decoy_dtype", [int, bool])
    def test_decoys_keep_no_quantities(self, level_dfs, decoy_dtype):
        """Given a decoy with the keys of a target, when merged, then only the target gets quantities."""
        # Given
        psm_df = pd.DataFrame(
            {
                "mod_seq_charge_hash": ["A1", "A1"],
                "mod_seq_hash": ["A", "A"],
                "pg": ["PG1", "PG1"],
                "run": ["run1", "run1"],
                "decoy": np.array([0, 1], dtype=decoy_dtype),
            }
        )

        # When
        result = merge_quant_levels_to_psm(psm_df, level_dfs)

        # Then
        assert result.loc[0, "precursor_lfq_intensity"] == 100.0
        assert (
            result.loc[1, ["precursor_lfq_intensity", "pg_lfq_intensity"]].isna().all()
        )


def _varmodel_feature_dfs():
    """Two precursors of PG001 and one of PG002, four fragments each, run2 at twice run1.

    The third fragment of precursor 0 is interfered in run2 (16-fold instead of 2-fold).
    """
    intensity_library = np.array([1.0, 0.6, 0.3, 0.1])
    precursor_level = {0: 1000.0, 1: 4000.0, 2: 500.0}
    rows = []
    for precursor_idx, level in precursor_level.items():
        for i_fragment, fragment_intensity_library in enumerate(intensity_library):
            run1 = level * fragment_intensity_library
            run2 = 2 * run1
            if precursor_idx == 0 and i_fragment == 2:
                run2 = 16 * run1
            rows.append(
                {
                    "precursor_idx": precursor_idx,
                    "ion": precursor_idx * 10 + i_fragment,
                    "run1": run1,
                    "run2": run2,
                    INTENSITY_LIBRARY_COLUMN: fragment_intensity_library,
                    "correlation": 0.9,
                    "pg": "PG001" if precursor_idx < 2 else "PG002",
                    "mod_seq_hash": precursor_idx + 1,
                    "mod_seq_charge_hash": (precursor_idx + 1) * 10,
                }
            )
    df = pd.DataFrame(rows)
    metadata = [
        "precursor_idx",
        "ion",
        "pg",
        "mod_seq_hash",
        "mod_seq_charge_hash",
        INTENSITY_LIBRARY_COLUMN,
    ]

    def matrix(column):
        out = df[metadata].copy()
        out["run1"] = df[column] if column != "intensity" else df["run1"]
        out["run2"] = df[column] if column != "intensity" else df["run2"]
        return out

    return {"intensity": matrix("intensity"), "correlation": matrix("correlation")}


class TestVarModelQuantification:
    """The variance-model precursor quantities feed directLFQ on every level."""

    @pytest.fixture
    def varmodel_config(self, config):
        config["search_output"]["normalization_method"] = (
            NormalizationMethods.VARMODEL_DIRECTLFQ
        )
        config["search_output"]["normalize_directlfq"] = False
        config["search_output"]["min_k_fragments"] = 4
        config["search_output"]["min_nonnan"] = 3
        return config

    @pytest.fixture
    def varmodel_psm_df(self):
        return pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "decoy": [0, 0, 0],
                "pg": ["PG001", "PG001", "PG002"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
                "sequence": ["PEPTIDE", "SEQUENCE", "PROTEIN"],
                "mods": ["", "", ""],
                "mod_sites": ["", "", ""],
                "charge": [2, 2, 2],
                "run": ["run1", "run1", "run2"],
            }
        )

    def test_loads_the_library_intensity(self, varmodel_psm_df, varmodel_config):
        # Given / When
        builder = QuantOutputBuilder(varmodel_psm_df, varmodel_config)

        # Then the library intensity is loaded as a constant ion feature, not as a run matrix
        assert (
            INTENSITY_LIBRARY_COLUMN in builder.fragment_loader.constant_feature_columns
        )
        assert INTENSITY_LIBRARY_COLUMN not in builder.fragment_loader.columns

    def test_every_level_is_rolled_up_from_the_varmodel_precursors(
        self, varmodel_psm_df, varmodel_config
    ):
        # Given
        builder = QuantOutputBuilder(varmodel_psm_df, varmodel_config)
        quantlevel_configs = builder._requested_quant_level_configs()

        # When
        level_dfs = builder._quantify_with_varmodel_precursors(
            quantlevel_configs, _varmodel_feature_dfs()
        )

        # Then the interfered fragment does not distort the 2-fold change
        precursor_df = level_dfs[QuantificationLevelName.PRECURSOR].set_index(
            "mod_seq_charge_hash"
        )
        np.testing.assert_allclose(
            precursor_df["run2"] / precursor_df["run1"], 2.0, rtol=0.05
        )
        # and the precursor quantities are on the summed fragment scale
        np.testing.assert_allclose(
            precursor_df.loc[20, "run1"], 4000.0 * 2.0, rtol=0.05
        )
        # and the protein with only two precursors is quantified despite min_nonnan 3
        protein_df = level_dfs[QuantificationLevelName.PROTEIN].set_index("pg")
        assert set(protein_df.index) == {"PG001", "PG002"}
        np.testing.assert_allclose(
            protein_df["run2"] / protein_df["run1"], 2.0, rtol=0.05
        )
        peptide_df = level_dfs[QuantificationLevelName.PEPTIDE]
        assert set(peptide_df["mod_seq_hash"]) == {1, 2, 3}

    def test_every_level_normalizes_the_same_precursor_table(
        self, varmodel_psm_df, varmodel_config
    ):
        # Given
        varmodel_config["search_output"]["normalize_directlfq"] = True
        builder = QuantOutputBuilder(varmodel_psm_df, varmodel_config)

        # When
        with patch(
            "alphadia.outputtransform.quantification.quant_builder._normalize_ion_table",
            wraps=quant_builder._normalize_ion_table,
        ) as spy_normalize:
            level_dfs = builder._quantify_with_varmodel_precursors(
                builder._requested_quant_level_configs(), _varmodel_feature_dfs()
            )

        # Then directLFQ normalizes per level, the ion table being the same precursors each time
        assert spy_normalize.call_count == len(level_dfs)
        assert set(level_dfs) == {
            QuantificationLevelName.PRECURSOR,
            QuantificationLevelName.PEPTIDE,
            QuantificationLevelName.PROTEIN,
        }

    def test_directlfq_rollup_uses_the_fragments(self, varmodel_psm_df, config):
        # Given the directLFQ precursor rollup
        builder = QuantOutputBuilder(varmodel_psm_df, config)

        # When
        with patch(
            "alphadia.outputtransform.quantification.quant_output_builder.varmodel_precursor_table"
        ) as mock_varmodel:
            builder._quantify_with_directlfq(
                builder._requested_quant_level_configs(), _varmodel_feature_dfs()
            )

        # Then
        mock_varmodel.assert_not_called()
        assert INTENSITY_LIBRARY_COLUMN not in builder.fragment_loader.columns
        assert (
            INTENSITY_LIBRARY_COLUMN
            not in builder.fragment_loader.constant_feature_columns
        )

from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd
import pytest

from alphadia.constants.keys import NormalizationMethods
from alphadia.outputtransform.quantification.quant_builder import QuantBuilder


@pytest.fixture
def psm_df():
    """PSM dataframe for quantification."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "pg": ["PG001", "PG002", "PG003"],
            "mod_seq_hash": [1, 2, 3],
            "mod_seq_charge_hash": [10, 20, 30],
        }
    )


@pytest.fixture
def filtering_data():
    """Intensity and quality dataframes for filtering tests."""
    intensity_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 0, 1, 1, 1],
            "ion": [100, 101, 102, 200, 201, 202],
            "run1": [1000.0, 2000.0, 3000.0, 1500.0, 2500.0, 3500.0],
            "run2": [1100.0, 2100.0, 3100.0, 1600.0, 2600.0, 3600.0],
            "pg": ["PG001", "PG001", "PG001", "PG002", "PG002", "PG002"],
            "mod_seq_hash": [1, 1, 1, 2, 2, 2],
            "mod_seq_charge_hash": [10, 10, 10, 20, 20, 20],
        }
    )

    quality_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 0, 1, 1, 1],
            "ion": [100, 101, 102, 200, 201, 202],
            "run1": [0.9, 0.7, 0.3, 0.8, 0.6, 0.4],
            "run2": [0.8, 0.6, 0.4, 0.9, 0.7, 0.5],
            "pg": ["PG001", "PG001", "PG001", "PG002", "PG002", "PG002"],
            "mod_seq_hash": [1, 1, 1, 2, 2, 2],
            "mod_seq_charge_hash": [10, 10, 10, 20, 20, 20],
        }
    )

    return intensity_df, quality_df


@pytest.fixture
def lfq_data():
    """Data for LFQ tests."""
    intensity_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 1, 1, 2, 2],
            "ion": [100, 100, 101, 101, 102, 102],
            "run1": [1000.0, 0.0, 2000.0, 0.0, 3000.0, 1500.0],
            "run2": [0.0, 1100.0, 0.0, 2100.0, 3100.0, 1600.0],
            "run3": [1200.0, 1300.0, 2200.0, 2300.0, 0.0, 0.0],
            "pg": ["PG001", "PG001", "PG002", "PG002", "PG003", "PG003"],
            "mod_seq_hash": [1, 1, 2, 2, 3, 3],
            "mod_seq_charge_hash": [10, 10, 20, 20, 30, 30],
        }
    )

    quality_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 1, 1, 2, 2],
            "ion": [100, 100, 101, 101, 102, 102],
            "run1": [0.9, 0.0, 0.8, 0.0, 0.7, 0.8],
            "run2": [0.0, 0.9, 0.0, 0.8, 0.7, 0.8],
            "run3": [0.8, 0.9, 0.8, 0.9, 0.0, 0.0],
            "pg": ["PG001", "PG001", "PG002", "PG002", "PG003", "PG003"],
            "mod_seq_hash": [1, 1, 2, 2, 3, 3],
            "mod_seq_charge_hash": [10, 10, 20, 20, 30, 30],
        }
    )

    return {"intensity": intensity_df, "correlation": quality_df}


@pytest.fixture
def search_config():
    return {
        "search_output": {
            "num_cores": 4,
            "num_samples_quadratic": 50,
            "min_nonnan": 1,
            "min_k_fragments": 1,
            "min_correlation": 0,
            "normalize_lfq": True,
        },
        "general": {
            "thread_count": 1,
        },
    }


@pytest.fixture
def lfq_config():
    @dataclass
    class LFQOutputConfig:
        quant_level: str
        normalization_method: str | None = NormalizationMethods.DIRECT_LFQ

    def _create_config(
        quant_level: str,
        normalization_method: str = NormalizationMethods.DIRECT_LFQ,
    ):
        return LFQOutputConfig(
            quant_level=quant_level,
            normalization_method=normalization_method,
        )

    return _create_config


@pytest.fixture
def ms2_features():
    """MS2 features consisting of DataFrames for helper function tests."""
    df = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "run1": [
                15453501,
                3,
                15453503,
                15453502,
                15453502,
                1,
                15453502,
                15453501,
                15453501,
            ]
            * 2,
            "run2": [
                15453501,
                2,
                15453502,
                15453501,
                15453501,
                5,
                15453501,
                15453501,
                15453502,
            ]
            * 2,
            "run3": [
                15453502,
                2,
                15453505,
                15453501,
                15453501,
                3,
                15453503,
                15453502,
                15453503,
            ]
            * 2,
            "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
            "mod_seq_hash": [
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
            ]
            * 2,
            "mod_seq_charge_hash": [
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
            ]
            * 2,
        }
    )

    # Correlation data
    df_corr = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "run1": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
            "run2": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
            "run3": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
            "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
            "mod_seq_hash": [
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
            ]
            * 2,
            "mod_seq_charge_hash": [
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
            ]
            * 2,
        }
    )

    # Mass error data
    mass_error_data = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "run1": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
            "run2": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
            "run3": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
            "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
            "mod_seq_hash": [
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
            ]
            * 2,
            "mod_seq_charge_hash": [
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
            ]
            * 2,
        }
    )

    # Height data
    height_data = pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
            "run1": [114, 144, 114, 113, 114, 514, 134, 144, 131] * 2,
            "run2": [184, 114, 144, 114, 144, 114, 134, 115, 321] * 2,
            "run3": [114, 124, 114, 114, 164, 144, 114, 114, 411] * 2,
            "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
            "mod_seq_hash": [
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
            ]
            * 2,
            "mod_seq_charge_hash": [
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
            ]
            * 2,
        }
    )

    return {
        "intensity": df,
        "correlation": df_corr,
        "mass_error": mass_error_data,
        "height": height_data,
    }


@pytest.fixture
def psm_file():
    """PSM file for helper function tests."""
    return pd.DataFrame(
        {
            "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9]
            * 6,  # 3 runs × 2 protein groups
            "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 6,
            "pg": ["TNAA_ECOLI"] * 27 + ["TNAB_ECOLI"] * 27,
            "mod_seq_hash": [
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                6831315783892314113,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
                1784898696230645364,
            ]
            * 6,
            "mod_seq_charge_hash": [
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3157800000000000000,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
                3178489869623064536,
            ]
            * 6,
            "run": ["run1"] * 9
            + ["run2"] * 9
            + ["run3"] * 9
            + ["run1"] * 9
            + ["run2"] * 9
            + ["run3"] * 9,
            "intensity": [
                15453501,
                3,
                15453503,
                15453502,
                15453502,
                1,
                15453502,
                15453501,
                15453501,
                15453501,
                2,
                15453502,
                15453501,
                15453501,
                5,
                15453501,
                15453501,
                15453502,
                15453502,
                4,
                15453505,
                15453501,
                15453501,
                3,
                15453503,
                15453502,
                15453503,
            ]
            * 2,
            "delta_rt": [
                15453501,
                2,
                15453503,
                15453502,
                15453502,
                1,
                15453502,
                15453501,
                15453501,
                15453501,
                1,
                15453502,
                15453501,
                15453501,
                5,
                15453501,
                15453501,
                15453502,
                15453502,
                2,
                15453505,
                15453501,
                15453501,
                3,
                15453503,
                15453502,
                15453503,
            ]
            * 2,
        }
    )


class TestFilterFragDf:
    """Test fragment filtering by quality."""

    def test_filters_by_top_n_per_group(self, filtering_data, psm_df):
        """Given fragments from multiple groups, when filtered by top N, then keeps top N per group."""
        # Given
        intensity_df, quality_df = filtering_data
        builder = QuantBuilder(psm_df)

        # When
        filtered_intensity, _ = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.5, top_n=2
        )

        # Then
        assert len(filtered_intensity) == 4
        pg001_ions = filtered_intensity[filtered_intensity["pg"] == "PG001"][
            "ion"
        ].values
        pg002_ions = filtered_intensity[filtered_intensity["pg"] == "PG002"][
            "ion"
        ].values
        assert len(pg001_ions) == 2
        assert len(pg002_ions) == 2

    def test_filters_by_min_correlation(self, filtering_data, psm_df):
        """Given fragments with varying quality, when filtered by correlation, then keeps high-quality fragments."""
        # Given
        intensity_df, quality_df = filtering_data
        builder = QuantBuilder(psm_df)

        # When
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.6, top_n=1
        )

        # Then - Should keep top 1 OR above 0.6
        kept_ions = set(filtered_intensity["ion"].values)
        assert kept_ions == {100, 101, 200, 201}

    @pytest.mark.parametrize(
        "group_column,expected_groups",
        [
            ("pg", ["PG001", "PG002"]),
            ("mod_seq_hash", [1, 2]),
        ],
    )
    def test_respects_group_column(
        self, filtering_data, psm_df, group_column, expected_groups
    ):
        """Given custom group column, when filtered, then groups by specified column."""
        # Given
        intensity_df, quality_df = filtering_data
        builder = QuantBuilder(psm_df)

        # When
        filtered_intensity, _ = builder.filter_frag_df(
            intensity_df,
            quality_df,
            min_correlation=2.0,
            top_n=1,
            group_column=group_column,
        )

        # Then
        groups = filtered_intensity[group_column].unique()
        assert set(groups) == set(expected_groups)

    def test_handles_empty_input(self, psm_df):
        """Given empty dataframes, when filtered, then returns empty dataframes."""
        # Given
        empty_intensity = pd.DataFrame(
            columns=[
                "precursor_idx",
                "ion",
                "run1",
                "pg",
                "mod_seq_hash",
                "mod_seq_charge_hash",
            ]
        )
        empty_quality = empty_intensity.copy()
        builder = QuantBuilder(psm_df)

        # When
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            empty_intensity, empty_quality, min_correlation=0.5, top_n=3
        )

        # Then
        assert len(filtered_intensity) == 0
        assert "total" in filtered_quality.columns
        assert "rank" in filtered_quality.columns


class TestLfq:
    """Test label-free quantification."""

    @pytest.fixture
    def mock_directlfq(self):
        """Mock directLFQ functions."""
        with (
            patch(
                "alphadia.outputtransform.quantification.quant_builder.lfqconfig"
            ) as mock_config,
            patch(
                "alphadia.outputtransform.quantification.quant_builder.lfqutils"
            ) as mock_utils,
            patch(
                "alphadia.outputtransform.quantification.quant_builder.lfqnorm"
            ) as mock_norm,
            patch(
                "alphadia.outputtransform.quantification.quant_builder.lfqprot_estimation"
            ) as mock_prot,
        ):
            mock_utils.index_and_log_transform_input_df.return_value = pd.DataFrame(
                {"pg": ["PG001", "PG002"], "ion": [100, 101], "run1": [10.0, 11.0]}
            )
            mock_utils.remove_allnan_rows_input_df.return_value = pd.DataFrame(
                {"pg": ["PG001", "PG002"], "ion": [100, 101], "run1": [10.0, 11.0]}
            )
            mock_norm_manager = (
                mock_norm.NormalizationManagerSamplesOnSelectedProteins.return_value
            )
            mock_norm_manager.complete_dataframe = pd.DataFrame(
                {"pg": ["PG001", "PG002"], "ion": [100, 101], "run1": [9.8, 10.8]}
            )
            mock_prot.estimate_protein_intensities.return_value = (
                pd.DataFrame({"pg": ["PG001", "PG002"], "run1": [20.0, 21.0]}),
                None,
            )

            yield {
                "config": mock_config,
                "utils": mock_utils,
                "norm": mock_norm,
                "prot": mock_prot,
            }

    def test_performs_quantification(
        self, lfq_data, psm_df, lfq_config, search_config, mock_directlfq
    ):
        """Given fragment data, when LFQ is run, then returns protein quantification."""
        # Given
        feature_dfs_dict = lfq_data
        builder = QuantBuilder(psm_df)
        lfq_config = lfq_config("pg", NormalizationMethods.DIRECT_LFQ)
        config = search_config

        # When
        result_df = builder.lfq(feature_dfs_dict, lfq_config, config)

        # Then
        assert isinstance(result_df, pd.DataFrame)
        assert "pg" in result_df.columns
        assert len(result_df) == 2

    def test_configures_directlfq(
        self, lfq_data, psm_df, lfq_config, search_config, mock_directlfq
    ):
        """Given LFQ parameters, when run, then configures directLFQ correctly."""
        # Given
        feature_dfs_dict = lfq_data
        builder = QuantBuilder(psm_df)
        lfq_config = lfq_config("pg", NormalizationMethods.DIRECT_LFQ)
        config = search_config

        # When
        builder.lfq(feature_dfs_dict, lfq_config, config)

        # Then
        mock_config = mock_directlfq["config"]
        mock_config.set_global_protein_and_ion_id.assert_called_once_with(
            protein_id="pg", quant_id="ion"
        )

    @pytest.mark.parametrize("norm_method", [NormalizationMethods.DIRECT_LFQ, None])
    def test_respects_normalization_flag(
        self, lfq_data, psm_df, lfq_config, search_config, mock_directlfq, norm_method
    ):
        """Given normalization flag, when LFQ is run, then applies normalization conditionally."""
        # Given
        feature_dfs_dict = lfq_data
        builder = QuantBuilder(psm_df)
        lfq_config = lfq_config("pg", norm_method)
        config = search_config

        # When
        builder.lfq(feature_dfs_dict, lfq_config, config)

        # Then
        mock_norm = mock_directlfq["norm"]
        if norm_method == NormalizationMethods.DIRECT_LFQ:
            mock_norm.NormalizationManagerSamplesOnSelectedProteins.assert_called_once()
        else:
            mock_norm.NormalizationManagerSamplesOnSelectedProteins.assert_not_called()

    def test_handles_custom_group_column(
        self, lfq_data, psm_df, lfq_config, search_config, mock_directlfq
    ):
        """Given custom group column, when LFQ is run, then groups by specified column."""
        # Given
        feature_dfs_dict = lfq_data
        builder = QuantBuilder(psm_df)
        lfq_config = lfq_config("mod_seq_hash", NormalizationMethods.DIRECT_LFQ)
        config = search_config

        # When
        builder.lfq(feature_dfs_dict, lfq_config, config)

        # Then
        mock_config = mock_directlfq["config"]
        mock_config.set_global_protein_and_ion_id.assert_called_with(
            protein_id="mod_seq_hash", quant_id="ion"
        )

        mock_utils = mock_directlfq["utils"]
        called_df = mock_utils.index_and_log_transform_input_df.call_args[0][0]
        assert "mod_seq_hash" in called_df.columns
        assert "pg" not in called_df.columns

    def test_quantselect_should_perform_basic_quantification(
        self, ms2_features, psm_file, lfq_config, search_config
    ):
        """Test that lfq performs basic label-free quantification with quantselect."""
        # given
        feature_dfs_dict = ms2_features
        builder = QuantBuilder(psm_file.assign(decoy=0))
        lfq_config = lfq_config("pg", NormalizationMethods.QUANT_SELECT)
        config = search_config

        # when
        result_df = builder.lfq(feature_dfs_dict, lfq_config, config)
        # then
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2  # Three protein groups
        assert "pg" in result_df.columns
        assert "run1" in result_df.columns
        assert "run2" in result_df.columns
        assert "run3" in result_df.columns

        # Verify expected protein groups
        assert set(result_df["pg"]) == {"TNAA_ECOLI", "TNAB_ECOLI"}

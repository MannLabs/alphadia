import platform
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphadia.constants.keys import NormalizationMethods
from alphadia.outputtransform.quantification.quant_builder import (
    FRAGMENT_CORRELATION_POWER,
    QuantBuilder,
    compute_fragment_weights,
    compute_mean_correlation,
)


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
def precursor_df():
    """Precursor quantities with a two-fold change between runs; PG001 has two precursors."""
    return pd.DataFrame(
        {
            "mod_seq_charge_hash": [10, 20, 30],
            "mod_seq_hash": [1, 2, 3],
            "pg": ["PG001", "PG001", "PG002"],
            "run1": [101.0, 50.0, 800.0],
            "run2": [202.0, 100.0, 1600.0],
        }
    )


RUN_COLUMNS = ["run1", "run2", "run3", "run4"]


@pytest.fixture
def fragment_sum_data():
    """Two precursors over four runs: precursor 10 has three fragments, precursor 20 a single one.

    Intensities of precursor 10 double from run to run so that cross-run ratios are exact.
    Ion 102 is missing in run4, ion 200 is missing in run2.
    """
    intensity_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 0, 1],
            "ion": [100, 101, 102, 200],
            "run1": [100.0, 10.0, 1000.0, 50.0],
            "run2": [200.0, 20.0, 2000.0, 0.0],
            "run3": [400.0, 40.0, 4000.0, 60.0],
            "run4": [800.0, 80.0, 0.0, 70.0],
            "pg": ["PG001", "PG001", "PG001", "PG002"],
            "mod_seq_hash": [1, 1, 1, 2],
            "mod_seq_charge_hash": [10, 10, 10, 20],
        }
    )
    correlation_df = intensity_df.copy()
    correlation_df[RUN_COLUMNS] = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    return intensity_df, correlation_df


@pytest.fixture
def search_config():
    return {
        "search_output": {
            "num_cores": 4,
            "num_samples_quadratic": 50,
            "min_nonnan": 1,
            "normalization_method": NormalizationMethods.DIRECTLFQ,
            "normalize_directlfq": True,
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
        normalization_method: str | None = NormalizationMethods.DIRECTLFQ

    def _create_config(
        quant_level: str,
        normalization_method: str = NormalizationMethods.DIRECTLFQ,
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
            "precursor.idx": [1, 2, 3, 4, 5, 6, 7, 8, 9]
            * 6,  # 3 runs × 2 protein groups
            "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 6,
            "pg.proteins": ["TNAA_ECOLI"] * 27 + ["TNAB_ECOLI"] * 27,
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
            "precursor.mod_seq_charge_hash": [
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
            "raw.name": ["run1"] * 9
            + ["run2"] * 9
            + ["run3"] * 9
            + ["run1"] * 9
            + ["run2"] * 9
            + ["run3"] * 9,
            "precursor.intensity": [
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
            "precursor.rt.library": [100.0] * 54,
            "precursor.rt.observed": [101.0] * 54,
        }
    )


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
        self, precursor_df, psm_df, lfq_config, search_config, mock_directlfq
    ):
        """Given precursor quantities, when direct_lfq is run, then returns protein quantification."""
        # Given
        builder = QuantBuilder(psm_df)

        # When
        result_df = builder.direct_lfq(precursor_df, lfq_config("pg"), search_config)

        # Then
        assert isinstance(result_df, pd.DataFrame)
        assert "pg" in result_df.columns
        assert len(result_df) == 2

    def test_configures_directlfq(
        self, precursor_df, psm_df, lfq_config, search_config, mock_directlfq
    ):
        """Given precursor quantities, when run, then precursors are configured as directLFQ's ions."""
        # Given
        builder = QuantBuilder(psm_df)

        # When
        builder.direct_lfq(precursor_df, lfq_config("pg"), search_config)

        # Then
        mock_config = mock_directlfq["config"]
        mock_config.set_global_protein_and_ion_id.assert_called_once_with(
            protein_id="pg", quant_id="ion"
        )

    def test_handles_custom_group_column(
        self, precursor_df, psm_df, lfq_config, search_config, mock_directlfq
    ):
        """Given custom group column, when LFQ is run, then groups by specified column."""
        # Given
        builder = QuantBuilder(psm_df)

        # When
        builder.direct_lfq(precursor_df, lfq_config("mod_seq_hash"), search_config)

        # Then
        mock_utils = mock_directlfq["utils"]
        called_df = mock_utils.index_and_log_transform_input_df.call_args[0][0]
        assert list(called_df.columns) == ["ion", "mod_seq_hash", "run1", "run2"]

    def test_estimates_groups_from_precursor_quantities(
        self, precursor_df, psm_df, lfq_config, search_config
    ):
        """Given precursor quantities with a two-fold change between runs, when LFQ is run on the protein level, then groups are estimated from the precursors without a second normalization."""
        # Given
        builder = QuantBuilder(psm_df)

        # When
        result_df = builder.direct_lfq(precursor_df, lfq_config("pg"), search_config)

        # Then - a second normalization would have removed the two-fold change
        pg_df = result_df.set_index("pg")
        assert pg_df.loc["PG002"].tolist() == pytest.approx([800.0, 1600.0])
        assert pg_df.loc["PG001"].tolist() == pytest.approx([151.0, 302.0])

    def test_merges_a_precursor_listed_under_two_protein_groups(
        self, psm_df, lfq_config, search_config
    ):
        """Given one precursor under two protein groups, when LFQ is run on the peptide level, then both entries count once for the peptide."""
        # Given
        precursor_df = pd.DataFrame(
            {
                "mod_seq_charge_hash": [10, 10],
                "mod_seq_hash": [1, 1],
                "pg": ["PG001", "PG002"],
                "run1": [100.0, 50.0],
                "run2": [100.0, 50.0],
            }
        )
        builder = QuantBuilder(psm_df)

        # When
        result_df = builder.direct_lfq(
            precursor_df, lfq_config("mod_seq_hash"), search_config
        )

        # Then
        assert result_df.set_index("mod_seq_hash").loc[1].tolist() == pytest.approx(
            [150.0, 150.0]
        )

    def test_returns_empty_frame_when_nothing_was_observed(
        self, precursor_df, psm_df, lfq_config, search_config
    ):
        """Given precursors with zero intensity everywhere, when LFQ is run, then an empty frame is returned instead of failing inside directLFQ."""
        # Given
        precursor_df[["run1", "run2"]] = 0.0
        builder = QuantBuilder(psm_df)

        # When
        result_df = builder.direct_lfq(precursor_df, lfq_config("pg"), search_config)

        # Then
        assert result_df.empty

    @pytest.mark.skipif(
        sys.platform == "darwin"
        and platform.machine() == "x86_64"
        and sys.version_info[:2] in [(3, 11), (3, 12)],
        reason="Fails with 'joblib.externals.loky.process_executor.BrokenProcessPool: A task has failed to un-serialize'",  # TODO: fix
    )
    def test_quantselect_should_perform_basic_quantification(
        self, ms2_features, psm_file, lfq_config
    ):
        """Test that quantselect_lfq performs basic label-free quantification with quantselect."""
        # given
        feature_dfs_dict = ms2_features
        builder = QuantBuilder(psm_file.assign(**{"precursor.decoy": 0}))
        lfq_config = lfq_config("pg", NormalizationMethods.QUANTSELECT)

        # when
        result_df = builder.quantselect_lfq(feature_dfs_dict, lfq_config)
        # then
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2  # Three protein groups
        assert "pg" in result_df.columns
        assert "run1" in result_df.columns
        assert "run2" in result_df.columns
        assert "run3" in result_df.columns

        # Verify expected protein groups
        assert set(result_df["pg"]) == {"TNAA_ECOLI", "TNAB_ECOLI"}


class TestComputeMeanCorrelation:
    """Test the per-fragment mean correlation."""

    def test_averages_over_observed_runs_only(self):
        """Given fragments missing in some runs, when the mean is computed, then only observed runs count and a never observed fragment gets 0."""
        # Given
        intensity_df = pd.DataFrame(
            {
                "ion": [100, 101],
                "run1": [100.0, 0.0],
                "run2": [0.0, 0.0],
                "run3": [300.0, 0.0],
            }
        )
        correlation_df = pd.DataFrame(
            {
                "ion": [100, 101],
                "run1": [0.9, 0.5],
                "run2": [0.0, 0.5],
                "run3": [0.7, 0.5],
            }
        )

        # When
        mean_correlation = compute_mean_correlation(intensity_df, correlation_df)

        # Then
        assert mean_correlation.tolist() == pytest.approx([0.8, 0.0])

    def test_rejects_misaligned_tables(self, fragment_sum_data):
        """Given correlation rows in a different order, when the mean is computed, then it fails loudly."""
        # Given
        intensity_df, correlation_df = fragment_sum_data

        # When / Then
        with pytest.raises(ValueError, match="same order"):
            compute_mean_correlation(intensity_df, correlation_df.iloc[::-1])


class TestComputeFragmentWeights:
    """Test the weights relative to the best fragment of a precursor."""

    def test_weights_are_relative_to_the_best_fragment_of_each_precursor(self):
        """Given two precursors, when weighted, then the best fragment of each gets 1, the others the power of their correlation, and a precursor without correlating fragments gets 1 everywhere."""
        # Given
        mean_correlation = np.array([1.0, 0.5, 0.0, 0.0])
        precursor_hash = np.array([10, 10, 20, 20])

        # When
        weights = compute_fragment_weights(mean_correlation, precursor_hash)

        # Then
        assert weights.tolist() == pytest.approx(
            [1.0, 0.5**FRAGMENT_CORRELATION_POWER, 1.0, 1.0]
        )


class TestSumFragmentsToPrecursors:
    """Test the correlation-weighted fragment sum used as the precursor rollup."""

    @pytest.fixture
    def sum_config(self, search_config):
        search_config["search_output"]["normalize_directlfq"] = False
        return search_config

    def test_sums_fragments_with_correlation_weights(
        self, fragment_sum_data, psm_df, sum_config
    ):
        """Given fragments of two precursors, when summed, then the uncorrelated fragment of precursor 10 does not count, precursor 20 without correlating fragments keeps its plain sum, and a run without fragments is 0."""
        # Given
        intensity_df, correlation_df = fragment_sum_data

        # When
        result_df = QuantBuilder(psm_df).sum_fragments_to_precursors(
            intensity_df, correlation_df, sum_config
        )

        # Then
        expected_df = pd.DataFrame(
            {
                "mod_seq_charge_hash": [10, 20],
                "mod_seq_hash": [1, 2],
                "pg": ["PG001", "PG002"],
                "run1": [110.0, 50.0],
                "run2": [220.0, 0.0],
                "run3": [440.0, 60.0],
                "run4": [880.0, 70.0],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_applies_power_to_correlation(self, fragment_sum_data, psm_df, sum_config):
        """Given a fragment with correlation 0.5, when summed, then its weight is 0.5 to the correlation power."""
        # Given
        intensity_df, correlation_df = fragment_sum_data
        correlation_df.loc[correlation_df["ion"] == 101, RUN_COLUMNS] = 0.5

        # When
        result_df = QuantBuilder(psm_df).sum_fragments_to_precursors(
            intensity_df, correlation_df, sum_config
        )

        # Then
        assert result_df["run1"].iloc[0] == pytest.approx(
            100.0 + 10.0 * 0.5**FRAGMENT_CORRELATION_POWER
        )

    def test_run_where_only_an_uncorrelated_fragment_was_seen_is_zero(
        self, fragment_sum_data, psm_df, sum_config
    ):
        """Given a run in which only the uncorrelated fragment of a precursor was observed, when summed, then that run reports 0 instead of a tiny value."""
        # Given - in run4 only ion 102 with correlation 0 is left for precursor 10
        intensity_df, correlation_df = fragment_sum_data
        intensity_df.loc[intensity_df["ion"].isin([100, 101]), "run4"] = 0.0
        intensity_df.loc[intensity_df["ion"] == 102, "run4"] = 8000.0

        # When
        result_df = QuantBuilder(psm_df).sum_fragments_to_precursors(
            intensity_df, correlation_df, sum_config
        )

        # Then
        assert result_df["run4"].iloc[0] == 0.0

    @pytest.mark.parametrize("normalize_directlfq", [True, False])
    def test_returns_empty_frame_when_nothing_was_observed(
        self, fragment_sum_data, psm_df, sum_config, normalize_directlfq
    ):
        """Given fragments with zero intensity everywhere, when summed, then an empty frame is returned instead of failing inside the normalization."""
        # Given
        intensity_df, correlation_df = fragment_sum_data
        intensity_df[RUN_COLUMNS] = 0.0
        sum_config["search_output"]["normalize_directlfq"] = normalize_directlfq

        # When
        result_df = QuantBuilder(psm_df).sum_fragments_to_precursors(
            intensity_df, correlation_df, sum_config
        )

        # Then
        assert result_df.empty

    @pytest.mark.parametrize(
        "normalize_directlfq, run1_factor", [(True, 2.0), (False, 1.0)]
    )
    def test_respects_normalization_flag(
        self, fragment_sum_data, psm_df, sum_config, normalize_directlfq, run1_factor
    ):
        """Given the normalization flag, when summed, then the sample shift is applied to the fragments only if enabled."""
        # Given
        intensity_df, correlation_df = fragment_sum_data
        sum_config["search_output"]["normalize_directlfq"] = normalize_directlfq

        def shift_run1_by_one_log2_unit(lfq_df, **kwargs):
            return MagicMock(
                complete_dataframe=lfq_df.assign(run1=lfq_df["run1"] + 1.0)
            )

        # When
        with patch(
            "alphadia.outputtransform.quantification.quant_builder.lfqnorm.NormalizationManagerSamplesOnSelectedProteins",
            side_effect=shift_run1_by_one_log2_unit,
        ):
            result_df = QuantBuilder(psm_df).sum_fragments_to_precursors(
                intensity_df, correlation_df, sum_config
            )

        # Then - the normalization shift doubles run1 only when enabled
        assert result_df["run1"].iloc[0] == pytest.approx(run1_factor * 110.0)
        assert result_df["run2"].iloc[0] == pytest.approx(220.0)

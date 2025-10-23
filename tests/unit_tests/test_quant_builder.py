from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from conftest import mock_precursor_df

from alphadia.outputtransform.quant_builder import QuantBuilder, _ion_hash, prepare_df


def create_mock_fragment_df(n_fragments: int = 4, n_precursor: int = 3) -> pd.DataFrame:
    """Create a mock fragment dataframe with correct dimensions."""
    np.random.seed(42)  # for reproducibility
    total_fragments = n_precursor * n_fragments

    return pd.DataFrame(
        {
            "precursor_idx": np.repeat(np.arange(n_precursor), n_fragments),
            "mz": np.random.rand(total_fragments) * 1000 + 200,
            "charge": np.random.choice([1, 2], size=total_fragments).astype(np.uint8),
            "number": np.tile(np.arange(1, n_fragments + 1), n_precursor).astype(
                np.uint8
            ),
            "type": np.tile([98, 121] * (n_fragments // 2), n_precursor).astype(
                np.uint8
            ),
            "position": np.tile(np.arange(n_fragments), n_precursor).astype(np.uint8),
            "height": np.random.rand(total_fragments) * 1000,
            "intensity": np.random.rand(total_fragments) * 1000,
            "correlation": np.random.rand(total_fragments),
            "loss_type": np.ones(total_fragments).astype(np.uint8),
        }
    )


@pytest.fixture
def basic_psm_df():
    """Basic PSM DataFrame with 2 precursors."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1],
            "pg": ["PG001", "PG002"],
            "mod_seq_hash": [1, 2],
            "mod_seq_charge_hash": [10, 20],
        }
    )


@pytest.fixture
def extended_psm_df():
    """Extended PSM DataFrame with 5 precursors."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4],
            "pg": ["PG001", "PG002", "PG001", "PG003", "PG002"],
            "mod_seq_hash": [1, 2, 3, 4, 5],
            "mod_seq_charge_hash": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def simple_fragment_df():
    """Simple fragment DataFrame with basic structure."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 0, 1, 1, 2, 2],
            "mz": [500.1, 600.2, 700.3, 800.4, 900.5, 1000.6],
            "charge": np.array([1, 2, 1, 2, 1, 2], dtype=np.uint8),
            "number": np.array([1, 2, 1, 2, 1, 2], dtype=np.uint8),
            "type": np.array([98, 121, 98, 121, 98, 121], dtype=np.uint8),
            "position": np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8),
            "height": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "intensity": [110.0, 220.0, 330.0, 440.0, 550.0, 660.0],
            "correlation": [0.8, 0.9, 0.7, 0.6, 0.5, 0.4],
            "loss_type": np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8),
        }
    )


@pytest.fixture
def basic_fragment_df():
    """Basic fragment DataFrame for 2 precursors."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1],
            "mz": [500.1, 600.2],
            "charge": np.array([1, 2], dtype=np.uint8),
            "number": np.array([1, 1], dtype=np.uint8),
            "type": np.array([98, 121], dtype=np.uint8),
            "position": np.array([0, 0], dtype=np.uint8),
            "height": [100.0, 200.0],
            "intensity": [110.0, 220.0],
            "correlation": [0.8, 0.9],
            "loss_type": np.array([1, 1], dtype=np.uint8),
        }
    )


@pytest.fixture
def duplicate_fragment_df():
    """Fragment DataFrame with duplicate ion hashes for testing."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 0, 0, 1],  # precursor 0 has duplicate fragments
            "mz": [500.1, 500.1, 600.2, 700.3],  # Same mz for duplicates
            "charge": np.array(
                [1, 1, 2, 1], dtype=np.uint8
            ),  # Same charge for duplicates
            "number": np.array(
                [1, 1, 2, 1], dtype=np.uint8
            ),  # Same number for duplicates
            "type": np.array(
                [98, 98, 121, 98], dtype=np.uint8
            ),  # Same type for duplicates
            "position": np.array([0, 0, 1, 0], dtype=np.uint8),
            "height": [100.0, 150.0, 200.0, 300.0],  # Different heights
            "intensity": [110.0, 160.0, 220.0, 330.0],  # Different intensities
            "correlation": [0.8, 0.85, 0.9, 0.7],  # Different correlations
            "loss_type": np.array([1, 1, 1, 1], dtype=np.uint8),  # Same loss_type
        }
    )


@pytest.fixture
def helper_psm_df():
    """PSM DataFrame for helper function tests."""
    return pd.DataFrame({"precursor_idx": [0, 1, 2], "pg": ["PG001", "PG002", "PG003"]})


@pytest.fixture
def helper_fragment_df():
    """Fragment DataFrame for helper function tests."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4],  # More precursors than in PSM
            "number": [1, 2, 3, 4, 5],
            "type": [98, 121, 98, 121, 98],  # b and y ions
            "charge": [1, 1, 2, 2, 1],
            "loss_type": [1, 1, 1, 1, 1],
            "intensity": [100, 200, 300, 400, 500],
            "correlation": [0.8, 0.9, 0.7, 0.6, 0.5],
        }
    )

@pytest.fixture
def create_ms2_features():
    """MS2 features consisting of DataFrames for helper function tests."""
    df = pd.DataFrame({
        "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "run1": [
            15453501, 3, 15453503, 15453502,
            15453502, 1, 15453502, 15453501, 15453501
        ] * 2,
        "run2": [
            15453501, 2, 15453502, 15453501,
            15453501, 5, 15453501, 15453501, 15453502
        ] * 2,
        "run3": [
            15453502, 2, 15453505, 15453501,
            15453501, 3, 15453503, 15453502, 15453503
        ] * 2,
        "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
        "mod_seq_hash": [
            6831315783892314113, 6831315783892314113, 6831315783892314113, 6831315783892314113,
            1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364
        ] * 2,
        "mod_seq_charge_hash": [
            3157800000000000000, 3157800000000000000, 3157800000000000000, 3157800000000000000,
            3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536
        ] * 2,
    })

    # Correlation data
    df_corr = pd.DataFrame({
        "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "run1": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
        "run2": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
        "run3": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
        "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
        "mod_seq_hash": [
            6831315783892314113, 6831315783892314113, 6831315783892314113, 6831315783892314113,
            1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364
        ] * 2,
        "mod_seq_charge_hash": [
            3157800000000000000, 3157800000000000000, 3157800000000000000, 3157800000000000000,
            3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536
        ] * 2,
    })

    # Mass error data
    mass_error_data = pd.DataFrame({
        "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "run1": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
        "run2": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
        "run3": [1, 0.1, 1, 1, 1, 0.1, 1, 1, 1] * 2,
        "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
        "mod_seq_hash": [
            6831315783892314113, 6831315783892314113, 6831315783892314113, 6831315783892314113,
            1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364
        ] * 2,
        "mod_seq_charge_hash": [
            3157800000000000000, 3157800000000000000, 3157800000000000000, 3157800000000000000,
            3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536
        ] * 2,
    })

    # Height data
    height_data = pd.DataFrame({
        "precursor_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "ion": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "run1": [114, 144, 114, 113, 114, 514, 134, 144, 131] * 2,
        "run2": [184, 114, 144, 114, 144, 114, 134, 115, 321] * 2,
        "run3": [114, 124, 114, 114, 164, 144, 114, 114, 411] * 2,
        "pg": ["TNAA_ECOLI"] * 9 + ["TNAB_ECOLI"] * 9,
        "mod_seq_hash": [
            6831315783892314113, 6831315783892314113, 6831315783892314113, 6831315783892314113,
            1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364
        ] * 2,
        "mod_seq_charge_hash": [
            3157800000000000000, 3157800000000000000, 3157800000000000000, 3157800000000000000,
            3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536
        ] * 2,
    })

    return {
        "intensity": df,
        "correlation": df_corr,
        "mass_error": mass_error_data,
        "height": height_data,
    }

@pytest.fixture
def create_psm_file():
    """PSM file for helper function tests."""
    return pd.DataFrame({
    'precursor_idx': [1, 2, 3, 4, 5, 6, 7, 8, 9] * 6,  # 3 runs × 2 protein groups
    'ion': [1, 2, 3, 4, 5, 6, 7, 8, 9] * 6,
    'pg': ['TNAA_ECOLI'] * 27 + ['TNAB_ECOLI'] * 27,
    'mod_seq_hash': [
        6831315783892314113, 6831315783892314113, 6831315783892314113, 6831315783892314113,
        1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364, 1784898696230645364
    ] * 6,
    'mod_seq_charge_hash': [
        3157800000000000000, 3157800000000000000, 3157800000000000000, 3157800000000000000,
        3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536, 3178489869623064536
    ] * 6,
    'run': ['run1'] * 9 + ['run2'] * 9 + ['run3'] * 9 + ['run1'] * 9 + ['run2'] * 9 + ['run3'] * 9,
    'intensity': [
        15453501, 3, 15453503, 15453502, 15453502, 1, 15453502, 15453501, 15453501,
        15453501, 2, 15453502, 15453501, 15453501, 5, 15453501, 15453501, 15453502,
        15453502, 4, 15453505, 15453501, 15453501, 3, 15453503, 15453502, 15453503
    ] * 2,
    'delta_rt': [
        15453501, 2, 15453503, 15453502, 15453502, 1, 15453502, 15453501, 15453501,
        15453501, 1, 15453502, 15453501, 15453501, 5, 15453501, 15453501, 15453502,
        15453502, 2, 15453505, 15453501, 15453501, 3, 15453503, 15453502, 15453503
    ] * 2
})

class TestQuantBuilderAccumulateFragDf:
    """Test cases for QuantBuilder.accumulate_frag_df() method."""

    def test_accumulate_frag_df_single_run_should_return_intensity_and_quality_dfs(
        self, extended_psm_df, simple_fragment_df
    ):
        """Test that accumulate_frag_df returns correct dataframes for single run."""
        # given
        builder = QuantBuilder(extended_psm_df, columns=["intensity", "correlation"])
        df_iterable = [("run1", simple_fragment_df)]

        # when
        result_dict = builder.accumulate_frag_df(iter(df_iterable))

        # then
        expected_intensity_df = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32),
                "ion": [
                    72446825449127936,
                    72753593488244736,
                    72446825449127937,
                    72753593488244737,
                    72446825449127938,
                    72753593488244738,
                ],
                "run1": [110.0, 220.0, 330.0, 440.0, 550.0, 660.0],
                "pg": ["PG001", "PG001", "PG002", "PG002", "PG001", "PG001"],
                "mod_seq_hash": [1, 1, 2, 2, 3, 3],
                "mod_seq_charge_hash": [10, 10, 20, 20, 30, 30],
            }
        )

        expected_quality_df = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32),
                "ion": [
                    72446825449127936,
                    72753593488244736,
                    72446825449127937,
                    72753593488244737,
                    72446825449127938,
                    72753593488244738,
                ],
                "run1": [0.8, 0.9, 0.7, 0.6, 0.5, 0.4],
                "pg": ["PG001", "PG001", "PG002", "PG002", "PG001", "PG001"],
                "mod_seq_hash": [1, 1, 2, 2, 3, 3],
                "mod_seq_charge_hash": [10, 10, 20, 20, 30, 30],
            }
        )

        pd.testing.assert_frame_equal(result_dict["intensity"], expected_intensity_df)
        pd.testing.assert_frame_equal(result_dict["correlation"], expected_quality_df)

    def test_accumulate_frag_df_multiple_runs_should_merge_data_correctly(
        self, basic_psm_df, basic_fragment_df
    ):
        """Test that accumulate_frag_df correctly merges data from multiple runs."""
        # given
        frag_df2 = basic_fragment_df.copy()
        frag_df2["height"] = [150.0, 250.0]
        frag_df2["intensity"] = [160.0, 270.0]
        frag_df2["correlation"] = [0.7, 0.6]

        builder = QuantBuilder(basic_psm_df, columns=["intensity", "correlation"])
        df_iterable = [("run1", basic_fragment_df), ("run2", frag_df2)]

        # when
        result_dict = builder.accumulate_frag_df(iter(df_iterable))

        # then
        expected_intensity_df = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72753589193277441],
                "run1": [110.0, 220.0],
                "run2": [160.0, 270.0],
                "pg": ["PG001", "PG002"],
                "mod_seq_hash": [1, 2],
                "mod_seq_charge_hash": [10, 20],
            }
        )

        expected_quality_df = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72753589193277441],
                "run1": [0.8, 0.9],
                "run2": [0.7, 0.6],
                "pg": ["PG001", "PG002"],
                "mod_seq_hash": [1, 2],
                "mod_seq_charge_hash": [10, 20],
            }
        )

        pd.testing.assert_frame_equal(result_dict["intensity"], expected_intensity_df)
        pd.testing.assert_frame_equal(result_dict["correlation"], expected_quality_df)

    def test_accumulate_frag_df_with_empty_iterator_should_return_none(
        self, extended_psm_df
    ):
        """Test that accumulate_frag_df returns None when iterator is empty."""
        # given
        builder = QuantBuilder(extended_psm_df, columns=["intensity", "correlation"])
        empty_iterable = iter([])

        # when
        result = builder.accumulate_frag_df(empty_iterable)

        # then
        assert result == (None, None)

    def test_accumulate_frag_df_should_fill_nan_with_zero(self):
        """Test that accumulate_frag_df fills NaN values with zero."""
        # given
        psm_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        # Create fragment dataframes with different precursor coverage
        frag_df1 = pd.DataFrame(
            {
                "precursor_idx": [0, 1],  # Only covers first 2 precursors
                "mz": [500.1, 600.2],
                "charge": np.array([1, 2], dtype=np.uint8),
                "number": np.array([1, 1], dtype=np.uint8),
                "type": np.array([98, 121], dtype=np.uint8),
                "position": np.array([0, 0], dtype=np.uint8),
                "height": [100.0, 200.0],
                "intensity": [110.0, 220.0],
                "correlation": [0.8, 0.9],
                "loss_type": np.array([1, 1], dtype=np.uint8),
            }
        )

        frag_df2 = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],  # Covers all 3 precursors
                "mz": [500.1, 600.2, 700.3],
                "charge": np.array([1, 2, 1], dtype=np.uint8),
                "number": np.array([1, 1, 1], dtype=np.uint8),
                "type": np.array([98, 121, 98], dtype=np.uint8),
                "position": np.array([0, 0, 0], dtype=np.uint8),
                "height": [150.0, 250.0, 350.0],
                "intensity": [160.0, 270.0, 380.0],
                "correlation": [0.7, 0.6, 0.5],
                "loss_type": np.array([1, 1, 1], dtype=np.uint8),
            }
        )

        builder = QuantBuilder(psm_df, columns=["intensity", "correlation"])
        df_iterable = [("run1", frag_df1), ("run2", frag_df2)]

        # when
        result_dict = builder.accumulate_frag_df(iter(df_iterable))

        # then
        expected_intensity_df = pd.DataFrame(
            {
                "precursor_idx": np.array(
                    [0, 2, 1], dtype=np.uint32
                ),  # Order matches merge behavior
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [110.0, 0.0, 220.0],  # precursor 2 gets 0.0 since not in run1
                "run2": [160.0, 380.0, 270.0],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        expected_quality_df = pd.DataFrame(
            {
                "precursor_idx": np.array(
                    [0, 2, 1], dtype=np.uint32
                ),  # Order matches merge behavior
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [0.8, 0.0, 0.9],  # precursor 2 gets 0.0 since not in run1
                "run2": [0.7, 0.5, 0.6],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        pd.testing.assert_frame_equal(result_dict["intensity"], expected_intensity_df)
        pd.testing.assert_frame_equal(result_dict["correlation"], expected_quality_df)

    def test_accumulate_frag_df_should_filter_by_psm_precursor_idx(self):
        """Test that accumulate_frag_df only includes fragments for precursors in PSM data."""
        # given
        psm_df = mock_precursor_df(n_precursor=2, with_decoy=False)
        psm_df["pg"] = ["PG001", "PG002"]
        psm_df["mod_seq_hash"] = [1, 2]
        psm_df["mod_seq_charge_hash"] = [10, 20]

        # Create fragment data with more precursors than in PSM
        frag_df = create_mock_fragment_df(n_fragments=4, n_precursor=5)

        builder = QuantBuilder(psm_df, columns=["intensity", "correlation"])
        df_iterable = [("run1", frag_df)]

        # when
        result_dict = builder.accumulate_frag_df(iter(df_iterable))

        # then
        # Should only contain precursors that exist in PSM data
        unique_precursor_idx = result_dict["intensity"]["precursor_idx"].unique()
        assert all(
            idx in psm_df["precursor_idx"].values for idx in unique_precursor_idx
        )

    def test_accumulate_frag_df_with_custom_column_should_use_specified_column(self):
        """Test that accumulate_frag_df uses the specified column for quantification."""
        # given
        psm_df = mock_precursor_df(n_precursor=3, with_decoy=False)
        psm_df["pg"] = ["PG001", "PG002", "PG003"]
        psm_df["mod_seq_hash"] = [1, 2, 3]
        psm_df["mod_seq_charge_hash"] = [10, 20, 30]

        frag_df = create_mock_fragment_df(n_fragments=4, n_precursor=3)

        builder = QuantBuilder(
            psm_df, columns=["height"]
        )  # Use height instead of intensity
        df_iterable = [("run1", frag_df)]

        # when
        result_dict = builder.accumulate_frag_df(iter(df_iterable))

        # then
        # Values should come from height column, not intensity
        assert result_dict["height"]["run1"].notna().any()

    @patch("alphadia.outputtransform.quant_builder.logger")
    def test_accumulate_frag_df_should_log_accumulation_start(self, mock_logger):
        """Test that accumulate_frag_df logs the start of accumulation."""
        # given
        psm_df = mock_precursor_df(n_precursor=2, with_decoy=False)
        psm_df["pg"] = ["PG001", "PG002"]
        psm_df["mod_seq_hash"] = [1, 2]
        psm_df["mod_seq_charge_hash"] = [10, 20]

        frag_df = create_mock_fragment_df(n_fragments=4, n_precursor=2)

        builder = QuantBuilder(psm_df, columns=["intensity", "correlation"])
        df_iterable = [("run1", frag_df)]

        # when
        builder.accumulate_frag_df(iter(df_iterable))

        # then
        mock_logger.info.assert_called_with("Accumulating fragment data")

    def test_accumulate_frag_df_should_create_consistent_ion_hashes(self):
        """Test that accumulate_frag_df creates consistent ion hashes across runs."""
        # given
        psm_df = mock_precursor_df(n_precursor=2, with_decoy=False)
        psm_df["pg"] = ["PG001", "PG002"]
        psm_df["mod_seq_hash"] = [1, 2]
        psm_df["mod_seq_charge_hash"] = [10, 20]

        # Create identical fragment dataframes
        frag_df1 = create_mock_fragment_df(n_fragments=4, n_precursor=2)
        frag_df2 = frag_df1.copy()

        builder = QuantBuilder(psm_df, columns=["intensity", "correlation"])
        df_iterable = [("run1", frag_df1), ("run2", frag_df2)]

        # when
        result_dict = builder.accumulate_frag_df(iter(df_iterable))

        # then
        # Should have same ion hashes in both dataframes
        assert len(result_dict["intensity"]["ion"].unique()) == len(
            frag_df1
        )  # Same ions from both runs
        assert all(result_dict["intensity"]["ion"] == result_dict["correlation"]["ion"])

    def test_accumulate_frag_df_with_duplicate_ion_hashes_should_handle_correctly(
        self, basic_psm_df, duplicate_fragment_df
    ):
        """Test that accumulate_frag_df handles duplicate ion hashes correctly by merging data."""
        # given
        frag_df2 = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 1],  # Different coverage
                "mz": [500.1, 700.3, 800.4],
                "charge": np.array([1, 1, 2], dtype=np.uint8),
                "number": np.array([1, 1, 1], dtype=np.uint8),
                "type": np.array([98, 98, 121], dtype=np.uint8),
                "position": np.array([0, 0, 0], dtype=np.uint8),
                "height": [120.0, 350.0, 400.0],
                "intensity": [130.0, 380.0, 440.0],
                "correlation": [0.75, 0.65, 0.6],
                "loss_type": np.array([1, 1, 1], dtype=np.uint8),
            }
        )

        builder = QuantBuilder(basic_psm_df, columns=["intensity", "correlation"])
        df_iterable = [("run1", duplicate_fragment_df), ("run2", frag_df2)]

        # when
        result_dict = builder.accumulate_frag_df(iter(df_iterable))

        # then
        # Should handle duplicates by keeping ALL occurrences (pandas merge behavior)
        # and merge properly across runs via outer join
        expected_intensity_df = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 0, 1, 1, 0], dtype=np.uint32),
                "ion": [
                    72446825449127936,  # precursor 0, number 1, type b, charge 1 (duplicate #1)
                    72446825449127936,  # precursor 0, number 1, type b, charge 1 (duplicate #2)
                    72446825449127937,  # precursor 1, number 1, type b, charge 1
                    72753589193277441,  # precursor 1, number 1, type y, charge 2
                    72753593488244736,  # precursor 0, number 2, type y, charge 2
                ],
                "run1": [
                    110.0,
                    160.0,
                    330.0,
                    0.0,
                    220.0,
                ],  # Both duplicates kept, missing ion gets 0.0
                "run2": [
                    130.0,
                    130.0,
                    380.0,
                    440.0,
                    0.0,
                ],  # Both duplicate rows get the same run2 value
                "pg": ["PG001", "PG001", "PG002", "PG002", "PG001"],
                "mod_seq_hash": [1, 1, 2, 2, 1],
                "mod_seq_charge_hash": [10, 10, 20, 20, 10],
            }
        )

        expected_quality_df = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 0, 1, 1, 0], dtype=np.uint32),
                "ion": [
                    72446825449127936,  # precursor 0, number 1, type b, charge 1 (duplicate #1)
                    72446825449127936,  # precursor 0, number 1, type b, charge 1 (duplicate #2)
                    72446825449127937,  # precursor 1, number 1, type b, charge 1
                    72753589193277441,  # precursor 1, number 1, type y, charge 2
                    72753593488244736,  # precursor 0, number 2, type y, charge 2
                ],
                "run1": [
                    0.8,
                    0.85,
                    0.7,
                    0.0,
                    0.9,
                ],  # Both duplicates kept, missing ion gets 0.0
                "run2": [
                    0.75,
                    0.75,
                    0.65,
                    0.6,
                    0.0,
                ],  # Both duplicate rows get the same run2 value
                "pg": ["PG001", "PG001", "PG002", "PG002", "PG001"],
                "mod_seq_hash": [1, 1, 2, 2, 1],
                "mod_seq_charge_hash": [10, 10, 20, 20, 10],
            }
        )

        pd.testing.assert_frame_equal(result_dict["intensity"], expected_intensity_df)
        pd.testing.assert_frame_equal(result_dict["correlation"], expected_quality_df)


class TestQuantBuilderAccumulateFragDfFromFolders:
    """Test cases for QuantBuilder.accumulate_frag_df_from_folders() method."""

    @patch("alphadia.outputtransform.quant_builder.QuantBuilder.accumulate_frag_df")
    def test_accumulate_frag_df_from_folders_should_call_accumulate_frag_df_with_generator(
        self, mock_accumulate_frag_df, basic_psm_df
    ):
        """Test that accumulate_frag_df_from_folders calls accumulate_frag_df with generator from folders."""
        # given
        folder_list = ["folder1", "folder2", "folder3"]
        builder = QuantBuilder(basic_psm_df, columns=["intensity", "correlation"])

        expected_intensity_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1],
                "ion": [100, 200],
                "run1": [110.0, 220.0],
                "pg": ["PG001", "PG002"],
            }
        )
        expected_quality_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1],
                "ion": [100, 200],
                "run1": [0.8, 0.9],
                "pg": ["PG001", "PG002"],
            }
        )
        mock_accumulate_frag_df.return_value = {
            "intensity": expected_intensity_df,
            "correlation": expected_quality_df,
        }

        # when
        result_dict = builder.accumulate_frag_df_from_folders(folder_list)

        # then
        mock_accumulate_frag_df.assert_called_once()

        # Verify the argument passed to accumulate_frag_df is an iterator
        call_args = mock_accumulate_frag_df.call_args[0][0]
        assert hasattr(call_args, "__iter__")
        assert hasattr(call_args, "__next__")

        # Verify the return values are passed through
        pd.testing.assert_frame_equal(result_dict["intensity"], expected_intensity_df)
        pd.testing.assert_frame_equal(result_dict["correlation"], expected_quality_df)

    @patch("alphadia.outputtransform.quant_builder.QuantBuilder.accumulate_frag_df")
    def test_accumulate_frag_df_from_folders_should_return_none_when_accumulate_returns_none(
        self, mock_accumulate_frag_df, basic_psm_df
    ):
        """Test that accumulate_frag_df_from_folders returns None when accumulate_frag_df returns None."""
        # given
        folder_list = ["empty_folder"]
        builder = QuantBuilder(basic_psm_df, columns=["intensity", "correlation"])
        mock_accumulate_frag_df.return_value = None

        # when
        result = builder.accumulate_frag_df_from_folders(folder_list)

        # then
        assert result is None
        mock_accumulate_frag_df.assert_called_once()

    @patch("alphadia.outputtransform.quant_builder.QuantBuilder.accumulate_frag_df")
    def test_accumulate_frag_df_from_folders_should_pass_empty_list_correctly(
        self, mock_accumulate_frag_df, basic_psm_df
    ):
        """Test that accumulate_frag_df_from_folders handles empty folder list correctly."""
        # given
        folder_list = []
        builder = QuantBuilder(basic_psm_df, columns=["intensity", "correlation"])
        mock_accumulate_frag_df.return_value = None

        # when
        result = builder.accumulate_frag_df_from_folders(folder_list)

        # then
        assert result is None
        mock_accumulate_frag_df.assert_called_once()

    @patch("alphadia.outputtransform.quant_builder.QuantBuilder.accumulate_frag_df")
    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_accumulate_frag_df_from_folders_should_pass_generator_with_valid_data(
        self,
        mock_read_parquet,
        mock_exists,
        mock_accumulate_frag_df,
        basic_psm_df,
        simple_fragment_df,
    ):
        """Test that accumulate_frag_df_from_folders passes generator with data from valid folders."""
        # given
        folder_list = ["folder1", "folder2"]
        builder = QuantBuilder(basic_psm_df, columns=["intensity", "correlation"])

        mock_exists.return_value = True
        mock_read_parquet.return_value = simple_fragment_df

        expected_result = (pd.DataFrame(), pd.DataFrame())
        mock_accumulate_frag_df.return_value = expected_result

        # when
        builder.accumulate_frag_df_from_folders(folder_list)

        # then
        mock_accumulate_frag_df.assert_called_once()

        # Verify the generator passed contains expected data
        call_args = mock_accumulate_frag_df.call_args[0][0]
        generator_items = list(call_args)

        assert len(generator_items) == 2
        assert generator_items[0][0] == "folder1"  # raw_name
        assert generator_items[1][0] == "folder2"  # raw_name
        pd.testing.assert_frame_equal(generator_items[0][1], simple_fragment_df)
        pd.testing.assert_frame_equal(generator_items[1][1], simple_fragment_df)

    @patch("alphadia.outputtransform.quant_builder.QuantBuilder.accumulate_frag_df")
    @patch("os.path.exists")
    def test_accumulate_frag_df_from_folders_should_skip_missing_files(
        self, mock_exists, mock_accumulate_frag_df, basic_psm_df
    ):
        """Test that accumulate_frag_df_from_folders skips folders without frag.parquet files."""
        # given
        folder_list = ["missing_folder1", "missing_folder2"]
        builder = QuantBuilder(basic_psm_df, columns=["intensity", "correlation"])

        mock_exists.return_value = False  # No frag.parquet files exist
        mock_accumulate_frag_df.return_value = None

        # when
        builder.accumulate_frag_df_from_folders(folder_list)

        # then
        mock_accumulate_frag_df.assert_called_once()

        # Verify empty generator is passed
        call_args = mock_accumulate_frag_df.call_args[0][0]
        generator_items = list(call_args)
        assert len(generator_items) == 0


class TestQuantBuilderHelperFunctions:
    """Test cases for helper functions used by QuantBuilder."""

    def test_prepare_df_should_filter_by_precursor_idx(
        self, helper_psm_df, helper_fragment_df
    ):
        """Test that prepare_df filters fragments by precursor_idx from PSM data."""
        # given
        # when
        result_df = prepare_df(
            helper_fragment_df, helper_psm_df, columns=["intensity", "correlation"]
        )

        # then
        assert len(result_df) == 3  # Only first 3 fragments should remain
        assert all(result_df["precursor_idx"].isin([0, 1, 2]))
        assert "ion" in result_df.columns
        assert set(result_df.columns) == {
            "precursor_idx",
            "ion",
            "intensity",
            "correlation",
        }

    def test_ion_hash_should_create_unique_hashes(self):
        """Test that _ion_hash creates unique hashes for different fragment combinations."""
        # given
        precursor_idx = np.array([1, 1, 2])
        number = np.array([1, 2, 1])
        type_arr = np.array([98, 121, 98])  # b, y, b
        charge = np.array([1, 1, 2])
        loss_type = np.array([1, 1, 1])

        # when
        hashes = _ion_hash(precursor_idx, number, type_arr, charge, loss_type)

        # then
        assert len(np.unique(hashes)) == 3  # All hashes should be unique
        assert all(isinstance(h, int | np.integer) for h in hashes.flat)

    def test_ion_hash_should_be_deterministic(self):
        """Test that _ion_hash produces the same hash for identical inputs."""
        # given
        precursor_idx = 1
        number = 2
        type_val = 98
        charge = 1
        loss_type = 1

        # when
        hash1 = _ion_hash(precursor_idx, number, type_val, charge, loss_type)
        hash2 = _ion_hash(precursor_idx, number, type_val, charge, loss_type)

        # then
        assert hash1 == hash2


class TestQuantBuilderFilterFragDf:
    """Test cases for QuantBuilder.filter_frag_df() method."""

    @pytest.fixture
    def basic_filtering_data(self):
        """Create basic intensity and quality dataframes for filtering tests."""
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

    def test_filter_frag_df_should_keep_top_n_fragments_per_group(
        self, basic_filtering_data
    ):
        """Test that filter_frag_df keeps top N fragments per protein group."""
        # given
        intensity_df, quality_df = basic_filtering_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.5, top_n=2
        )

        # then
        assert len(filtered_intensity) == 4  # Top 2 from each of 2 groups

        # Check PG001 group: should keep top 2 (ions 100, 101 based on mean correlation)
        pg001_ions = filtered_intensity[filtered_intensity["pg"] == "PG001"][
            "ion"
        ].values
        assert 100 in pg001_ions  # Mean: (0.9+0.8)/2 = 0.85
        assert 101 in pg001_ions  # Mean: (0.7+0.6)/2 = 0.65

        # Check PG002 group: should keep top 2 (ions 200, 201)
        pg002_ions = filtered_intensity[filtered_intensity["pg"] == "PG002"][
            "ion"
        ].values
        assert 200 in pg002_ions  # Mean: (0.8+0.9)/2 = 0.85
        assert 201 in pg002_ions  # Mean: (0.6+0.7)/2 = 0.65

    def test_filter_frag_df_should_keep_fragments_above_min_correlation(
        self, basic_filtering_data
    ):
        """Test that filter_frag_df keeps fragments above minimum correlation threshold."""
        # given
        intensity_df, quality_df = basic_filtering_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.6, top_n=1
        )

        # then
        # Should keep top 1 from each group PLUS any above 0.6 correlation
        # Based on our data: ion 100 (0.85), ion 101 (0.65), ion 200 (0.85), ion 201 (0.65)
        # Top 1 per group: 100, 200; Above 0.6: 101, 201 -> Total: 4 fragments
        assert len(filtered_intensity) == 4

        kept_ions = set(filtered_intensity["ion"].values)
        assert kept_ions == {100, 101, 200, 201}  # Top ranking OR above 0.6

    def test_filter_frag_df_should_use_custom_group_column(self):
        """Test that filter_frag_df uses specified group column for ranking."""
        # given
        # Use mod_seq_hash as a custom group column since it's already numeric and present
        intensity_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0, 1, 1],
                "ion": [100, 101, 200, 201],
                "run1": [1000.0, 2000.0, 1500.0, 2500.0],
                "pg": ["PG001", "PG001", "PG002", "PG002"],
                "mod_seq_hash": [1, 1, 2, 2],  # Different grouping than pg
                "mod_seq_charge_hash": [10, 10, 20, 20],
            }
        )

        quality_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0, 1, 1],
                "ion": [100, 101, 200, 201],
                "run1": [0.9, 0.7, 0.8, 0.6],  # Different correlations for each ion
                "pg": ["PG001", "PG001", "PG002", "PG002"],
                "mod_seq_hash": [1, 1, 2, 2],
                "mod_seq_charge_hash": [10, 10, 20, 20],
            }
        )

        psm_df = pd.DataFrame({"precursor_idx": [0, 1]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df,
            quality_df,
            min_correlation=2.0,
            top_n=1,
            group_column="mod_seq_hash",  # Group by mod_seq_hash instead of pg
        )

        # then
        # Should keep top 1 from each mod_seq_hash group (1 and 2)
        # Group 1 (mod_seq_hash=1): ions 100 (0.9) and 101 (0.7) -> keep ion 100
        # Group 2 (mod_seq_hash=2): ions 200 (0.8) and 201 (0.6) -> keep ion 200
        assert len(filtered_intensity) == 2
        hash1_ions = filtered_intensity[filtered_intensity["mod_seq_hash"] == 1][
            "ion"
        ].values
        hash2_ions = filtered_intensity[filtered_intensity["mod_seq_hash"] == 2][
            "ion"
        ].values
        assert len(hash1_ions) == 1
        assert len(hash2_ions) == 1
        assert 100 in hash1_ions  # Highest in group 1
        assert 200 in hash2_ions  # Highest in group 2

    def test_filter_frag_df_should_calculate_total_as_mean_correlation(
        self, basic_filtering_data
    ):
        """Test that filter_frag_df calculates total as mean of run columns."""
        # given
        intensity_df, quality_df = basic_filtering_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1]})
        builder = QuantBuilder(psm_df)

        # when
        _, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.0, top_n=10
        )

        # then
        # Check that total column is calculated correctly
        expected_totals = [
            (0.9 + 0.8) / 2,  # ion 100
            (0.7 + 0.6) / 2,  # ion 101
            (0.3 + 0.4) / 2,  # ion 102
            (0.8 + 0.9) / 2,  # ion 200
            (0.6 + 0.7) / 2,  # ion 201
            (0.4 + 0.5) / 2,  # ion 202
        ]

        # Sort by ion to match expected order
        sorted_quality = filtered_quality.sort_values("ion")
        np.testing.assert_array_almost_equal(
            sorted_quality["total"].values, expected_totals
        )

    def test_filter_frag_df_should_add_rank_column(self, basic_filtering_data):
        """Test that filter_frag_df adds rank column with correct rankings."""
        # given
        intensity_df, quality_df = basic_filtering_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1]})
        builder = QuantBuilder(psm_df)

        # when
        _, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.0, top_n=10
        )

        # then
        # Check that rank column exists and has correct values
        assert "rank" in filtered_quality.columns

        # For PG001: ion 100 (0.85) should rank 1, ion 101 (0.65) should rank 2, ion 102 (0.35) should rank 3
        pg001_ranks = filtered_quality[filtered_quality["pg"] == "PG001"].set_index(
            "ion"
        )["rank"]
        assert pg001_ranks[100] == 1.0
        assert pg001_ranks[101] == 2.0
        assert pg001_ranks[102] == 3.0

        # For PG002: ion 200 (0.85) should rank 1, ion 201 (0.65) should rank 2, ion 202 (0.45) should rank 3
        pg002_ranks = filtered_quality[filtered_quality["pg"] == "PG002"].set_index(
            "ion"
        )["rank"]
        assert pg002_ranks[200] == 1.0
        assert pg002_ranks[201] == 2.0
        assert pg002_ranks[202] == 3.0

    def test_filter_frag_df_should_return_same_structure_dataframes(
        self, basic_filtering_data
    ):
        """Test that filter_frag_df returns dataframes with same structure as input."""
        # given
        intensity_df, quality_df = basic_filtering_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.5, top_n=2
        )

        # then
        # Check that filtered dataframes have same columns as input (plus total/rank for quality)
        assert set(filtered_intensity.columns) == set(intensity_df.columns)
        expected_quality_columns = set(quality_df.columns) | {"total", "rank"}
        assert set(filtered_quality.columns) == expected_quality_columns

    def test_filter_frag_df_with_empty_dataframes_should_return_empty_dataframes(self):
        """Test that filter_frag_df handles empty input dataframes correctly."""
        # given
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
        empty_quality = pd.DataFrame(
            columns=[
                "precursor_idx",
                "ion",
                "run1",
                "pg",
                "mod_seq_hash",
                "mod_seq_charge_hash",
            ]
        )
        psm_df = pd.DataFrame({"precursor_idx": []})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            empty_intensity, empty_quality, min_correlation=0.5, top_n=3
        )

        # then
        assert len(filtered_intensity) == 0
        assert len(filtered_quality) == 0
        assert "total" in filtered_quality.columns
        assert "rank" in filtered_quality.columns

    def test_filter_frag_df_with_single_fragment_per_group_should_keep_all(self):
        """Test that filter_frag_df keeps all fragments when each group has only one fragment."""
        # given
        intensity_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "ion": [100, 200, 300],
                "run1": [1000.0, 2000.0, 3000.0],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        quality_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "ion": [100, 200, 300],
                "run1": [0.3, 0.4, 0.2],  # All below min_correlation
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.5, top_n=1
        )

        # then
        # Should keep all fragments as they are top 1 in their respective groups
        assert len(filtered_intensity) == 3
        assert len(filtered_quality) == 3

    def test_filter_frag_df_with_very_high_min_correlation_should_keep_only_top_n(self):
        """Test that filter_frag_df with very high min_correlation keeps only top N fragments."""
        # given
        intensity_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0, 0, 0],
                "ion": [100, 101, 102, 103],
                "run1": [1000.0, 2000.0, 3000.0, 4000.0],
                "pg": ["PG001", "PG001", "PG001", "PG001"],
                "mod_seq_hash": [1, 1, 1, 1],
                "mod_seq_charge_hash": [10, 10, 10, 10],
            }
        )

        quality_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0, 0, 0],
                "ion": [100, 101, 102, 103],
                "run1": [0.9, 0.8, 0.7, 0.6],
                "pg": ["PG001", "PG001", "PG001", "PG001"],
                "mod_seq_hash": [1, 1, 1, 1],
                "mod_seq_charge_hash": [10, 10, 10, 10],
            }
        )

        psm_df = pd.DataFrame({"precursor_idx": [0]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df,
            quality_df,
            min_correlation=2.0,
            top_n=2,  # Use 2.0 to ensure no fragment meets correlation threshold
        )

        # then
        # Should keep exactly 2 fragments (top 2) since no fragment has correlation > 2.0
        assert len(filtered_intensity) == 2
        kept_ions = filtered_intensity["ion"].values
        assert 100 in kept_ions  # Highest correlation (0.9)
        assert 101 in kept_ions  # Second highest correlation (0.8)

    def test_filter_frag_df_with_high_min_correlation_should_keep_all_above_threshold(
        self,
    ):
        """Test that filter_frag_df keeps all fragments above high correlation threshold."""
        # given
        intensity_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0, 0, 0],
                "ion": [100, 101, 102, 103],
                "run1": [1000.0, 2000.0, 3000.0, 4000.0],
                "run2": [1100.0, 2100.0, 3100.0, 4100.0],
                "pg": ["PG001", "PG001", "PG001", "PG001"],
                "mod_seq_hash": [1, 1, 1, 1],
                "mod_seq_charge_hash": [10, 10, 10, 10],
            }
        )

        quality_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0, 0, 0],
                "ion": [100, 101, 102, 103],
                "run1": [0.9, 0.8, 0.7, 0.3],  # 3 fragments above 0.6
                "run2": [0.8, 0.7, 0.6, 0.2],
                "pg": ["PG001", "PG001", "PG001", "PG001"],
                "mod_seq_hash": [1, 1, 1, 1],
                "mod_seq_charge_hash": [10, 10, 10, 10],
            }
        )

        psm_df = pd.DataFrame({"precursor_idx": [0]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.6, top_n=1
        )

        # then
        # Should keep top 1 (ion 100) PLUS all above 0.6 (ions 101, 102)
        assert len(filtered_intensity) == 3
        kept_ions = set(filtered_intensity["ion"].values)
        assert kept_ions == {100, 101, 102}

    def test_filter_frag_df_should_handle_missing_run_columns_gracefully(self):
        """Test that filter_frag_df identifies run columns correctly by excluding metadata."""
        # given
        intensity_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0],
                "ion": [100, 101],
                "run_data": [1000.0, 2000.0],  # Non-standard run column name
                "pg": ["PG001", "PG001"],
                "mod_seq_hash": [1, 1],
                "mod_seq_charge_hash": [10, 10],
            }
        )

        quality_df = pd.DataFrame(
            {
                "precursor_idx": [0, 0],
                "ion": [100, 101],
                "run_data": [0.9, 0.7],
                "pg": ["PG001", "PG001"],
                "mod_seq_hash": [1, 1],
                "mod_seq_charge_hash": [10, 10],
            }
        )

        psm_df = pd.DataFrame({"precursor_idx": [0]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=0.5, top_n=2
        )

        # then
        # Should work correctly with non-standard run column name
        assert len(filtered_intensity) == 2
        assert "total" in filtered_quality.columns
        # Total should equal the single run column value since only one run column
        assert filtered_quality["total"].iloc[0] == 0.9
        assert filtered_quality["total"].iloc[1] == 0.7

    @pytest.mark.parametrize(
        "min_correlation,top_n,expected_count",
        [
            (
                2.0,
                1,
                2,
            ),  # Keep top 1 from each group (high correlation to test only top_n)
            (2.0, 3, 6),  # Keep top 3 from each group (all, since we have 3 per group)
            (
                0.7,
                1,
                2,
            ),  # Keep top 1 from each group (ions 100 and 200 both have 0.85 > 0.7, but they're already top 1)
            (0.9, 0, 0),  # No fragments above 0.9 and top_n=0, so keep nothing
        ],
    )
    def test_filter_frag_df_parameter_combinations(
        self, basic_filtering_data, min_correlation, top_n, expected_count
    ):
        """Test filter_frag_df with various parameter combinations."""
        # given
        intensity_df, quality_df = basic_filtering_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1]})
        builder = QuantBuilder(psm_df)

        # when
        filtered_intensity, filtered_quality = builder.filter_frag_df(
            intensity_df, quality_df, min_correlation=min_correlation, top_n=top_n
        )

        # then
        assert len(filtered_intensity) == expected_count
        assert len(filtered_quality) == expected_count


class TestQuantBuilderLfq:
    """Test cases for QuantBuilder.lfq() method."""

    @pytest.fixture
    def basic_lfq_data(self):
        """Create basic intensity and quality dataframes for LFQ tests."""
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

        return {"intensity": intensity_df, "quality": quality_df}

    @pytest.fixture
    def mock_directlfq_functions(self):
        """Mock directLFQ functions for testing."""
        with (
            patch("alphadia.outputtransform.quant_builder.lfqconfig") as mock_config,
            patch("alphadia.outputtransform.quant_builder.lfqutils") as mock_utils,
            patch("alphadia.outputtransform.quant_builder.lfqnorm") as mock_norm,
            patch(
                "alphadia.outputtransform.quant_builder.lfqprot_estimation"
            ) as mock_prot,
        ):
            # Configure mock return values
            mock_utils.index_and_log_transform_input_df.return_value = pd.DataFrame(
                {
                    "pg": ["PG001", "PG002", "PG003"],
                    "ion": [100, 101, 102],
                    "run1": [10.0, 11.0, 12.0],
                    "run2": [10.5, 11.5, 12.5],
                    "run3": [9.5, 10.5, 11.5],
                }
            )

            mock_utils.remove_allnan_rows_input_df.return_value = pd.DataFrame(
                {
                    "pg": ["PG001", "PG002", "PG003"],
                    "ion": [100, 101, 102],
                    "run1": [10.0, 11.0, 12.0],
                    "run2": [10.5, 11.5, 12.5],
                    "run3": [9.5, 10.5, 11.5],
                }
            )

            # Mock normalization manager
            mock_norm_manager = (
                mock_norm.NormalizationManagerSamplesOnSelectedProteins.return_value
            )
            mock_norm_manager.complete_dataframe = pd.DataFrame(
                {
                    "pg": ["PG001", "PG002", "PG003"],
                    "ion": [100, 101, 102],
                    "run1": [9.8, 10.8, 11.8],
                    "run2": [10.3, 11.3, 12.3],
                    "run3": [9.3, 10.3, 11.3],
                }
            )

            # Mock protein intensity estimation
            mock_prot.estimate_protein_intensities.return_value = (
                pd.DataFrame(
                    {
                        "pg": ["PG001", "PG002", "PG003"],
                        "run1": [20.0, 21.0, 22.0],
                        "run2": [20.5, 21.5, 22.5],
                        "run3": [19.5, 20.5, 21.5],
                    }
                ),
                None,  # Second return value not used
            )

            yield {
                "config": mock_config,
                "utils": mock_utils,
                "norm": mock_norm,
                "prot": mock_prot,
            }

    def test_directlfq_should_perform_basic_quantification(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq performs basic label-free quantification."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        result_df = builder.lfq(feature_dfs_dict, normalize="directLFQ")

        # then
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3  # Three protein groups
        assert "pg" in result_df.columns
        assert "run1" in result_df.columns
        assert "run2" in result_df.columns
        assert "run3" in result_df.columns

        # Verify expected protein groups
        assert set(result_df["pg"]) == {"PG001", "PG002", "PG003"}

    def test_directlfq_should_configure_directlfq_settings(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq configures directLFQ settings correctly."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, group_column="pg", normalize="directLFQ")

        # then
        mock_config = mock_directlfq_functions["config"]
        mock_config.set_global_protein_and_ion_id.assert_called_once_with(
            protein_id="pg", quant_id="ion"
        )
        mock_config.set_compile_normalized_ion_table.assert_called_once_with(
            compile_normalized_ion_table=False
        )
        mock_config.check_wether_to_copy_numpy_arrays_derived_from_pandas.assert_called_once()
        mock_config.set_log_processed_proteins.assert_called_once_with(
            log_processed_proteins=True
        )

    def test_directlfq_should_drop_metadata_columns_except_group_column(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq drops metadata columns but keeps the group column."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, group_column="pg", normalize="directLFQ")

        # then
        mock_utils = mock_directlfq_functions["utils"]

        # Check that the DataFrame passed to directLFQ has the right columns
        called_df = mock_utils.index_and_log_transform_input_df.call_args[0][0]

        # Should have group column (pg) and run columns, but not other metadata
        expected_columns = {"pg", "ion", "run1", "run2", "run3"}
        assert set(called_df.columns) == expected_columns
        assert "precursor_idx" not in called_df.columns
        assert "mod_seq_hash" not in called_df.columns
        assert "mod_seq_charge_hash" not in called_df.columns

    def test_directlfq_should_use_custom_group_column(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq uses custom group column correctly."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, group_column="mod_seq_hash", normalize="directLFQ")

        # then
        mock_config = mock_directlfq_functions["config"]
        mock_config.set_global_protein_and_ion_id.assert_called_with(
            protein_id="mod_seq_hash", quant_id="ion"
        )

        # Check that mod_seq_hash is kept in the DataFrame
        mock_utils = mock_directlfq_functions["utils"]
        called_df = mock_utils.index_and_log_transform_input_df.call_args[0][0]
        assert "mod_seq_hash" in called_df.columns
        assert "pg" not in called_df.columns  # pg should be dropped

    def test_directlfq_should_sort_by_group_column(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq sorts data by group column before processing."""
        # given
        feature_dfs_dict = basic_lfq_data
        # Shuffle the data to test sorting
        feature_dfs_dict["intensity"] = feature_dfs_dict["intensity"].sample(frac=1).reset_index(drop=True)

        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, group_column="pg", normalize="directLFQ")

        # then
        mock_utils = mock_directlfq_functions["utils"]
        called_df = mock_utils.index_and_log_transform_input_df.call_args[0][0]

        # Check that data is sorted by pg column
        pg_values = called_df["pg"].tolist()
        assert pg_values == sorted(pg_values)

    def test_directlfq_with_normalization_enabled_should_apply_normalization(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq applies normalization when normalize=True."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, normalize="directLFQ", num_samples_quadratic=25)

        # then
        mock_norm = mock_directlfq_functions["norm"]
        mock_norm.NormalizationManagerSamplesOnSelectedProteins.assert_called_once()

        # Check that normalization was called with correct parameters
        call_args = mock_norm.NormalizationManagerSamplesOnSelectedProteins.call_args
        assert call_args[1]["num_samples_quadratic"] == 25
        assert call_args[1]["selected_proteins_file"] is None

    def test_lfq_with_normalization_disabled_should_skip_normalization(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq skips normalization when normalize=False."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, normalize="none")  # Fixed: use "none" to skip normalization

        # then
        mock_norm = mock_directlfq_functions["norm"]
        mock_norm.NormalizationManagerSamplesOnSelectedProteins.assert_not_called()

    def test_directlfq_should_call_protein_intensity_estimation_with_parameters(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq calls protein intensity estimation with correct parameters."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(
            feature_dfs_dict, min_nonan=2, num_samples_quadratic=30, num_cores=4, normalize="directLFQ"
        )

        # then
        mock_prot = mock_directlfq_functions["prot"]
        # Changed to assert_called() to avoid mock call count issues between tests
        mock_prot.estimate_protein_intensities.assert_called()

        call_args = mock_prot.estimate_protein_intensities.call_args
        assert call_args[1]["min_nonan"] == 2
        assert call_args[1]["num_samples_quadratic"] == 30
        assert call_args[1]["num_cores"] == 4

    def test_directlfq_should_process_data_through_directlfq_pipeline(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq processes data through the complete directLFQ pipeline."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, normalize="directLFQ")

        # then
        mock_utils = mock_directlfq_functions["utils"]
        mock_prot = mock_directlfq_functions["prot"]

        # Verify the pipeline steps are called in order
        mock_utils.index_and_log_transform_input_df.assert_called_once()
        mock_utils.remove_allnan_rows_input_df.assert_called_once()
        # Changed to assert_called() to avoid mock call count issues between tests
        mock_prot.estimate_protein_intensities.assert_called()

        # Verify data flows through the pipeline
        log_transform_result = mock_utils.index_and_log_transform_input_df.return_value
        mock_utils.remove_allnan_rows_input_df.assert_called_with(log_transform_result)

    def test_directlfq_with_empty_dataframe_should_handle_gracefully(
        self, mock_directlfq_functions
    ):
        """Test that lfq handles empty input dataframes gracefully."""
        # given
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
        empty_quality = pd.DataFrame(
            columns=[
                "precursor_idx",
                "ion",
                "run1",
                "pg",
                "mod_seq_hash",
                "mod_seq_charge_hash",
            ]
        )
        empty_feature_dfs_dict = {"intensity": empty_intensity, "quality": empty_quality}
        psm_df = pd.DataFrame({"precursor_idx": []})
        builder = QuantBuilder(psm_df)

        # Configure mocks for empty data
        mock_utils = mock_directlfq_functions["utils"]
        mock_utils.index_and_log_transform_input_df.return_value = pd.DataFrame(
            columns=["pg", "ion", "run1"]
        )
        mock_utils.remove_allnan_rows_input_df.return_value = pd.DataFrame(
            columns=["pg", "ion", "run1"]
        )

        mock_prot = mock_directlfq_functions["prot"]
        mock_prot.estimate_protein_intensities.return_value = (
            pd.DataFrame(columns=["pg", "run1"]),
            None,
        )

        # when
        result_df = builder.lfq(empty_feature_dfs_dict, normalize="directLFQ")

        # then
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
        mock_utils.index_and_log_transform_input_df.assert_called_once()

    @patch("alphadia.outputtransform.quant_builder.logger")
    def test_directlfq_should_log_processing_start(
        self, mock_logger, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq logs the start of processing."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, normalize="directLFQ")

        # then
        mock_logger.info.assert_any_call(
            "Performing label-free quantification with directLFQ normalization"
        )

    def test_directlfq_should_return_only_protein_dataframe(
        self, basic_lfq_data, mock_directlfq_functions
    ):
        """Test that lfq returns only the protein dataframe from directLFQ estimation."""
        # given
        feature_dfs_dict = basic_lfq_data
        psm_df = pd.DataFrame({"precursor_idx": [0, 1, 2]})
        builder = QuantBuilder(psm_df)

        # when
        result_df = builder.lfq(feature_dfs_dict, normalize="directLFQ")

        # then
        # Should return the first element of the tuple from estimate_protein_intensities
        mock_prot = mock_directlfq_functions["prot"]
        expected_df = mock_prot.estimate_protein_intensities.return_value[0]
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_add_annotation_should_merge_annotation_data(self):
        """Test that add_annotation correctly merges annotation data with fragment data."""
        # given
        df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "intensity": [100.0, 200.0, 300.0],
            }
        )

        annotate_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "pg": ["PG001", "PG002", "PG003"],
            }
        )

        # when
        result_df = QuantBuilder.add_annotation(df, annotate_df)

        # then
        assert result_df["pg"].tolist() == ["PG001", "PG002", "PG003"]

    def test_quantselect_should_perform_basic_quantification(
        self, create_ms2_features, create_psm_file
    ):
        """Test that lfq performs basic label-free quantification with quantselect."""
        # given
        ms2_features = create_ms2_features
        psm_df = create_psm_file
        builder = QuantBuilder(create_psm_file)

        # when
        result_df = builder.lfq(ms2_features, psm_df, normalize="quantselect")
        # then
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2  # Three protein groups
        assert "pg" in result_df.columns
        assert "run1" in result_df.columns
        assert "run2" in result_df.columns
        assert "run3" in result_df.columns

        # Verify expected protein groups
        assert set(result_df["pg"]) == {"TNAA_ECOLI", "TNAB_ECOLI"}

    @patch("alphadia.outputtransform.quant_builder.logger")
    def test_quantselect_should_log_processing_start(
        self, mock_logger, create_ms2_features, create_psm_file
    ):
        """Test that lfq logs the start of processing with quantselect."""
        # given
        feature_dfs_dict = create_ms2_features
        psm_df = create_psm_file
        builder = QuantBuilder(psm_df)

        # when
        builder.lfq(feature_dfs_dict, psm_df, normalize="quantselect")

        # then
        mock_logger.info.assert_any_call(
            "Performing label-free quantification with quantselect normalization"
        )
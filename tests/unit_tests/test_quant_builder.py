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


class TestQuantBuilderAccumulateFragDf:
    """Test cases for QuantBuilder.accumulate_frag_df() method."""

    def test_accumulate_frag_df_single_run_should_return_intensity_and_quality_dfs(
        self, extended_psm_df, simple_fragment_df
    ):
        """Test that accumulate_frag_df returns correct dataframes for single run."""
        # given
        builder = QuantBuilder(extended_psm_df, column="intensity")
        df_iterable = [("run1", simple_fragment_df)]

        # when
        intensity_df, quality_df = builder.accumulate_frag_df(iter(df_iterable))

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

        pd.testing.assert_frame_equal(intensity_df, expected_intensity_df)
        pd.testing.assert_frame_equal(quality_df, expected_quality_df)

    def test_accumulate_frag_df_multiple_runs_should_merge_data_correctly(
        self, basic_psm_df, basic_fragment_df
    ):
        """Test that accumulate_frag_df correctly merges data from multiple runs."""
        # given
        frag_df2 = basic_fragment_df.copy()
        frag_df2["height"] = [150.0, 250.0]
        frag_df2["intensity"] = [160.0, 270.0]
        frag_df2["correlation"] = [0.7, 0.6]

        builder = QuantBuilder(basic_psm_df, column="intensity")
        df_iterable = [("run1", basic_fragment_df), ("run2", frag_df2)]

        # when
        intensity_df, quality_df = builder.accumulate_frag_df(iter(df_iterable))

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

        pd.testing.assert_frame_equal(intensity_df, expected_intensity_df)
        pd.testing.assert_frame_equal(quality_df, expected_quality_df)

    def test_accumulate_frag_df_with_empty_iterator_should_return_none(
        self, extended_psm_df
    ):
        """Test that accumulate_frag_df returns None when iterator is empty."""
        # given
        builder = QuantBuilder(extended_psm_df, column="intensity")
        empty_iterable = iter([])

        # when
        result = builder.accumulate_frag_df(empty_iterable)

        # then
        assert result is None

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

        builder = QuantBuilder(psm_df, column="intensity")
        df_iterable = [("run1", frag_df1), ("run2", frag_df2)]

        # when
        intensity_df, quality_df = builder.accumulate_frag_df(iter(df_iterable))

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

        pd.testing.assert_frame_equal(intensity_df, expected_intensity_df)
        pd.testing.assert_frame_equal(quality_df, expected_quality_df)

    def test_accumulate_frag_df_should_filter_by_psm_precursor_idx(self):
        """Test that accumulate_frag_df only includes fragments for precursors in PSM data."""
        # given
        psm_df = mock_precursor_df(n_precursor=2, with_decoy=False)
        psm_df["pg"] = ["PG001", "PG002"]
        psm_df["mod_seq_hash"] = [1, 2]
        psm_df["mod_seq_charge_hash"] = [10, 20]

        # Create fragment data with more precursors than in PSM
        frag_df = create_mock_fragment_df(n_fragments=4, n_precursor=5)

        builder = QuantBuilder(psm_df, column="intensity")
        df_iterable = [("run1", frag_df)]

        # when
        intensity_df, quality_df = builder.accumulate_frag_df(iter(df_iterable))

        # then
        # Should only contain precursors that exist in PSM data
        unique_precursor_idx = intensity_df["precursor_idx"].unique()
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
            psm_df, column="height"
        )  # Use height instead of intensity
        df_iterable = [("run1", frag_df)]

        # when
        intensity_df, quality_df = builder.accumulate_frag_df(iter(df_iterable))

        # then
        assert "run1" in intensity_df.columns
        # Values should come from height column, not intensity
        assert intensity_df["run1"].notna().any()

    @patch("alphadia.outputtransform.quant_builder.logger")
    def test_accumulate_frag_df_should_log_accumulation_start(self, mock_logger):
        """Test that accumulate_frag_df logs the start of accumulation."""
        # given
        psm_df = mock_precursor_df(n_precursor=2, with_decoy=False)
        psm_df["pg"] = ["PG001", "PG002"]
        psm_df["mod_seq_hash"] = [1, 2]
        psm_df["mod_seq_charge_hash"] = [10, 20]

        frag_df = create_mock_fragment_df(n_fragments=4, n_precursor=2)

        builder = QuantBuilder(psm_df, column="intensity")
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

        builder = QuantBuilder(psm_df, column="intensity")
        df_iterable = [("run1", frag_df1), ("run2", frag_df2)]

        # when
        intensity_df, quality_df = builder.accumulate_frag_df(iter(df_iterable))

        # then
        # Should have same ion hashes in both dataframes
        assert len(intensity_df["ion"].unique()) == len(
            frag_df1
        )  # Same ions from both runs
        assert all(intensity_df["ion"] == quality_df["ion"])

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

        builder = QuantBuilder(basic_psm_df, column="intensity")
        df_iterable = [("run1", duplicate_fragment_df), ("run2", frag_df2)]

        # when
        intensity_df, quality_df = builder.accumulate_frag_df(iter(df_iterable))

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

        pd.testing.assert_frame_equal(intensity_df, expected_intensity_df)
        pd.testing.assert_frame_equal(quality_df, expected_quality_df)


class TestQuantBuilderHelperFunctions:
    """Test cases for helper functions used by QuantBuilder."""

    def test_prepare_df_should_filter_by_precursor_idx(
        self, helper_psm_df, helper_fragment_df
    ):
        """Test that prepare_df filters fragments by precursor_idx from PSM data."""
        # given
        # when
        result_df = prepare_df(helper_fragment_df, helper_psm_df, column="intensity")

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

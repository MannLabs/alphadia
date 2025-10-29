from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform.quantification import FragmentQuantLoader


@pytest.fixture
def basic_psm_df():
    """Basic PSM DataFrame with precursor information."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "pg": ["PG001", "PG002", "PG003"],
            "mod_seq_hash": [1, 2, 3],
            "mod_seq_charge_hash": [10, 20, 30],
        }
    )


@pytest.fixture
def basic_fragment_df():
    """Basic fragment DataFrame."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "mz": [500.1, 600.2, 700.3],
            "charge": np.array([1, 2, 1], dtype=np.uint8),
            "number": np.array([1, 1, 1], dtype=np.uint8),
            "type": np.array([98, 121, 98], dtype=np.uint8),
            "position": np.array([0, 0, 0], dtype=np.uint8),
            "height": [100.0, 200.0, 300.0],
            "intensity": [110.0, 220.0, 330.0],
            "correlation": [0.8, 0.9, 0.7],
            "loss_type": np.array([1, 1, 1], dtype=np.uint8),
        }
    )


class TestFragmentQuantLoaderAccumulate:
    """Test cases for FragmentQuantLoader.accumulate() method."""

    def test_accumulate_single_run(self, basic_psm_df, basic_fragment_df):
        """Test that accumulate handles a single run correctly."""
        loader = FragmentQuantLoader(basic_psm_df)
        df_iterable = iter([("run1", basic_fragment_df)])

        result = loader.accumulate(df_iterable)

        # Create expected dataframes
        expected_intensity = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1, 2], dtype=np.uint32),
                "ion": [72446825449127936, 72753589193277441, 72446825449127938],
                "run1": [110.0, 220.0, 330.0],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        expected_correlation = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1, 2], dtype=np.uint32),
                "ion": [72446825449127936, 72753589193277441, 72446825449127938],
                "run1": [0.8, 0.9, 0.7],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        pd.testing.assert_frame_equal(result["intensity"], expected_intensity)
        pd.testing.assert_frame_equal(result["correlation"], expected_correlation)

    def test_accumulate_multiple_runs(self, basic_psm_df, basic_fragment_df):
        """Test that accumulate merges multiple runs correctly."""
        loader = FragmentQuantLoader(basic_psm_df)

        frag_df2 = basic_fragment_df.copy()
        frag_df2["intensity"] = [120.0, 240.0, 360.0]
        frag_df2["correlation"] = [0.75, 0.85, 0.65]

        df_iterable = iter([("run1", basic_fragment_df), ("run2", frag_df2)])

        result = loader.accumulate(df_iterable)

        expected_intensity = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [110.0, 330.0, 220.0],
                "run2": [120.0, 360.0, 240.0],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        expected_correlation = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [0.8, 0.7, 0.9],
                "run2": [0.75, 0.65, 0.85],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        pd.testing.assert_frame_equal(result["intensity"], expected_intensity)
        pd.testing.assert_frame_equal(result["correlation"], expected_correlation)

    def test_accumulate_with_empty_iterator(self, basic_psm_df):
        """Test that accumulate returns None for empty iterator."""
        loader = FragmentQuantLoader(basic_psm_df)
        df_iterable = iter([])

        result = loader.accumulate(df_iterable)

        assert result is None

    def test_accumulate_fills_missing_values_with_zero(self, basic_psm_df):
        """Test that accumulate fills missing values with zeros."""
        loader = FragmentQuantLoader(basic_psm_df)

        frag_df1 = pd.DataFrame(
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

        frag_df2 = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
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

        df_iterable = iter([("run1", frag_df1), ("run2", frag_df2)])
        result = loader.accumulate(df_iterable)

        expected_intensity = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [110.0, 0.0, 220.0],  # run1 missing precursor_idx=2
                "run2": [160.0, 380.0, 270.0],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        expected_correlation = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [0.8, 0.0, 0.9],
                "run2": [0.7, 0.5, 0.6],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        pd.testing.assert_frame_equal(result["intensity"], expected_intensity)
        pd.testing.assert_frame_equal(result["correlation"], expected_correlation)

    def test_accumulate_filters_by_psm_precursor_idx(self, basic_psm_df):
        """Test that accumulate only includes fragments for precursors in PSM data."""
        loader = FragmentQuantLoader(basic_psm_df)

        frag_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2, 3, 4],
                "mz": [500.1, 600.2, 700.3, 800.4, 900.5],
                "charge": np.array([1, 2, 1, 2, 1], dtype=np.uint8),
                "number": np.array([1, 1, 1, 1, 1], dtype=np.uint8),
                "type": np.array([98, 121, 98, 121, 98], dtype=np.uint8),
                "position": np.array([0, 0, 0, 0, 0], dtype=np.uint8),
                "height": [100.0, 200.0, 300.0, 400.0, 500.0],
                "intensity": [110.0, 220.0, 330.0, 440.0, 550.0],
                "correlation": [0.8, 0.9, 0.7, 0.6, 0.5],
                "loss_type": np.array([1, 1, 1, 1, 1], dtype=np.uint8),
            }
        )

        df_iterable = iter([("run1", frag_df)])
        result = loader.accumulate(df_iterable)

        # Only precursor_idx 0, 1, 2 should be included (from basic_psm_df)
        expected_intensity = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1, 2], dtype=np.uint32),
                "ion": [72446825449127936, 72753589193277441, 72446825449127938],
                "run1": [110.0, 220.0, 330.0],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        expected_correlation = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 1, 2], dtype=np.uint32),
                "ion": [72446825449127936, 72753589193277441, 72446825449127938],
                "run1": [0.8, 0.9, 0.7],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        pd.testing.assert_frame_equal(result["intensity"], expected_intensity)
        pd.testing.assert_frame_equal(result["correlation"], expected_correlation)


class TestFragmentQuantLoaderAccumulateFromFolders:
    """Test cases for FragmentQuantLoader.accumulate_from_folders() method."""

    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_accumulate_from_folders_reads_parquet_files(
        self, mock_read_parquet, mock_exists, basic_psm_df, basic_fragment_df
    ):
        """Test that accumulate_from_folders reads parquet files from folders."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = basic_fragment_df

        loader = FragmentQuantLoader(basic_psm_df)
        result = loader.accumulate_from_folders(["folder1", "folder2"])

        assert mock_read_parquet.call_count == 2

        expected_intensity = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "folder1": [110.0, 330.0, 220.0],
                "folder2": [110.0, 330.0, 220.0],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        expected_correlation = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "folder1": [0.8, 0.7, 0.9],
                "folder2": [0.8, 0.7, 0.9],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        pd.testing.assert_frame_equal(result["intensity"], expected_intensity)
        pd.testing.assert_frame_equal(result["correlation"], expected_correlation)

    @patch("os.path.exists")
    def test_accumulate_from_folders_skips_missing_files(
        self, mock_exists, basic_psm_df
    ):
        """Test that accumulate_from_folders skips folders without frag.parquet."""
        mock_exists.return_value = False

        loader = FragmentQuantLoader(basic_psm_df)
        result = loader.accumulate_from_folders(["missing_folder"])

        assert result is None

    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_accumulate_from_folders_handles_read_errors(
        self, mock_read_parquet, mock_exists, basic_psm_df
    ):
        """Test that accumulate_from_folders handles read errors gracefully."""
        mock_exists.return_value = True
        mock_read_parquet.side_effect = Exception("Read error")

        loader = FragmentQuantLoader(basic_psm_df)
        result = loader.accumulate_from_folders(["error_folder"])

        assert result is None

    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_accumulate_from_folders_uses_folder_basename_as_run_name(
        self, mock_read_parquet, mock_exists, basic_psm_df, basic_fragment_df
    ):
        """Test that accumulate_from_folders uses folder basename as run name."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = basic_fragment_df

        loader = FragmentQuantLoader(basic_psm_df)
        result = loader.accumulate_from_folders(["/path/to/run1", "/path/to/run2"])

        expected_intensity = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [110.0, 330.0, 220.0],
                "run2": [110.0, 330.0, 220.0],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        expected_correlation = pd.DataFrame(
            {
                "precursor_idx": np.array([0, 2, 1], dtype=np.uint32),
                "ion": [72446825449127936, 72446825449127938, 72753589193277441],
                "run1": [0.8, 0.7, 0.9],
                "run2": [0.8, 0.7, 0.9],
                "pg": ["PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 3, 2],
                "mod_seq_charge_hash": [10, 30, 20],
            }
        )

        pd.testing.assert_frame_equal(result["intensity"], expected_intensity)
        pd.testing.assert_frame_equal(result["correlation"], expected_correlation)

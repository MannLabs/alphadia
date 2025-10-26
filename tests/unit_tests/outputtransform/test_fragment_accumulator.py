from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform.fragment_accumulator import FragmentQuantLoader


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


class TestFragmentQuantLoaderInitialization:
    """Test cases for FragmentQuantLoader initialization."""

    def test_initialization_with_default_columns(self, basic_psm_df):
        """Test that FragmentQuantLoader initializes with default columns."""
        loader = FragmentQuantLoader(basic_psm_df)

        assert loader.psm_df is basic_psm_df
        assert loader.columns == ["intensity", "correlation"]

    def test_initialization_with_custom_columns(self, basic_psm_df):
        """Test that FragmentQuantLoader initializes with custom columns."""
        loader = FragmentQuantLoader(basic_psm_df, columns=["height", "intensity"])

        assert loader.columns == ["height", "intensity"]


class TestFragmentQuantLoaderAccumulate:
    """Test cases for FragmentQuantLoader.accumulate() method."""

    def test_accumulate_single_run(self, basic_psm_df, basic_fragment_df):
        """Test that accumulate handles a single run correctly."""
        loader = FragmentQuantLoader(basic_psm_df)
        df_iterable = iter([("run1", basic_fragment_df)])

        result = loader.accumulate(df_iterable)

        assert "intensity" in result
        assert "correlation" in result
        assert "run1" in result["intensity"].columns
        assert "pg" in result["intensity"].columns
        assert len(result["intensity"]) == 3

    def test_accumulate_multiple_runs(self, basic_psm_df, basic_fragment_df):
        """Test that accumulate merges multiple runs correctly."""
        loader = FragmentQuantLoader(basic_psm_df)

        frag_df2 = basic_fragment_df.copy()
        frag_df2["intensity"] = [120.0, 240.0, 360.0]
        frag_df2["correlation"] = [0.75, 0.85, 0.65]

        df_iterable = iter([("run1", basic_fragment_df), ("run2", frag_df2)])

        result = loader.accumulate(df_iterable)

        assert "run1" in result["intensity"].columns
        assert "run2" in result["intensity"].columns
        assert len(result["intensity"]) == 3

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

        assert (result["intensity"]["run1"] == 0).any()

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

        unique_precursor_idx = result["intensity"]["precursor_idx"].unique()
        assert all(
            idx in basic_psm_df["precursor_idx"].values for idx in unique_precursor_idx
        )
        assert 3 not in unique_precursor_idx
        assert 4 not in unique_precursor_idx

    def test_accumulate_adds_annotation_columns(self, basic_psm_df, basic_fragment_df):
        """Test that accumulate adds pg, mod_seq_hash, and mod_seq_charge_hash columns."""
        loader = FragmentQuantLoader(basic_psm_df)
        df_iterable = iter([("run1", basic_fragment_df)])

        result = loader.accumulate(df_iterable)

        assert "pg" in result["intensity"].columns
        assert "mod_seq_hash" in result["intensity"].columns
        assert "mod_seq_charge_hash" in result["intensity"].columns


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
        assert result is not None

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

        assert "run1" in result["intensity"].columns
        assert "run2" in result["intensity"].columns


class TestFragmentQuantLoaderGetFragDfGenerator:
    """Test cases for FragmentQuantLoader._get_frag_df_generator() method."""

    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_get_frag_df_generator_yields_tuples(
        self, mock_read_parquet, mock_exists, basic_psm_df, basic_fragment_df
    ):
        """Test that _get_frag_df_generator yields (run_name, dataframe) tuples."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = basic_fragment_df

        loader = FragmentQuantLoader(basic_psm_df)
        generator = loader._get_frag_df_generator(["folder1", "folder2"])

        results = list(generator)
        assert len(results) == 2
        assert results[0][0] == "folder1"
        assert results[1][0] == "folder2"
        pd.testing.assert_frame_equal(results[0][1], basic_fragment_df)

    @patch("os.path.exists")
    def test_get_frag_df_generator_skips_missing_files(self, mock_exists, basic_psm_df):
        """Test that _get_frag_df_generator skips missing files."""
        mock_exists.return_value = False

        loader = FragmentQuantLoader(basic_psm_df)
        generator = loader._get_frag_df_generator(["missing1", "missing2"])

        results = list(generator)
        assert len(results) == 0


class TestFragmentQuantLoaderAddAnnotation:
    """Test cases for FragmentQuantLoader._add_annotation() static method."""

    def test_add_annotation_merges_annotation_data(self):
        """Test that _add_annotation merges annotation data correctly."""
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
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        result = FragmentQuantLoader._add_annotation(df, annotate_df)

        assert "pg" in result.columns
        assert "mod_seq_hash" in result.columns
        assert "mod_seq_charge_hash" in result.columns
        assert result["pg"].tolist() == ["PG001", "PG002", "PG003"]

    def test_add_annotation_fills_nan_with_zero(self):
        """Test that _add_annotation fills NaN values with zero."""
        df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "intensity": [100.0, np.nan, 300.0],
            }
        )

        annotate_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        result = FragmentQuantLoader._add_annotation(df, annotate_df)

        assert result["intensity"].iloc[1] == 0.0

    def test_add_annotation_converts_precursor_idx_to_uint32(self):
        """Test that _add_annotation converts precursor_idx to uint32."""
        df = pd.DataFrame(
            {
                "precursor_idx": [0.0, 1.0, 2.0],
                "intensity": [100.0, 200.0, 300.0],
            }
        )

        annotate_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2],
                "pg": ["PG001", "PG002", "PG003"],
                "mod_seq_hash": [1, 2, 3],
                "mod_seq_charge_hash": [10, 20, 30],
            }
        )

        result = FragmentQuantLoader._add_annotation(df, annotate_df)

        assert result["precursor_idx"].dtype == np.uint32

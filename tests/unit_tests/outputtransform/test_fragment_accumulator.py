from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform.quantification import FragmentQuantLoader


@pytest.fixture
def psm_df():
    """PSM DataFrame with precursor information."""
    return pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2],
            "pg": ["PG001", "PG002", "PG003"],
            "mod_seq_hash": [1, 2, 3],
            "mod_seq_charge_hash": [10, 20, 30],
        }
    )


@pytest.fixture
def fragment_df():
    """Fragment DataFrame with all required columns."""
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

    def test_accumulate_single_run(self, psm_df, fragment_df):
        """Test accumulate with single run."""
        # given
        loader = FragmentQuantLoader(psm_df)
        df_iterable = iter([("run1", fragment_df)])

        # when
        result = loader.accumulate(df_iterable)

        # then
        assert result["intensity"].shape == (3, 6)
        assert list(result["intensity"].columns) == [
            "precursor_idx",
            "ion",
            "run1",
            "pg",
            "mod_seq_hash",
            "mod_seq_charge_hash",
        ]
        assert result["intensity"]["run1"].tolist() == [110.0, 220.0, 330.0]
        assert result["correlation"]["run1"].tolist() == [0.8, 0.9, 0.7]

    def test_accumulate_multiple_runs(self, psm_df, fragment_df):
        """Test accumulate merges multiple runs correctly."""
        # given
        loader = FragmentQuantLoader(psm_df)
        frag_df2 = fragment_df.copy()
        frag_df2["intensity"] = [120.0, 240.0, 360.0]
        frag_df2["correlation"] = [0.75, 0.85, 0.65]
        df_iterable = iter([("run1", fragment_df), ("run2", frag_df2)])

        # when
        result = loader.accumulate(df_iterable)

        # then
        assert "run1" in result["intensity"].columns
        assert "run2" in result["intensity"].columns
        assert result["intensity"].shape[0] == 3

    def test_accumulate_empty_iterator(self, psm_df):
        """Test accumulate returns None for empty iterator."""
        # given
        loader = FragmentQuantLoader(psm_df)
        df_iterable = iter([])

        # when
        result = loader.accumulate(df_iterable)

        # then
        assert result is None

    def test_accumulate_fills_missing_values(self, psm_df, fragment_df):
        """Test accumulate fills missing values with zeros."""
        # given
        loader = FragmentQuantLoader(psm_df)
        frag_df1 = fragment_df[fragment_df["precursor_idx"] != 2].copy()
        df_iterable = iter([("run1", frag_df1), ("run2", fragment_df)])

        # when
        result = loader.accumulate(df_iterable)

        # then
        assert (
            result["intensity"]
            .loc[result["intensity"]["precursor_idx"] == 2, "run1"]
            .iloc[0]
            == 0.0
        )
        assert (
            result["correlation"]
            .loc[result["correlation"]["precursor_idx"] == 2, "run1"]
            .iloc[0]
            == 0.0
        )

    def test_accumulate_filters_by_psm_precursor_idx(self, psm_df, fragment_df):
        """Test accumulate only includes fragments for precursors in PSM data."""
        # given
        loader = FragmentQuantLoader(psm_df)
        frag_df_extra = pd.concat(
            [
                fragment_df,
                pd.DataFrame(
                    {
                        "precursor_idx": [3, 4],
                        "mz": [800.4, 900.5],
                        "charge": np.array([2, 1], dtype=np.uint8),
                        "number": np.array([1, 1], dtype=np.uint8),
                        "type": np.array([121, 98], dtype=np.uint8),
                        "position": np.array([0, 0], dtype=np.uint8),
                        "height": [400.0, 500.0],
                        "intensity": [440.0, 550.0],
                        "correlation": [0.6, 0.5],
                        "loss_type": np.array([1, 1], dtype=np.uint8),
                    }
                ),
            ]
        )
        df_iterable = iter([("run1", frag_df_extra)])

        # when
        result = loader.accumulate(df_iterable)

        # then
        assert result["intensity"].shape[0] == 3
        assert result["intensity"]["precursor_idx"].max() == 2


class TestFragmentQuantLoaderAccumulateFromFolders:
    """Test cases for FragmentQuantLoader.accumulate_from_folders() method."""

    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_accumulate_from_folders_success(
        self, mock_read_parquet, mock_exists, psm_df, fragment_df
    ):
        """Test accumulate_from_folders reads parquet files and uses folder basename."""
        # given
        mock_exists.return_value = True
        mock_read_parquet.return_value = fragment_df
        loader = FragmentQuantLoader(psm_df)
        folders = ["/path/to/run1", "/path/to/run2"]

        # when
        result = loader.accumulate_from_folders(folders)

        # then
        assert mock_read_parquet.call_count == 2
        assert "run1" in result["intensity"].columns
        assert "run2" in result["intensity"].columns
        assert result["intensity"].shape[0] == 3

    @patch("os.path.exists")
    def test_accumulate_from_folders_missing_files(self, mock_exists, psm_df):
        """Test accumulate_from_folders returns None for missing files."""
        # given
        mock_exists.return_value = False
        loader = FragmentQuantLoader(psm_df)

        # when
        result = loader.accumulate_from_folders(["missing_folder"])

        # then
        assert result is None

    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_accumulate_from_folders_read_errors(
        self, mock_read_parquet, mock_exists, psm_df
    ):
        """Test accumulate_from_folders handles read errors gracefully."""
        # given
        mock_exists.return_value = True
        mock_read_parquet.side_effect = Exception("Read error")
        loader = FragmentQuantLoader(psm_df)

        # when
        result = loader.accumulate_from_folders(["error_folder"])

        # then
        assert result is None

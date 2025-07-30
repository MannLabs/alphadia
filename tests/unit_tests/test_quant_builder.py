from unittest.mock import patch

import numpy as np
import pandas as pd
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


class TestQuantBuilderAccumulateFragDf:
    """Test cases for QuantBuilder.accumulate_frag_df() method."""

    def test_accumulate_frag_df_single_run_should_return_intensity_and_quality_dfs(
        self,
    ):
        """Test that accumulate_frag_df returns correct dataframes for single run."""
        # given
        psm_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1, 2, 3, 4],
                "pg": ["PG001", "PG002", "PG001", "PG003", "PG002"],
                "mod_seq_hash": [1, 2, 3, 4, 5],
                "mod_seq_charge_hash": [10, 20, 30, 40, 50],
            }
        )

        frag_df = pd.DataFrame(
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

        builder = QuantBuilder(psm_df, column="intensity")
        df_iterable = [("run1", frag_df)]

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

    def test_accumulate_frag_df_multiple_runs_should_merge_data_correctly(self):
        """Test that accumulate_frag_df correctly merges data from multiple runs."""
        # given
        psm_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1],
                "pg": ["PG001", "PG002"],
                "mod_seq_hash": [1, 2],
                "mod_seq_charge_hash": [10, 20],
            }
        )

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
                "precursor_idx": [0, 1],
                "mz": [500.1, 600.2],
                "charge": np.array([1, 2], dtype=np.uint8),
                "number": np.array([1, 1], dtype=np.uint8),
                "type": np.array([98, 121], dtype=np.uint8),
                "position": np.array([0, 0], dtype=np.uint8),
                "height": [150.0, 250.0],
                "intensity": [160.0, 270.0],
                "correlation": [0.7, 0.6],
                "loss_type": np.array([1, 1], dtype=np.uint8),
            }
        )

        builder = QuantBuilder(psm_df, column="intensity")
        df_iterable = [("run1", frag_df1), ("run2", frag_df2)]

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

    def test_accumulate_frag_df_with_empty_iterator_should_return_none(self):
        """Test that accumulate_frag_df returns None when iterator is empty."""
        # given
        psm_df = mock_precursor_df(n_precursor=5, with_decoy=False)
        builder = QuantBuilder(psm_df, column="intensity")
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


class TestQuantBuilderHelperFunctions:
    """Test cases for helper functions used by QuantBuilder."""

    def test_prepare_df_should_filter_by_precursor_idx(self):
        """Test that prepare_df filters fragments by precursor_idx from PSM data."""
        # given
        psm_df = pd.DataFrame(
            {"precursor_idx": [0, 1, 2], "pg": ["PG001", "PG002", "PG003"]}
        )

        frag_df = pd.DataFrame(
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

        # when
        result_df = prepare_df(frag_df, psm_df, column="intensity")

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


# Additional test cases to implement:
# test_accumulate_frag_df_with_missing_required_columns_should_raise_error
# test_accumulate_frag_df_with_duplicate_ion_hashes_should_handle_correctly
# test_accumulate_frag_df_should_preserve_data_types
# test_accumulate_frag_df_with_very_large_datasets_should_not_exceed_memory_limits
# test_accumulate_frag_df_should_handle_unicode_raw_names

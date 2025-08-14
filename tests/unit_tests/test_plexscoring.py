"""Tests for CandidateScoring.collect_candidates method."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from alphadia.plexscoring.plexscoring import CandidateScoring


class TestCandidateScoringCollectCandidates:
    """Test cases for CandidateScoring.collect_candidates method."""

    @pytest.fixture
    def minimal_precursors_flat_df(self) -> pd.DataFrame:
        """Create minimal precursors_flat DataFrame for testing."""
        return pd.DataFrame(
            {
                "elution_group_idx": np.array([0, 1], dtype=np.uint32),
                "precursor_idx": np.array([0, 1], dtype=np.uint32),
                "channel": np.array([0, 0], dtype=np.uint32),
                "decoy": np.array([0, 1], dtype=np.uint8),
                "flat_frag_start_idx": np.array([0, 5], dtype=np.uint32),
                "flat_frag_stop_idx": np.array([5, 10], dtype=np.uint32),
                "charge": np.array([2, 3], dtype=np.uint8),
                "rt_library": np.array([100.0, 200.0], dtype=np.float32),
                "mobility_library": np.array([0.8, 0.9], dtype=np.float32),
                "mz_library": np.array([500.0, 600.0], dtype=np.float32),
                "proteins": ["P1", "P2"],
                "genes": ["G1", "G2"],
                "sequence": ["PEPTIDEK", "ANOTHERR"],
                "mods": ["", ""],
                "mod_sites": ["", ""],
                "i_0": np.array([1.0, 1.0], dtype=np.float32),
            }
        )

    @pytest.fixture
    def minimal_fragments_flat_df(self) -> pd.DataFrame:
        """Create minimal fragments_flat DataFrame for testing."""
        return pd.DataFrame(
            {
                "mz_library": np.array([200.0, 300.0, 400.0], dtype=np.float32),
                "intensity": np.array([1000.0, 2000.0, 1500.0], dtype=np.float32),
                "cardinality": np.array([1, 1, 1], dtype=np.uint8),
                "type": np.array([0, 1, 0], dtype=np.uint8),  # b=0, y=1
                "loss_type": np.array([0, 0, 0], dtype=np.uint8),
                "charge": np.array([1, 1, 2], dtype=np.uint8),
                "number": np.array([1, 2, 3], dtype=np.uint8),
                "position": np.array([1, 2, 3], dtype=np.uint8),
            }
        )

    @pytest.fixture
    def mock_dia_data(self):
        """Create mock DiaData object."""
        mock_data = Mock()
        mock_data.cycle = Mock()
        return mock_data

    @pytest.fixture
    def mock_psm_proto_df(self):
        """Create mock PSM proto DataFrame with to_precursor_df method."""
        mock_psm = Mock()
        # Mock the to_precursor_df method to return test data
        precursor_idx = np.array([0, 1])
        rank = np.array([1, 1])
        # Create minimal feature data (46 columns as defined in collect_candidates)
        features = np.random.rand(2, 46).astype(np.float32)
        mock_psm.to_precursor_df.return_value = (precursor_idx, rank, features)
        return mock_psm

    @pytest.fixture
    def candidate_scoring(
        self, mock_dia_data, minimal_precursors_flat_df, minimal_fragments_flat_df
    ):
        """Create CandidateScoring instance for testing."""
        from alphadia.plexscoring.config import CandidateConfig
        from alphadia.plexscoring.quadrupole import SimpleQuadrupole

        # Create a mock quadrupole calibration to avoid cycle initialization issues
        mock_quadrupole = Mock(spec=SimpleQuadrupole)
        mock_quadrupole.jit = Mock()

        return CandidateScoring(
            dia_data=mock_dia_data,
            precursors_flat=minimal_precursors_flat_df,
            fragments_flat=minimal_fragments_flat_df,
            quadrupole_calibration=mock_quadrupole,
            config=CandidateConfig(),
        )

    def test_collect_candidates_should_return_dataframe_with_expected_columns(
        self, candidate_scoring, mock_psm_proto_df
    ) -> None:
        """Test that collect_candidates returns DataFrame with expected structure."""
        # given
        candidates_df = pd.DataFrame(
            {
                "precursor_idx": [0, 1],
                "rank": [1, 1],
                "elution_group_idx": [0, 1],
                "scan_center": [100, 200],
                "scan_start": [90, 190],
                "scan_stop": [110, 210],
                "frame_center": [50, 100],
                "frame_start": [45, 95],
                "frame_stop": [55, 105],
            }
        )

        # when
        result_df = candidate_scoring.collect_candidates(
            candidates_df, mock_psm_proto_df
        )

        # then
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "precursor_idx" in result_df.columns
        assert "rank" in result_df.columns
        assert "rt_observed" in result_df.columns
        assert "delta_rt" in result_df.columns
        assert "n_K" in result_df.columns
        assert "n_R" in result_df.columns
        assert "n_P" in result_df.columns

        # Check that amino acid counts are calculated correctly
        assert result_df.loc[0, "n_K"] == 1  # PEPTIDEK has 1 K
        assert result_df.loc[0, "n_R"] == 0  # PEPTIDEK has 0 R
        assert result_df.loc[1, "n_R"] == 2  # ANOTHERR has 2 R


# Additional test cases to consider next:
# - test_collect_candidates_with_calibrated_columns
# - test_collect_candidates_with_score_column
# - test_collect_candidates_merges_precursor_data_correctly
# - test_collect_candidates_calculates_delta_rt_with_calibrated_rt
# - test_collect_candidates_handles_missing_columns_gracefully

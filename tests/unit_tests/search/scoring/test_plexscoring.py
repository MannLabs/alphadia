"""Tests for CandidateScoring.collect_candidates method."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from alphadia.search.scoring.plexscoring import CandidateScoring


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
        # Create deterministic feature data (46 columns as defined in DEFAULT_FEATURE_COLUMNS)
        # fmt: off
        features = np.array(
            [
                [1.0, 2.0, 100.5, 0.85, 1000.0, 800.0, 1500.0, 900.0, 0.1, 0.05, 500.1, 900.0, 700.0, 1400.0, 850.0,
                 0.95, 0.90, 10.0, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 2000.0, 1800.0, 200.0, 0.15, 0.88, 0.82,
                 0.78, 0.92, 0.86, 0.84, 5.0, 0.89, 6.0, 20.0, 15.0, 2.5, 0.02, 0.01, 3.0, 500.0, 0.03],
                [1.5, 2.5, 200.5, 0.90, 1100.0, 850.0, 1600.0, 950.0, 0.15, 0.08, 600.1, 950.0, 750.0, 1450.0, 900.0,
                 0.96, 0.91, 12.0, 0.87, 0.82, 0.77, 0.72, 0.67, 0.62, 0.57, 2100.0, 1900.0, 200.0, 0.18, 0.90, 0.84,
                 0.80, 0.94, 0.88, 0.86, 6.0, 0.91, 7.0, 22.0, 17.0, 3.0, 0.025, 0.015, 4.0, 550.0, 0.035]
            ],
            dtype=np.float32,
        )
        # fmt: on
        mock_psm.to_precursor_df.return_value = (precursor_idx, rank, features)
        return mock_psm

    @pytest.fixture
    def candidate_scoring(
        self, mock_dia_data, minimal_precursors_flat_df, minimal_fragments_flat_df
    ):
        """Create CandidateScoring instance for testing."""
        from alphadia.search.scoring.config import CandidateConfig
        from alphadia.search.scoring.quadrupole import SimpleQuadrupole

        # Create a mock quadrupole calibration to avoid cycle initialization issues
        mock_quadrupole = Mock(spec=SimpleQuadrupole)
        mock_quadrupole.jit = Mock()

        return CandidateScoring(
            dia_data=mock_dia_data,
            precursors_flat=minimal_precursors_flat_df,
            fragments_flat=minimal_fragments_flat_df,
            quadrupole_calibration=mock_quadrupole,
            config=CandidateConfig(),
            rt_column="rt_library",
            mobility_column="mobility_library",
            precursor_mz_column="mz_library",
            fragment_mz_column="mz_library",
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

        # then - Create expected DataFrame with all expected values
        expected_df = pd.DataFrame(
            {
                # Feature columns (first 46 columns from DEFAULT_FEATURE_COLUMNS)
                "base_width_mobility": [1.0, 1.5],
                "base_width_rt": [2.0, 2.5],
                "rt_observed": [100.5, 200.5],
                "mobility_observed": [0.85, 0.90],
                "mono_ms1_intensity": [1000.0, 1100.0],
                "top_ms1_intensity": [800.0, 850.0],
                "sum_ms1_intensity": [1500.0, 1600.0],
                "weighted_ms1_intensity": [900.0, 950.0],
                "weighted_mass_deviation": [0.1, 0.15],
                "weighted_mass_error": [0.05, 0.08],
                "mz_observed": [500.1, 600.1],
                "mono_ms1_height": [900.0, 950.0],
                "top_ms1_height": [700.0, 750.0],
                "sum_ms1_height": [1400.0, 1450.0],
                "weighted_ms1_height": [850.0, 900.0],
                "isotope_intensity_correlation": [0.95, 0.96],
                "isotope_height_correlation": [0.90, 0.91],
                "n_observations": [10.0, 12.0],
                "intensity_correlation": [0.85, 0.87],
                "height_correlation": [0.80, 0.82],
                "intensity_fraction": [0.75, 0.77],
                "height_fraction": [0.70, 0.72],
                "intensity_fraction_weighted": [0.65, 0.67],
                "height_fraction_weighted": [0.60, 0.62],
                "mean_observation_score": [0.55, 0.57],
                "sum_b_ion_intensity": [2000.0, 2100.0],
                "sum_y_ion_intensity": [1800.0, 1900.0],
                "diff_b_y_ion_intensity": [200.0, 200.0],
                "f_masked": [0.15, 0.18],
                "fragment_scan_correlation": [0.88, 0.90],
                "template_scan_correlation": [0.82, 0.84],
                "fragment_frame_correlation": [0.78, 0.80],
                "top3_frame_correlation": [0.92, 0.94],
                "template_frame_correlation": [0.86, 0.88],
                "top3_b_ion_correlation": [0.84, 0.86],
                "n_b_ions": [5.0, 6.0],
                "top3_y_ion_correlation": [0.89, 0.91],
                "n_y_ions": [6.0, 7.0],
                "cycle_fwhm": [20.0, 22.0],
                "mobility_fwhm": [15.0, 17.0],
                "delta_frame_peak": [2.5, 3.0],
                "top_3_ms2_mass_error": [0.02, 0.025],
                "mean_ms2_mass_error": [0.01, 0.015],
                "n_overlapping": [3.0, 4.0],
                "mean_overlapping_intensity": [500.0, 550.0],
                "mean_overlapping_mass_error": [0.03, 0.035],
                # precursor_idx and rank added by method
                "precursor_idx": [0, 1],
                "rank": [1, 1],
                # From candidates_df (DEFAULT_CANDIDATE_COLUMNS order in merged result)
                "elution_group_idx": [0, 1],
                "frame_center": [50, 100],
                "frame_stop": [55, 105],
                "scan_stop": [110, 210],
                "frame_start": [45, 95],
                "scan_start": [90, 190],
                "scan_center": [100, 200],
                # From precursors_flat_df merged columns (in merge order)
                "flat_frag_stop_idx": [5, 10],
                "mod_sites": ["", ""],
                "mz_library": [500.0, 600.0],
                "decoy": [0, 1],
                "charge": [2, 3],
                "mods": ["", ""],
                "rt_library": [100.0, 200.0],
                "proteins": ["P1", "P2"],
                "channel": [0, 0],
                "genes": ["G1", "G2"],
                "flat_frag_start_idx": [0, 5],
                "mobility_library": [0.8, 0.9],
                "i_0": [1.0, 1.0],
                "sequence": ["PEPTIDEK", "ANOTHERR"],
                # Calculated columns
                "delta_rt": [0.5, 0.5],
                "n_K": [1, 0],
                "n_R": [0, 2],
                "n_P": [2, 0],
            }
        )

        # Reorder expected DataFrame to match result column order
        expected_df = expected_df[result_df.columns]

        pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)

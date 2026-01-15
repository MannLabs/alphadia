"""Test charge prediction functionality in PeptDeepPrediction."""

import pandas as pd
import pytest
from alphabase.spectral_library.base import SpecLibBase
from peptdeep.pretrained_models import ModelManager

from alphadia.libtransform.prediction import PeptDeepPrediction


class TestChargePrediction:
    """Test charge prediction with PeptDeep ModelManager."""

    @pytest.fixture
    def sample_precursor_df(self):
        """Create a sample precursor dataframe without charge column."""
        return pd.DataFrame(
            {
                "sequence": ["PEPTIDE", "ANOTHERSEQ", "TESTPEPTIDE"],
                "mods": ["", "", ""],
                "mod_sites": ["", "", ""],
            }
        )

    @pytest.fixture
    def sample_precursor_df_with_charge(self):
        """Create a sample precursor dataframe with charge column."""
        return pd.DataFrame(
            {
                "sequence": ["PEPTIDE", "ANOTHERSEQ", "TESTPEPTIDE"],
                "mods": ["", "", ""],
                "mod_sites": ["", "", ""],
                "charge": [2, 3, 2],
            }
        )

    def test_predict_charge_expands_dataframe(self, sample_precursor_df):
        """Test that predict_charge expands dataframe with charge states."""
        model_mgr = ModelManager(device="cpu")

        result_df = model_mgr.predict_charge(
            sample_precursor_df,
            min_precursor_charge=2,
            max_precursor_charge=4,
            charge_prob_cutoff=0.3,
        )

        assert "charge" in result_df.columns
        assert len(result_df) >= len(sample_precursor_df)

    def test_predict_charge_with_low_cutoff(self, sample_precursor_df):
        """Test that low cutoff keeps more charge states."""
        model_mgr = ModelManager(device="cpu")

        result_high_cutoff = model_mgr.predict_charge(
            sample_precursor_df,
            min_precursor_charge=2,
            max_precursor_charge=4,
            charge_prob_cutoff=0.5,
        )

        result_low_cutoff = model_mgr.predict_charge(
            sample_precursor_df,
            min_precursor_charge=2,
            max_precursor_charge=4,
            charge_prob_cutoff=0.1,
        )

        assert len(result_low_cutoff) >= len(result_high_cutoff)

    def test_predict_all_after_charge_prediction(self, sample_precursor_df):
        """Test full workflow: predict_charge then predict_all."""
        model_mgr = ModelManager(device="cpu")

        precursor_df_with_charges = model_mgr.predict_charge(
            sample_precursor_df,
            min_precursor_charge=2,
            max_precursor_charge=4,
            charge_prob_cutoff=0.3,
        )

        result = model_mgr.predict_all(
            precursor_df_with_charges,
            predict_items=["rt", "mobility", "ms2"],
            frag_types=["b_z1", "b_z2", "y_z1", "y_z2"],
        )

        assert "precursor_df" in result
        assert "fragment_mz_df" in result
        assert "fragment_intensity_df" in result


class TestPeptDeepPredictionWithCharge:
    """Test PeptDeepPrediction class with charge prediction."""

    @pytest.fixture
    def sample_speclib(self):
        """Create a sample spectral library without charge column."""
        speclib = SpecLibBase()
        speclib._precursor_df = pd.DataFrame(
            {
                "sequence": ["PEPTIDE", "ANOTHERSEQ", "TESTPEPTIDE"],
                "mods": ["", "", ""],
                "mod_sites": ["", "", ""],
            }
        )
        return speclib

    def test_peptdeep_prediction_with_charge(self, sample_speclib):
        """Test PeptDeepPrediction with charge prediction enabled."""
        predictor = PeptDeepPrediction(
            use_gpu=False,
            predict_charge=True,
            min_charge_probability=0.3,
        )

        result = predictor(sample_speclib)

        assert "charge" in result.precursor_df.columns
        assert len(result.precursor_df) >= 3

    def test_peptdeep_prediction_without_charge(self, sample_speclib):
        """Test PeptDeepPrediction without charge prediction (default)."""
        sample_speclib._precursor_df["charge"] = [2, 3, 2]

        predictor = PeptDeepPrediction(
            use_gpu=False,
            predict_charge=False,
        )

        result = predictor(sample_speclib)

        assert "charge" in result.precursor_df.columns
        assert len(result.precursor_df) == 3

    def test_charge_prediction_filters_by_input_charge_range(self):
        """Test that charge prediction respects input library charge range.

        When the input library has charges 2-3, the predicted charges
        should be filtered to only include charges within that range,
        even though the model supports charges 1-10.
        """
        speclib = SpecLibBase()
        speclib._precursor_df = pd.DataFrame(
            {
                "sequence": ["PEPTIDE", "PEPTIDE", "ANOTHERSEQ", "ANOTHERSEQ"],
                "mods": ["", "", "", ""],
                "mod_sites": ["", "", "", ""],
                "charge": [2, 3, 2, 3],
            }
        )

        predictor = PeptDeepPrediction(
            use_gpu=False,
            predict_charge=True,
            min_charge_probability=0.01,
        )

        result = predictor(speclib)

        assert "charge" in result.precursor_df.columns
        assert "charge_prob" in result.precursor_df.columns

        result_charges = result.precursor_df["charge"].unique()
        assert all(
            2 <= c <= 3 for c in result_charges
        ), f"Expected charges in range 2-3, got {result_charges}"

    def test_charge_prediction_uses_full_range_without_input_charges(self):
        """Test that charge prediction uses full model range when no input charges.

        When the input library has no charge column, the model should
        predict using its full supported range (1-10).
        """
        speclib = SpecLibBase()
        speclib._precursor_df = pd.DataFrame(
            {
                "sequence": ["PEPTIDE", "ANOTHERSEQ"],
                "mods": ["", ""],
                "mod_sites": ["", ""],
            }
        )

        predictor = PeptDeepPrediction(
            use_gpu=False,
            predict_charge=True,
            min_charge_probability=0.3,
        )

        result = predictor(speclib)

        assert "charge" in result.precursor_df.columns
        assert "charge_prob" in result.precursor_df.columns
        assert len(result.precursor_df) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

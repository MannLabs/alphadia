import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform.quantification.intensity_drift_correction import (
    correct_intensity_drift,
)

RUNS = ["A1", "A2", "A3", "B1", "B2", "B3"]
DRIFTING_RUN = "A1"
GRADIENT_SECONDS = 1500.0
N_PRECURSORS = 5000
FRAGMENTS_PER_PRECURSOR = 3
# log2 drift of DRIFTING_RUN at the end of the gradient
DRIFT_LOG2 = 0.3
ION_NOISE_LOG2 = 0.1


def _make_tables(
    n_precursors: int = N_PRECURSORS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic fragment intensity, correlation and psm tables.

    Precursor RTs are uniform in [0, GRADIENT_SECONDS], the RT is identical in every run.
    The intensities of DRIFTING_RUN are multiplied by 2 ** (DRIFT_LOG2 * rt / GRADIENT_SECONDS).
    """
    rng = np.random.default_rng(0)
    precursor_idx = np.repeat(np.arange(n_precursors), FRAGMENTS_PER_PRECURSOR)
    n_ions = len(precursor_idx)
    rt = rng.uniform(0, GRADIENT_SECONDS, n_precursors)

    base = 2 ** rng.uniform(10, 20, n_ions)
    intensity = {run: base * 2 ** rng.normal(0, ION_NOISE_LOG2, n_ions) for run in RUNS}
    intensity[DRIFTING_RUN] *= 2 ** (DRIFT_LOG2 * rt[precursor_idx] / GRADIENT_SECONDS)

    metadata = {
        "precursor_idx": precursor_idx,
        "ion": np.arange(n_ions),
        "pg": [f"PG{i}" for i in precursor_idx],
        "mod_seq_hash": precursor_idx,
        "mod_seq_charge_hash": precursor_idx,
    }
    intensity_df = pd.DataFrame({**metadata, **intensity})
    correlation_df = pd.DataFrame({**metadata, **{run: 0.95 for run in RUNS}})

    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.tile(np.arange(n_precursors), len(RUNS)),
            "run": np.repeat(RUNS, n_precursors),
            "rt_observed": np.tile(rt, len(RUNS)),
        }
    )
    return intensity_df, correlation_df, psm_df


def _corrected(
    intensity_df: pd.DataFrame, correlation_df: pd.DataFrame, psm_df: pd.DataFrame
) -> pd.DataFrame:
    """Return a corrected copy of intensity_df."""
    result_df = intensity_df.copy()
    correct_intensity_drift(result_df, correlation_df, psm_df)
    return result_df


class TestCorrectIntensityDrift:
    def test_corrects_intensities_in_place_and_leaves_correlations(self):
        """Given a drifting run, when corrected, then the intensity table is corrected in place and the correlation table is untouched."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables()
        intensity_copy = intensity_df.copy()
        correlation_copy = correlation_df.copy()

        # When
        correct_intensity_drift(intensity_df, correlation_df, psm_df)

        # Then
        assert not intensity_df[DRIFTING_RUN].equals(intensity_copy[DRIFTING_RUN])
        pd.testing.assert_frame_equal(correlation_df, correlation_copy)

    @pytest.mark.parametrize("missing_column", ["rt_observed", "run"])
    def test_raises_when_psm_columns_are_missing(self, missing_column):
        """Given a psm table without rt_observed or run, when corrected, then a ValueError names the missing column."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables(n_precursors=10)
        psm_df = psm_df.drop(columns=[missing_column])

        # When / Then
        with pytest.raises(ValueError, match=missing_column):
            correct_intensity_drift(intensity_df, correlation_df, psm_df)

    def test_matches_ions_to_precursors_independent_of_row_order(self):
        """Given shuffled rows and sparse precursor indices, when corrected, then every ion gets the same correction as in the ordered table."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables()
        offset = 1_000_000
        for df in (intensity_df, correlation_df, psm_df):
            df["precursor_idx"] = df["precursor_idx"] * 7 + offset
        order = np.random.default_rng(1).permutation(len(intensity_df))
        shuffled_intensity_df = intensity_df.iloc[order].reset_index(drop=True)
        shuffled_correlation_df = correlation_df.iloc[order].reset_index(drop=True)

        # When
        result_df = _corrected(intensity_df, correlation_df, psm_df)
        shuffled_result_df = _corrected(
            shuffled_intensity_df, shuffled_correlation_df, psm_df
        )

        # Then
        np.testing.assert_allclose(
            shuffled_result_df[RUNS].to_numpy(), result_df[RUNS].to_numpy()[order]
        )

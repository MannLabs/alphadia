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
# log2 drift of DRIFTING_RUN at the end of the gradient
DRIFT_LOG2 = 0.3
PRECURSOR_NOISE_LOG2 = 0.05
# the drift left after the correction must be well below DRIFT_LOG2
MAX_RESIDUAL_DRIFT_LOG2 = 0.1


def _make_tables(
    n_precursors: int = N_PRECURSORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic variance-model precursor table and psm table.

    Precursor RTs are uniform in [0, GRADIENT_SECONDS], the RT is identical in every run.
    The intensities of DRIFTING_RUN are multiplied by 2 ** (DRIFT_LOG2 * rt / GRADIENT_SECONDS).
    The psm table has precursor_idx values that differ from the precursor hashes.
    """
    rng = np.random.default_rng(0)
    precursor_hash = np.arange(n_precursors)
    rt = rng.uniform(0, GRADIENT_SECONDS, n_precursors)

    base = 2 ** rng.uniform(10, 20, n_precursors)
    intensity_by_run = {
        run: base * 2 ** rng.normal(0, PRECURSOR_NOISE_LOG2, n_precursors)
        for run in RUNS
    }
    intensity_by_run[DRIFTING_RUN] *= 2 ** (DRIFT_LOG2 * rt / GRADIENT_SECONDS)

    precursor_df = pd.DataFrame(
        {
            "ion": precursor_hash,
            "mod_seq_charge_hash": precursor_hash,
            "pg": [f"PG{i}" for i in precursor_hash],
            "mod_seq_hash": precursor_hash,
            **intensity_by_run,
        }
    )
    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_precursors * len(RUNS)),
            "mod_seq_charge_hash": np.tile(precursor_hash, len(RUNS)),
            "run": np.repeat(RUNS, n_precursors),
            "rt_observed": np.tile(rt, len(RUNS)),
        }
    )
    return precursor_df, psm_df


def _log2_drift_range(df: pd.DataFrame, rt: np.ndarray) -> float:
    """Difference of the mean log2 ratio DRIFTING_RUN / A2 between the last and the first RT quarter."""
    log2_ratio = np.log2(df[DRIFTING_RUN].to_numpy() / df["A2"].to_numpy())
    first_quarter = rt < GRADIENT_SECONDS / 4
    last_quarter = rt > GRADIENT_SECONDS * 3 / 4
    return log2_ratio[last_quarter].mean() - log2_ratio[first_quarter].mean()


class TestCorrectIntensityDrift:
    def test_removes_the_drift_of_the_drifting_run(self):
        """Given a run whose intensity rises along the gradient, when corrected, then the rise is removed."""
        # Given
        precursor_df, psm_df = _make_tables()
        rt = psm_df.loc[psm_df["run"] == DRIFTING_RUN, "rt_observed"].to_numpy()

        # When
        result_df = correct_intensity_drift(precursor_df, psm_df)

        # Then
        assert _log2_drift_range(precursor_df, rt) > DRIFT_LOG2 / 2
        assert abs(_log2_drift_range(result_df, rt)) < MAX_RESIDUAL_DRIFT_LOG2

    def test_returns_a_corrected_float32_copy_and_leaves_the_inputs(self):
        """Given a drifting run, when corrected, then a corrected float32 copy is returned and the input tables are untouched."""
        # Given
        precursor_df, psm_df = _make_tables()
        precursor_copy = precursor_df.copy()
        psm_copy = psm_df.copy()

        # When
        result_df = correct_intensity_drift(precursor_df, psm_df)

        # Then
        assert not result_df[DRIFTING_RUN].equals(precursor_df[DRIFTING_RUN])
        assert (result_df[RUNS].dtypes == np.float32).all()
        pd.testing.assert_frame_equal(
            result_df.drop(columns=RUNS), precursor_df.drop(columns=RUNS)
        )
        pd.testing.assert_frame_equal(precursor_df, precursor_copy)
        pd.testing.assert_frame_equal(psm_df, psm_copy)

    @pytest.mark.parametrize(
        "missing_column", ["rt_observed", "run", "mod_seq_charge_hash"]
    )
    def test_raises_if_a_psm_column_is_missing(self, missing_column):
        """Given a psm table without one of the required columns, when corrected, then a ValueError names the missing column."""
        # Given
        precursor_df, psm_df = _make_tables(n_precursors=10)
        psm_df = psm_df.drop(columns=[missing_column])

        # When / Then
        with pytest.raises(ValueError, match=missing_column):
            correct_intensity_drift(precursor_df, psm_df)

    def test_looks_up_the_rt_by_precursor_hash_independent_of_row_order(self):
        """Given shuffled rows and sparse precursor hashes, when corrected, then every precursor gets the same correction as in the ordered table."""
        # Given
        precursor_df, psm_df = _make_tables()
        offset = 1_000_000
        precursor_df["mod_seq_charge_hash"] = (
            precursor_df["mod_seq_charge_hash"] * 7 + offset
        )
        precursor_df["ion"] = precursor_df["mod_seq_charge_hash"]
        psm_df["mod_seq_charge_hash"] = psm_df["mod_seq_charge_hash"] * 7 + offset
        order = np.random.default_rng(1).permutation(len(precursor_df))
        shuffled_precursor_df = precursor_df.iloc[order].reset_index(drop=True)
        shuffled_psm_df = psm_df.sample(frac=1, random_state=2)

        # When
        result_df = correct_intensity_drift(precursor_df, psm_df)
        shuffled_result_df = correct_intensity_drift(
            shuffled_precursor_df, shuffled_psm_df
        )

        # Then
        np.testing.assert_allclose(
            shuffled_result_df[RUNS].to_numpy(), result_df[RUNS].to_numpy()[order]
        )

    def test_uses_one_psm_per_precursor_and_run(self):
        """Given a later second psm of the same precursor and run, when corrected, then the result equals the one without it."""
        # Given
        precursor_df, psm_df = _make_tables()
        duplicate_psm_df = pd.concat(
            [psm_df, psm_df.assign(rt_observed=psm_df["rt_observed"] + 100.0)],
            ignore_index=True,
        )

        # When
        result_df = correct_intensity_drift(precursor_df, psm_df)
        duplicate_result_df = correct_intensity_drift(precursor_df, duplicate_psm_df)

        # Then
        pd.testing.assert_frame_equal(duplicate_result_df, result_df)

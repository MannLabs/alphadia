import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform.quantification.intensity_drift_correction import (
    correct_intensity_drift,
    correct_precursor_intensity_drift,
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
    intensity_by_run = {
        run: base * 2 ** rng.normal(0, ION_NOISE_LOG2, n_ions) for run in RUNS
    }
    intensity_by_run[DRIFTING_RUN] *= 2 ** (
        DRIFT_LOG2 * rt[precursor_idx] / GRADIENT_SECONDS
    )
    # the accumulated fragment tables hold float32
    intensity_by_run = {
        run: values.astype(np.float32) for run, values in intensity_by_run.items()
    }

    metadata = {
        "precursor_idx": precursor_idx,
        "ion": np.arange(n_ions),
        "pg": [f"PG{i}" for i in precursor_idx],
        "mod_seq_hash": precursor_idx,
        "mod_seq_charge_hash": precursor_idx,
    }
    intensity_df = pd.DataFrame({**metadata, **intensity_by_run})
    correlation_df = pd.DataFrame(
        {**metadata, **{run: np.float32(0.95) for run in RUNS}}
    )

    psm_df = pd.DataFrame(
        {
            "precursor_idx": np.tile(np.arange(n_precursors), len(RUNS)),
            "run": np.repeat(RUNS, n_precursors),
            "rt_observed": np.tile(rt, len(RUNS)),
        }
    )
    return intensity_df, correlation_df, psm_df


class TestCorrectIntensityDrift:
    def test_returns_a_corrected_float32_copy_and_leaves_the_inputs(self):
        """Given a drifting run, when corrected, then a corrected float32 copy is returned and the input tables are untouched."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables()
        intensity_copy = intensity_df.copy()
        correlation_copy = correlation_df.copy()

        # When
        result_df = correct_intensity_drift(intensity_df, correlation_df, psm_df)

        # Then
        assert not result_df[DRIFTING_RUN].equals(intensity_df[DRIFTING_RUN])
        assert (result_df[RUNS].dtypes == np.float32).all()
        pd.testing.assert_frame_equal(
            result_df.drop(columns=RUNS), intensity_df.drop(columns=RUNS)
        )
        pd.testing.assert_frame_equal(intensity_df, intensity_copy)
        pd.testing.assert_frame_equal(correlation_df, correlation_copy)

    @pytest.mark.parametrize("missing_column", ["rt_observed", "run"])
    def test_raises_if_a_psm_column_is_missing(self, missing_column):
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
        result_df = correct_intensity_drift(intensity_df, correlation_df, psm_df)
        shuffled_result_df = correct_intensity_drift(
            shuffled_intensity_df, shuffled_correlation_df, psm_df
        )

        # Then
        np.testing.assert_allclose(
            shuffled_result_df[RUNS].to_numpy(), result_df[RUNS].to_numpy()[order]
        )


PRECURSOR_RUNS = ["A1", "A2", "A3", "B1", "B2", "B3"]
N_DRIFT_PRECURSORS = 20000
MASS_RANGE = (800.0, 4000.0)
# log2 drift between the A and B runs at the end of the gradient for the heaviest precursors
PRECURSOR_DRIFT_LOG2 = 0.8
PRECURSOR_NOISE_LOG2 = 0.2
LATE_SHARE = 0.75


def _make_precursor_tables(
    n_precursors: int = N_DRIFT_PRECURSORS, runs: list[str] = PRECURSOR_RUNS
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Synthetic varmodel precursor table, psm table and the log2 drift of every cell (runs x precursors).

    The A runs drift up, the B runs down, in proportion to RT / GRADIENT_SECONDS times the relative mass of the
    precursor: a drift confounded with the condition.
    """
    rng = np.random.default_rng(0)
    rt = rng.uniform(0, GRADIENT_SECONDS, n_precursors)
    mz = rng.uniform(400.0, 1000.0, n_precursors)
    charge = rng.integers(2, 4, n_precursors)
    mass = mz * charge
    relative_mass = (mass - MASS_RANGE[0]) / (MASS_RANGE[1] - MASS_RANGE[0])
    level = rng.uniform(10, 20, n_precursors)
    sign = np.array([0.5 if run.startswith("A") else -0.5 for run in runs])
    drift = (
        sign[:, None]
        * PRECURSOR_DRIFT_LOG2
        * (rt / GRADIENT_SECONDS * relative_mass)[None, :]
    )
    log2 = (
        level[None, :]
        + drift
        + rng.normal(0, PRECURSOR_NOISE_LOG2, (len(runs), n_precursors))
    )
    precursor_hash = np.arange(n_precursors) * 7 + 1_000_000
    precursor_df = pd.DataFrame(
        {
            "ion": precursor_hash,
            "mod_seq_charge_hash": precursor_hash,
            "pg": [f"PG{i}" for i in range(n_precursors)],
            "mod_seq_hash": precursor_hash,
            **{run: np.exp2(log2[i]) for i, run in enumerate(runs)},
        }
    )
    psm_df = pd.DataFrame(
        {
            "mod_seq_charge_hash": np.tile(precursor_hash, len(runs)),
            "run": np.repeat(runs, n_precursors),
            "rt_observed": np.tile(rt, len(runs)),
            "mz_library": np.tile(mz, len(runs)),
            "charge": np.tile(charge, len(runs)),
        }
    )
    return precursor_df, psm_df, drift


def _late_heavy_a_vs_b(
    precursor_df: pd.DataFrame, psm_df: pd.DataFrame, runs: list[str]
) -> float:
    """Mean log2 A minus B difference of the late, heavy precursors relative to all precursors."""
    first_run = psm_df[psm_df["run"] == runs[0]]
    mass = (first_run["mz_library"] * first_run["charge"]).to_numpy()
    late_heavy = (
        first_run["rt_observed"].to_numpy() > LATE_SHARE * GRADIENT_SECONDS
    ) & (mass > np.quantile(mass, LATE_SHARE))
    log2 = np.log2(precursor_df[runs].to_numpy())
    a_minus_b = log2[:, [r.startswith("A") for r in runs]].mean(axis=1) - log2[
        :, [r.startswith("B") for r in runs]
    ].mean(axis=1)
    return a_minus_b[late_heavy].mean() - a_minus_b.mean()


class TestCorrectPrecursorIntensityDrift:
    def test_removes_the_condition_confounded_drift_along_rt_and_mass(self):
        """Given A and B runs drifting apart late in the gradient for heavy precursors, when corrected, then most of the late heavy A vs B difference is gone."""
        # Given
        precursor_df, psm_df, _ = _make_precursor_tables()
        before = _late_heavy_a_vs_b(precursor_df, psm_df, PRECURSOR_RUNS)

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        after = _late_heavy_a_vs_b(result_df, psm_df, PRECURSOR_RUNS)
        assert before > 0.2
        assert abs(after) < 0.1 * before

    def test_returns_a_corrected_copy_and_leaves_the_inputs(self):
        """Given a precursor table, when corrected, then a copy with changed run columns and unchanged metadata is returned and the inputs are untouched."""
        # Given
        precursor_df, psm_df, _ = _make_precursor_tables()
        precursor_copy = precursor_df.copy()
        psm_copy = psm_df.copy()

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        assert not result_df["A1"].equals(precursor_df["A1"])
        pd.testing.assert_frame_equal(
            result_df.drop(columns=PRECURSOR_RUNS),
            precursor_df.drop(columns=PRECURSOR_RUNS),
        )
        pd.testing.assert_frame_equal(precursor_df, precursor_copy)
        pd.testing.assert_frame_equal(psm_df, psm_copy)

    def test_keeps_missing_values_missing(self):
        """Given missing quantities, when corrected, then they stay 0."""
        # Given
        precursor_df, psm_df, _ = _make_precursor_tables()
        precursor_df.loc[::5, "B2"] = 0.0

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        assert (result_df.loc[::5, "B2"] == 0.0).all()
        assert (result_df.drop(index=result_df.index[::5])["B2"] > 0).all()

    def test_leaves_a_single_run_unchanged(self):
        """Given a single run, when corrected, then its quantities are unchanged."""
        # Given
        precursor_df, psm_df, _ = _make_precursor_tables(runs=["A1"])

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        np.testing.assert_array_equal(result_df["A1"], precursor_df["A1"])

    def test_matches_psms_to_precursors_independent_of_row_order(self):
        """Given shuffled precursor rows, when corrected, then every precursor gets the same correction as in the ordered table."""
        # Given
        precursor_df, psm_df, _ = _make_precursor_tables()
        order = np.random.default_rng(1).permutation(len(precursor_df))
        shuffled_df = precursor_df.iloc[order].reset_index(drop=True)

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)
        shuffled_result_df = correct_precursor_intensity_drift(shuffled_df, psm_df)

        # Then
        np.testing.assert_allclose(
            shuffled_result_df[PRECURSOR_RUNS].to_numpy(),
            result_df[PRECURSOR_RUNS].to_numpy()[order],
        )

    @pytest.mark.parametrize(
        "missing_column", ["rt_observed", "run", "mz_library", "charge"]
    )
    def test_raises_if_a_psm_column_is_missing(self, missing_column):
        """Given a psm table without a required column, when corrected, then a ValueError names the missing column."""
        # Given
        precursor_df, psm_df, _ = _make_precursor_tables(n_precursors=10)
        psm_df = psm_df.drop(columns=[missing_column])

        # When / Then
        with pytest.raises(ValueError, match=missing_column):
            correct_precursor_intensity_drift(precursor_df, psm_df)

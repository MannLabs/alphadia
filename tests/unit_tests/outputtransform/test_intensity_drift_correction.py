import logging

import numpy as np
import pandas as pd
import pytest
from alphabase.constants.atom import MASS_PROTON

from alphadia.outputtransform.quantification.intensity_drift_correction import (
    _neutral_precursor_mass,
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


N_DRIFT_PRECURSORS = 20000
# neutral precursor masses of the synthetic precursors: m/z 400-1000 at charge 2-4
MASS_RANGE = (800.0, 4000.0)
MZ_RANGE = (400.0, 1000.0)
CHARGE_RANGE = (2, 5)
# log2 drift between the A and B runs at the end of the gradient for the heaviest precursors
PRECURSOR_DRIFT_LOG2 = 0.8
PRECURSOR_NOISE_LOG2 = 0.2
LATE_SHARE = 0.75
# quantified precursors of a near-blank run, fewer than the reference precursors a run needs
N_BLANK_RUN_PRECURSORS = 50
BLANK_RUN = "B3"
# the other runs get the correction they get without the near-blank run, up to this share of the mean absolute correction
MAX_RELATIVE_CORRECTION_DIFFERENCE = 0.1


def _make_precursor_tables(
    n_precursors: int = N_DRIFT_PRECURSORS, runs: list[str] = RUNS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic varmodel precursor table and psm table.

    The A runs drift up, the B runs down, in proportion to RT / GRADIENT_SECONDS times the relative neutral mass of
    the precursor: a drift confounded with the condition. The RT of a precursor is the same in every run.
    """
    rng = np.random.default_rng(0)
    rt = rng.uniform(0, GRADIENT_SECONDS, n_precursors)
    mz = rng.uniform(*MZ_RANGE, n_precursors)
    charge = rng.integers(*CHARGE_RANGE, n_precursors)
    relative_mass = (_neutral_mass(mz, charge) - MASS_RANGE[0]) / (
        MASS_RANGE[1] - MASS_RANGE[0]
    )
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
    return precursor_df, psm_df


def _neutral_mass(mz: np.ndarray, charge: np.ndarray) -> np.ndarray:
    return (mz - MASS_PROTON) * charge


def _late_heavy_a_vs_b(
    precursor_df: pd.DataFrame, psm_df: pd.DataFrame, runs: list[str]
) -> float:
    """Mean log2 A minus B difference of the late, heavy precursors relative to all precursors."""
    first_run = psm_df[psm_df["run"] == runs[0]]
    mass = _neutral_mass(first_run["mz_library"], first_run["charge"]).to_numpy()
    late_heavy = (
        first_run["rt_observed"].to_numpy() > LATE_SHARE * GRADIENT_SECONDS
    ) & (mass > np.quantile(mass, LATE_SHARE))
    log2 = np.log2(precursor_df[runs].to_numpy())
    a_minus_b = np.nanmean(
        log2[:, [r.startswith("A") for r in runs]], axis=1
    ) - np.nanmean(log2[:, [r.startswith("B") for r in runs]], axis=1)
    return a_minus_b[late_heavy].mean() - a_minus_b.mean()


class TestCorrectPrecursorIntensityDrift:
    def test_removes_the_condition_confounded_drift_along_rt_and_mass(self):
        """Given A and B runs drifting apart late in the gradient for heavy precursors, when corrected, then most of the late heavy A vs B difference is gone."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        before = _late_heavy_a_vs_b(precursor_df, psm_df, RUNS)

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        after = _late_heavy_a_vs_b(result_df, psm_df, RUNS)
        assert before > 0.2
        assert abs(after) < 0.1 * before

    def test_returns_a_corrected_float64_copy_and_leaves_the_inputs(self):
        """Given a precursor table, when corrected, then a float64 copy with changed run columns and unchanged metadata is returned and the inputs are untouched."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        precursor_df[RUNS] = precursor_df[RUNS].astype(np.float32)
        precursor_copy = precursor_df.copy()
        psm_copy = psm_df.copy()

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        assert not result_df["A1"].equals(precursor_df["A1"])
        assert (result_df[RUNS].dtypes == np.float64).all()
        pd.testing.assert_frame_equal(
            result_df.drop(columns=RUNS), precursor_df.drop(columns=RUNS)
        )
        pd.testing.assert_frame_equal(precursor_df, precursor_copy)
        pd.testing.assert_frame_equal(psm_df, psm_copy)

    @pytest.mark.parametrize("missing_value", [0.0, np.nan])
    def test_keeps_missing_values_missing(self, missing_value):
        """Given quantities missing as 0 or NaN, when corrected, then they stay missing and the others are corrected."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        precursor_df.loc[::5, "B2"] = missing_value

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        np.testing.assert_array_equal(result_df.loc[::5, "B2"], missing_value)
        quantified = result_df.drop(index=result_df.index[::5])["B2"]
        assert (quantified > 0).all()
        assert not np.allclose(quantified, precursor_df["B2"].loc[quantified.index])

    def test_places_a_precursor_without_a_psm_in_a_run_at_its_mean_rt(self):
        """Given quantities without a psm in their run, when corrected, then they are corrected as at the RT of their other runs."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        without_psm = precursor_df["mod_seq_charge_hash"].iloc[::5]
        psm_without_df = psm_df[
            ~((psm_df["run"] == "B2") & psm_df["mod_seq_charge_hash"].isin(without_psm))
        ]

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)
        result_without_df = correct_precursor_intensity_drift(
            precursor_df, psm_without_df
        )

        # Then the RT of the synthetic precursors is the same in every run
        np.testing.assert_allclose(
            result_without_df[RUNS].to_numpy(), result_df[RUNS].to_numpy(), rtol=1e-9
        )

    def test_uses_the_first_psm_of_a_precursor_in_a_run(self):
        """Given a second psm of the same precursor and run with another RT and m/z, when corrected, then the first psm decides."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        duplicate_df = psm_df.iloc[::3].copy()
        duplicate_df["rt_observed"] = GRADIENT_SECONDS - duplicate_df["rt_observed"]
        duplicate_df["mz_library"] = duplicate_df["mz_library"] + 100.0
        psm_with_duplicates_df = pd.concat([psm_df, duplicate_df], ignore_index=True)

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)
        result_with_duplicates_df = correct_precursor_intensity_drift(
            precursor_df, psm_with_duplicates_df
        )

        # Then
        pd.testing.assert_frame_equal(result_with_duplicates_df, result_df)

    def test_neutral_precursor_mass_removes_the_proton_per_charge(self):
        """Given psms with library m/z and charge, when the precursor mass is computed, then it is the neutral mass and NaN for a precursor without a psm."""
        # Given
        psm_df = pd.DataFrame(
            {
                "mod_seq_charge_hash": [10, 20, 10],
                "mz_library": [500.0, 400.0, 500.0],
                "charge": [2, 3, 2],
            }
        )

        # When
        mass = _neutral_precursor_mass(psm_df, np.array([20, 10, 30]))

        # Then
        np.testing.assert_allclose(
            mass[:2], [(400.0 - MASS_PROTON) * 3, (500.0 - MASS_PROTON) * 2]
        )
        assert np.isnan(mass[2])

    def test_leaves_a_near_blank_run_uncorrected_and_warns(self, caplog):
        """Given a run with very few quantified precursors, when corrected, then that run is unchanged with a warning naming it, and the other runs are corrected as without it."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        precursor_df.loc[N_BLANK_RUN_PRECURSORS:, BLANK_RUN] = np.nan
        other_runs = [run for run in RUNS if run != BLANK_RUN]
        without_blank_df = precursor_df.drop(columns=[BLANK_RUN])

        # When
        with caplog.at_level(logging.WARNING):
            result_df = correct_precursor_intensity_drift(precursor_df, psm_df)
        without_blank_result_df = correct_precursor_intensity_drift(
            without_blank_df, psm_df[psm_df["run"] != BLANK_RUN]
        )

        # Then
        np.testing.assert_array_equal(result_df[BLANK_RUN], precursor_df[BLANK_RUN])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert BLANK_RUN in warnings[0].getMessage()
        assert not any(run in warnings[0].getMessage() for run in other_runs)
        correction = np.log2(precursor_df[other_runs] / result_df[other_runs])
        without_blank_correction = np.log2(
            without_blank_df[other_runs] / without_blank_result_df[other_runs]
        )
        difference = np.abs(correction - without_blank_correction).to_numpy().mean()
        assert (
            difference
            < MAX_RELATIVE_CORRECTION_DIFFERENCE
            * np.abs(without_blank_correction).to_numpy().mean()
        )

    @pytest.mark.parametrize(
        ("blank_run", "reference_runs"), [(False, len(RUNS)), (True, len(RUNS) - 1)]
    )
    def test_logs_the_runs_the_reference_precursors_are_quantified_in(
        self, caplog, blank_run, reference_runs
    ):
        """Given complete runs or one near-blank run, when corrected, then the summary names the runs the reference precursors are quantified in: all, or all but the near-blank one."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        if blank_run:
            precursor_df.loc[N_BLANK_RUN_PRECURSORS:, BLANK_RUN] = np.nan

        # When
        with caplog.at_level(logging.INFO):
            correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        summaries = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
        assert len(summaries) == 1
        assert (
            f"reference: precursors quantified in at least {reference_runs} of {len(RUNS)} runs"
            in summaries[0]
        )

    def test_leaves_a_single_run_unchanged_without_a_warning(self, caplog):
        """Given a single run, when corrected, then its quantities are unchanged and nothing is warned."""
        # Given
        precursor_df, psm_df = _make_precursor_tables(runs=["A1"])

        # When
        with caplog.at_level(logging.WARNING):
            result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        np.testing.assert_array_equal(result_df["A1"], precursor_df["A1"])
        assert not caplog.records

    def test_matches_psms_to_precursors_independent_of_row_order(self):
        """Given shuffled precursor rows, when corrected, then every precursor gets the same correction as in the ordered table."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        order = np.random.default_rng(1).permutation(len(precursor_df))
        shuffled_df = precursor_df.iloc[order].reset_index(drop=True)

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)
        shuffled_result_df = correct_precursor_intensity_drift(shuffled_df, psm_df)

        # Then
        np.testing.assert_allclose(
            shuffled_result_df[RUNS].to_numpy(),
            result_df[RUNS].to_numpy()[order],
        )

    @pytest.mark.parametrize(
        "missing_column", ["rt_observed", "run", "mz_library", "charge"]
    )
    def test_raises_if_a_psm_column_is_missing(self, missing_column):
        """Given a psm table without a required column, when corrected, then a ValueError names the missing column."""
        # Given
        precursor_df, psm_df = _make_precursor_tables(n_precursors=10)
        psm_df = psm_df.drop(columns=[missing_column])

        # When / Then
        with pytest.raises(ValueError, match=missing_column):
            correct_precursor_intensity_drift(precursor_df, psm_df)

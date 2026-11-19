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


DRIFT_MODULE = "alphadia.outputtransform.quantification.intensity_drift_correction"
PRECURSOR_RUNS = ["A1", "B1", "B2"]
# precursor keys in table order, which differs from the psm order
PRECURSOR_KEYS = [30, 10, 20]
# per precursor key: library m/z, charge and the observed RT in every run
MZ_BY_KEY = {10: 500.0, 20: 400.0, 30: 700.0}
CHARGE_BY_KEY = {10: 2, 20: 3, 30: 2}
RT_BY_KEY = {10: 100.0, 20: 200.0, 30: 300.0}


class _RecordingDriftCorrection:
    """Stands in for PrecursorDriftCorrection: records its inputs, halves the intensities of the runs `summary` marks as corrected and returns `summary`."""

    def __init__(self, summary: dict) -> None:
        self.summary = summary
        self.intensity = self.rt = self.mass = None

    def correct(self, intensity: np.ndarray, rt: np.ndarray, mass: np.ndarray) -> dict:
        self.intensity, self.rt, self.mass = intensity.copy(), rt, mass
        intensity[self.summary["corrected"]] /= 2
        return self.summary


def _summary(corrected: list[bool], reference_runs: int = 3) -> dict:
    return {
        "corrected": np.array(corrected),
        "max_abs_correction": np.where(corrected, 0.2, 0.0),
        "n_mass_bins": 8,
        "reference_runs": reference_runs,
    }


def _patch_correction(monkeypatch, summary: dict) -> _RecordingDriftCorrection:
    fake = _RecordingDriftCorrection(summary)
    monkeypatch.setattr(f"{DRIFT_MODULE}.PrecursorDriftCorrection", lambda: fake)
    return fake


def _make_precursor_tables(
    runs: list[str] = PRECURSOR_RUNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precursor table with PRECURSOR_KEYS in that order and a psm table sorted by key, one psm per precursor and run."""
    precursor_df = pd.DataFrame(
        {
            "ion": PRECURSOR_KEYS,
            "mod_seq_charge_hash": PRECURSOR_KEYS,
            "pg": [f"PG{key}" for key in PRECURSOR_KEYS],
            "mod_seq_hash": PRECURSOR_KEYS,
            **{
                run: [float(100 * (i + 1) + key) for key in PRECURSOR_KEYS]
                for i, run in enumerate(runs)
            },
        }
    )
    keys = sorted(PRECURSOR_KEYS)
    psm_df = pd.DataFrame(
        {
            "mod_seq_charge_hash": np.tile(keys, len(runs)),
            "run": np.repeat(runs, len(keys)),
            "rt_observed": np.tile([RT_BY_KEY[key] for key in keys], len(runs)),
            "mz_library": np.tile([MZ_BY_KEY[key] for key in keys], len(runs)),
            "charge": np.tile([CHARGE_BY_KEY[key] for key in keys], len(runs)),
        }
    )
    return precursor_df, psm_df


def _neutral_mass(key: int) -> float:
    return (MZ_BY_KEY[key] - MASS_PROTON) * CHARGE_BY_KEY[key]


class TestCorrectPrecursorIntensityDrift:
    def test_passes_run_major_matrices_in_table_order(self, monkeypatch):
        """Given a precursor table ordered unlike its psms, when corrected, then the intensities, RTs and neutral masses reach the correction run-major and in table order."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        fake = _patch_correction(monkeypatch, _summary([True, True, True]))

        # When
        correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        np.testing.assert_array_equal(
            fake.intensity, precursor_df[PRECURSOR_RUNS].to_numpy().T
        )
        expected_rt = [RT_BY_KEY[key] for key in PRECURSOR_KEYS]
        np.testing.assert_array_equal(fake.rt, [expected_rt] * len(PRECURSOR_RUNS))
        np.testing.assert_allclose(
            fake.mass, [_neutral_mass(key) for key in PRECURSOR_KEYS]
        )
        for array in (fake.rt, fake.mass):
            assert array.dtype == np.float64
            assert array.flags["C_CONTIGUOUS"]

    def test_returns_a_float64_copy_with_the_corrected_values(self, monkeypatch):
        """Given a float32 precursor table, when corrected, then a float64 copy with the corrected run columns and unchanged metadata is returned and the inputs are untouched."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        precursor_df[PRECURSOR_RUNS] = precursor_df[PRECURSOR_RUNS].astype(np.float32)
        precursor_copy = precursor_df.copy()
        psm_copy = psm_df.copy()
        _patch_correction(monkeypatch, _summary([True, False, True]))

        # When
        result_df = correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        assert (result_df[PRECURSOR_RUNS].dtypes == np.float64).all()
        np.testing.assert_allclose(result_df["A1"], precursor_df["A1"] / 2)
        np.testing.assert_allclose(result_df["B1"], precursor_df["B1"])
        np.testing.assert_allclose(result_df["B2"], precursor_df["B2"] / 2)
        pd.testing.assert_frame_equal(
            result_df.drop(columns=PRECURSOR_RUNS),
            precursor_df.drop(columns=PRECURSOR_RUNS),
        )
        pd.testing.assert_frame_equal(precursor_df, precursor_copy)
        pd.testing.assert_frame_equal(psm_df, psm_copy)

    def test_passes_nan_rt_where_a_run_has_no_psm(self, monkeypatch):
        """Given a precursor without a psm in one run, when corrected, then its RT in that run is NaN and its mass still comes from its other psms."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        psm_df = psm_df[
            ~((psm_df["run"] == "B1") & (psm_df["mod_seq_charge_hash"] == 10))
        ]
        fake = _patch_correction(monkeypatch, _summary([True, True, True]))

        # When
        correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        column = PRECURSOR_KEYS.index(10)
        assert np.isnan(fake.rt[PRECURSOR_RUNS.index("B1"), column])
        assert np.isfinite(
            np.delete(fake.rt[:, column], PRECURSOR_RUNS.index("B1"))
        ).all()
        np.testing.assert_allclose(fake.mass[column], _neutral_mass(10))

    def test_uses_the_first_psm_of_a_precursor_in_a_run(self, monkeypatch):
        """Given a second psm of the same precursor and run with another RT and m/z, when corrected, then the first psm gives the RT and the mass."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        duplicate_df = psm_df.copy()
        duplicate_df["rt_observed"] += 50.0
        duplicate_df["mz_library"] += 100.0
        fake = _patch_correction(monkeypatch, _summary([True, True, True]))

        # When
        correct_precursor_intensity_drift(
            precursor_df, pd.concat([psm_df, duplicate_df], ignore_index=True)
        )

        # Then
        expected_rt = [RT_BY_KEY[key] for key in PRECURSOR_KEYS]
        np.testing.assert_array_equal(fake.rt, [expected_rt] * len(PRECURSOR_RUNS))
        np.testing.assert_allclose(
            fake.mass, [_neutral_mass(key) for key in PRECURSOR_KEYS]
        )

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

    def test_warns_about_the_uncorrected_runs_only(self, monkeypatch, caplog):
        """Given a summary with one uncorrected run, when corrected, then one warning names that run and no other."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        _patch_correction(monkeypatch, _summary([True, False, True]))

        # When
        with caplog.at_level(logging.WARNING):
            correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert "B1" in warnings[0]
        assert "A1" not in warnings[0] and "B2" not in warnings[0]

    def test_logs_the_mass_bins_and_the_reference_runs(self, monkeypatch, caplog):
        """Given a summary, when corrected, then the info line gives the corrected runs, the mass bins and the runs the reference precursors are quantified in."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        _patch_correction(monkeypatch, _summary([True, False, True], reference_runs=2))

        # When
        with caplog.at_level(logging.INFO):
            correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        summaries = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
        assert len(summaries) == 1
        assert "2 of 3 runs in 8 mass bins" in summaries[0]
        assert (
            "reference: precursors quantified in at least 2 of 3 runs" in summaries[0]
        )

    def test_does_not_warn_for_a_single_run(self, monkeypatch, caplog):
        """Given a single uncorrected run, when corrected, then nothing is warned."""
        # Given
        precursor_df, psm_df = _make_precursor_tables(runs=["A1"])
        _patch_correction(monkeypatch, _summary([False], reference_runs=0))

        # When
        with caplog.at_level(logging.WARNING):
            correct_precursor_intensity_drift(precursor_df, psm_df)

        # Then
        assert not caplog.records

    @pytest.mark.parametrize(
        "missing_column", ["rt_observed", "run", "mz_library", "charge"]
    )
    def test_raises_if_a_psm_column_is_missing(self, missing_column):
        """Given a psm table without a required column, when corrected, then a ValueError names the missing column."""
        # Given
        precursor_df, psm_df = _make_precursor_tables()
        psm_df = psm_df.drop(columns=[missing_column])

        # When / Then
        with pytest.raises(ValueError, match=missing_column):
            correct_precursor_intensity_drift(precursor_df, psm_df)

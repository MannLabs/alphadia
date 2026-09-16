import logging

import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform.quantification.intensity_drift_correction import (
    IntensityDriftCorrector,
    IntensityDriftStatistic,
)

# two conditions with three replicates each, as in a typical benchmark design
CONDITION_A = ["A1", "A2", "A3"]
CONDITION_B = ["B1", "B2", "B3"]
RUNS = CONDITION_A + CONDITION_B
GRADIENT_SECONDS = 1500.0
N_PRECURSORS = 5000
FRAGMENTS_PER_PRECURSOR = 3
# log2 drift imposed on the drifting runs at the end of the gradient
DRIFT_LOG2 = 0.3
ION_NOISE_LOG2 = 0.1
METADATA_COLUMNS = ["precursor_idx", "ion", "pg", "mod_seq_hash", "mod_seq_charge_hash"]
STATISTICS = [IntensityDriftStatistic.MODE, IntensityDriftStatistic.MEDIAN]


def _make_tables(
    drifting_runs: list[str] | None = None,
    n_precursors: int = N_PRECURSORS,
    drift_log2: float = DRIFT_LOG2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic fragment intensity, correlation and psm tables.

    Every precursor elutes at a uniform RT in [0, GRADIENT_SECONDS]. The drifting runs
    (run A1 by default) are multiplied by 2 ** (drift_log2 * rt / GRADIENT_SECONDS), so
    their deviation from the other runs grows linearly along the gradient.
    """
    drifting_runs = ["A1"] if drifting_runs is None else drifting_runs
    rng = np.random.default_rng(seed)
    precursor_idx = np.repeat(np.arange(n_precursors), FRAGMENTS_PER_PRECURSOR)
    n_ions = len(precursor_idx)
    rt = rng.uniform(0, GRADIENT_SECONDS, n_precursors)

    base = 2 ** rng.uniform(10, 20, n_ions)
    drift = 2 ** (drift_log2 * rt[precursor_idx] / GRADIENT_SECONDS)
    intensity = {run: base * 2 ** rng.normal(0, ION_NOISE_LOG2, n_ions) for run in RUNS}
    for run in drifting_runs:
        intensity[run] = intensity[run] * drift

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


def _ion_rt(intensity_df: pd.DataFrame, psm_df: pd.DataFrame, run: str) -> np.ndarray:
    return (
        psm_df[psm_df["run"] == run]
        .set_index("precursor_idx")["rt_observed"]
        .reindex(intensity_df["precursor_idx"])
        .to_numpy()
    )


def _deviation(intensity_df: pd.DataFrame, drifting_runs: list[str]) -> np.ndarray:
    """Per-ion log2 difference between the drifting runs and the other runs."""
    log = np.log2(intensity_df[RUNS].to_numpy())
    drifting = [RUNS.index(run) for run in drifting_runs]
    others = [i for i in range(len(RUNS)) if i not in drifting]
    return log[:, drifting].mean(axis=1) - log[:, others].mean(axis=1)


def _slope_per_1000s(rt: np.ndarray, values: np.ndarray) -> float:
    return np.polyfit(rt / 1000.0, values, 1)[0]


def _log2_change(result_df: pd.DataFrame, intensity_df: pd.DataFrame, run: str):
    return np.log2(result_df[run].to_numpy() / intensity_df[run].to_numpy())


class TestIntensityDriftCorrector:
    @pytest.mark.parametrize("statistic", STATISTICS)
    def test_removes_drift_of_a_single_run(self, statistic):
        """Given one run drifting along RT, when corrected, then its deviation from the other runs is flat along RT."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables(drifting_runs=["A1"])
        rt = _ion_rt(intensity_df, psm_df, "A1")
        assert _slope_per_1000s(rt, _deviation(intensity_df, ["A1"])) > 0.15

        # When
        result_df = IntensityDriftCorrector(psm_df, statistic=statistic).correct(
            intensity_df, correlation_df
        )

        # Then
        assert abs(_slope_per_1000s(rt, _deviation(result_df, ["A1"]))) < 0.02

    @pytest.mark.parametrize("statistic", STATISTICS)
    def test_removes_drift_of_a_whole_condition(self, statistic):
        """Given one of two balanced conditions drifting along RT, when corrected, then most of the deviation between the conditions is removed."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables(drifting_runs=CONDITION_A)
        rt = _ion_rt(intensity_df, psm_df, "A1")
        slope_before = _slope_per_1000s(rt, _deviation(intensity_df, CONDITION_A))
        assert slope_before > 0.15

        # When
        result_df = IntensityDriftCorrector(psm_df, statistic=statistic).correct(
            intensity_df, correlation_df
        )

        # Then: the across-run median sits between the conditions and halves a run's
        # deviation whenever that run is one of the two middle values, which skews the
        # deviation distribution; the mode therefore removes slightly less than the median
        slope_after = _slope_per_1000s(rt, _deviation(result_df, CONDITION_A))
        assert abs(slope_after) < 0.2 * slope_before

    def test_is_shape_only_and_keeps_every_run_level(self):
        """Given drifting runs, when corrected, then the mean log2 intensity of every run is unchanged."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables(drifting_runs=CONDITION_A)

        # When
        result_df = IntensityDriftCorrector(psm_df).correct(
            intensity_df, correlation_df
        )

        # Then
        for run in RUNS:
            assert np.isclose(
                np.log2(result_df[run]).mean(), np.log2(intensity_df[run]).mean()
            )

    def test_leaves_undrifted_data_almost_unchanged(self):
        """Given runs without drift, when corrected, then no correction beyond the noise of the curve estimate is applied."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables(drift_log2=0.0)

        # When
        result_df = IntensityDriftCorrector(psm_df).correct(
            intensity_df, correlation_df
        )

        # Then
        for run in RUNS:
            assert np.abs(_log2_change(result_df, intensity_df, run)).max() < 0.1

    def test_preserves_shape_metadata_and_zeros(self):
        """Given a table with missing (zero) cells, when corrected, then zeros, metadata and row order are preserved."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables()
        intensity_df.loc[::7, "A1"] = 0.0
        zero_mask = intensity_df["A1"].to_numpy() == 0.0

        # When
        result_df = IntensityDriftCorrector(psm_df).correct(
            intensity_df, correlation_df
        )

        # Then
        assert list(result_df.columns) == list(intensity_df.columns)
        assert len(result_df) == len(intensity_df)
        assert (result_df["A1"].to_numpy()[zero_mask] == 0.0).all()
        assert (result_df["A1"].to_numpy()[~zero_mask] > 0.0).all()
        pd.testing.assert_frame_equal(
            result_df[METADATA_COLUMNS], intensity_df[METADATA_COLUMNS]
        )

    def test_does_not_modify_inputs(self):
        """Given intensity and correlation tables, when corrected, then the inputs are untouched."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables()
        intensity_copy = intensity_df.copy()
        correlation_copy = correlation_df.copy()

        # When
        IntensityDriftCorrector(psm_df).correct(intensity_df, correlation_df)

        # Then
        pd.testing.assert_frame_equal(intensity_df, intensity_copy)
        pd.testing.assert_frame_equal(correlation_df, correlation_copy)

    def test_uses_only_well_correlating_ions_for_the_curve(self):
        """Given poorly correlating ions carrying a fake drift, when corrected, then the curve ignores them."""
        # Given: the well-correlating ions have no drift, the poorly correlating ones do
        intensity_df, correlation_df, psm_df = _make_tables(drift_log2=0.0)
        bad = np.zeros(len(intensity_df), dtype=bool)
        bad[::2] = True
        rt = _ion_rt(intensity_df, psm_df, "A1")
        intensity_df.loc[bad, "A1"] *= 2 ** (1.0 * rt[bad] / GRADIENT_SECONDS)
        correlation_df.loc[bad, RUNS] = 0.3

        # When
        result_df = IntensityDriftCorrector(psm_df, min_correlation=0.8).correct(
            intensity_df, correlation_df
        )

        # Then: the good ions were not distorted by the bad ions' drift
        change = _log2_change(result_df, intensity_df, "A1")
        assert abs(_slope_per_1000s(rt[~bad], change[~bad])) < 0.02

    @pytest.mark.parametrize("missing_column", ["rt_observed", "run"])
    def test_skips_with_warning_when_psm_columns_are_missing(
        self, missing_column, caplog
    ):
        """Given a psm table without rt_observed or run, when corrected, then the table is returned unchanged with a warning."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables(n_precursors=10)
        psm_df = psm_df.drop(columns=[missing_column])

        # When
        with caplog.at_level(logging.WARNING):
            result_df = IntensityDriftCorrector(psm_df).correct(
                intensity_df, correlation_df
            )

        # Then
        pd.testing.assert_frame_equal(result_df, intensity_df)
        assert any(
            missing_column in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_skips_runs_with_too_few_ions(self):
        """Given fewer usable ions than the minimum per run, when corrected, then the table is returned unchanged."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables(n_precursors=100)
        assert len(intensity_df) < IntensityDriftCorrector.DEFAULT_MIN_IONS_PER_RUN

        # When
        result_df = IntensityDriftCorrector(psm_df).correct(
            intensity_df, correlation_df
        )

        # Then
        pd.testing.assert_frame_equal(result_df, intensity_df)

    def test_places_ions_without_rt_in_a_run_at_their_mean_rt(self):
        """Given a precursor without rt_observed in the drifting run, when corrected, then it is still corrected like its RT neighbours."""
        # Given
        intensity_df, correlation_df, psm_df = _make_tables()
        dropped_precursor = 0
        psm_df = psm_df[
            ~((psm_df["run"] == "A1") & (psm_df["precursor_idx"] == dropped_precursor))
        ]
        rows = intensity_df["precursor_idx"].to_numpy() == dropped_precursor
        rt = _ion_rt(intensity_df, psm_df, "B1")

        # When
        result_df = IntensityDriftCorrector(psm_df).correct(
            intensity_df, correlation_df
        )

        # Then
        change = _log2_change(result_df, intensity_df, "A1")
        neighbours = (np.abs(rt - rt[rows][0]) < 30) & ~rows
        assert np.isclose(change[rows].mean(), change[neighbours].mean(), atol=0.03)

    def test_rejects_unknown_statistic(self):
        """Given an unknown statistic, when constructed, then a ValueError is raised."""
        # When / Then
        with pytest.raises(ValueError, match="statistic"):
            IntensityDriftCorrector(pd.DataFrame(), statistic="mean")

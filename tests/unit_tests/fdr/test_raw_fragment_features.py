import numpy as np
import pytest

from alphadia.fdr import raw_fragment_features as rff

K = 4
TOLERANCE_PPM = 7.0
GAUSSIAN = np.exp(-0.5 * ((np.arange(rff.TRACE_WINDOW_CYCLES) - rff.APEX) / 1.5) ** 2)


def _traces(n_rows: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Rows with K slots (the last one empty) whose fragments follow one gaussian peak at the apex, with noise."""
    scale = rng.uniform(1e3, 1e5, (n_rows, K, 1))
    xic = (
        scale * GAUSSIAN + rng.uniform(0, 50, (n_rows, K, rff.TRACE_WINDOW_CYCLES))
    ).astype(np.float32)
    xic[:, K - 1] = 0.0
    error = rng.normal(0, 2.0, xic.shape).astype(np.float32)
    error[xic == 0] = np.nan
    return {
        "xic": xic,
        "mz_error_ppm": error,
        "isotope_xic": (0.3 * xic).astype(np.float32),
        "companion_xic": np.zeros_like(xic),
        "fragment_mz": np.tile(np.array([400.0, 600.0, 800.0, 0.0], np.float32), (n_rows, 1)),
        "fragment_charge": np.tile(np.array([1, 1, 2, 0], np.uint8), (n_rows, 1)),
        "library_intensity": np.tile(np.array([1.0, 0.6, 0.3, 0.0], np.float32), (n_rows, 1)),
        "apex_tic": rng.uniform(1e5, 1e7, n_rows).astype(np.float32),
    }  # fmt: skip


def test_fit_precision_recovers_the_core_width_on_a_uniform_background():
    rng = np.random.default_rng(1)
    n = 40_000
    core = rng.random(n) < 0.7
    error = np.where(
        core, rng.normal(0, 1.5, n), rng.uniform(-TOLERANCE_PPM, TOLERANCE_PPM, n)
    )
    total = rng.uniform(1e3, 1e6, n)

    centres, sd = rff.fit_precision(error, total, TOLERANCE_PPM)

    assert len(centres) == len(sd) == rff.INTENSITY_BINS
    assert np.all(np.diff(centres) > 0)
    assert np.allclose(sd, 1.5, atol=0.25)


def test_fragment_errors_weight_the_scored_cycles_by_intensity():
    xic = np.zeros((1, 2, rff.TRACE_WINDOW_CYCLES), np.float32)
    error = np.full(xic.shape, np.nan, np.float32)
    xic[0, 0, rff.APEX], error[0, 0, rff.APEX] = 3.0, 2.0
    xic[0, 0, rff.APEX + 1], error[0, 0, rff.APEX + 1] = 1.0, -2.0
    # outside the scored cycles: ignored
    xic[0, 0, 0], error[0, 0, 0] = 100.0, 5.0

    mean, total = rff.fragment_errors(xic, error)

    assert mean[0, 0] == pytest.approx(1.0)
    assert total[0, 0] == pytest.approx(4.0)
    assert np.isnan(mean[0, 1])
    assert total[0, 1] == 0.0


def test_chunk_features_are_finite_and_follow_the_traces():
    rng = np.random.default_rng(0)
    traces = _traces(200, rng)
    precision = (
        np.linspace(3, 6, rff.INTENSITY_BINS),
        np.full(rff.INTENSITY_BINS, 2.0),
    )

    values = rff.chunk_features(traces, precision, TOLERANCE_PPM)

    assert set(values) == set(rff.FEATURES)
    for name in rff.FEATURES:
        assert values[name].shape == (200,)
        assert values[name].dtype == np.float32
        assert np.isfinite(values[name]).all(), name
    # every fragment co-elutes with the peak and its isotope trace has the same shape
    assert np.allclose(values["t5_iso_coh_share"], 1.0)
    assert (values["t5_snr_libw"] > 0).all()
    assert np.allclose(values["t5_wrongz_libw"], 0.0)
    assert np.allclose(values[rff.WINDOW_TIC], np.log1p(traces["apex_tic"]))


def test_chunk_features_of_a_candidate_without_signal_are_zero():
    traces = _traces(1, np.random.default_rng(0))
    for name in ("xic", "isotope_xic", "companion_xic"):
        traces[name][:] = 0.0
    traces["mz_error_ppm"][:] = np.nan
    precision = (
        np.linspace(3, 6, rff.INTENSITY_BINS),
        np.full(rff.INTENSITY_BINS, 2.0),
    )

    values = rff.chunk_features(traces, precision, TOLERANCE_PPM)

    for name in (*rff.MASS, *rff.ISOTOPES, *rff.PEAK):
        assert values[name][0] == 0.0, name


def test_mass_error_llr_prefers_errors_within_the_precision():
    rng = np.random.default_rng(0)
    traces = _traces(2, rng)
    traces["mz_error_ppm"][0] = np.where(
        np.isnan(traces["mz_error_ppm"][0]), np.nan, 0.5
    )
    traces["mz_error_ppm"][1] = np.where(
        np.isnan(traces["mz_error_ppm"][1]), np.nan, 6.0
    )
    precision = (
        np.linspace(3, 6, rff.INTENSITY_BINS),
        np.full(rff.INTENSITY_BINS, 1.0),
    )

    values = rff.chunk_features(traces, precision, TOLERANCE_PPM)

    assert values["t5_me_llr_libw"][0] > 0 > values["t5_me_llr_libw"][1]
    assert values["t5_me_llr_sum"][0] > values["t5_me_llr_sum"][1]


def test_compute_matches_chunk_features_in_chunks(monkeypatch):
    monkeypatch.setattr(rff, "CHUNK", 64)
    monkeypatch.setattr(rff, "PRECISION_SAMPLE_ROWS", 100)
    traces = _traces(150, np.random.default_rng(3))
    calls = []

    def export(rows: np.ndarray) -> dict[str, np.ndarray]:
        calls.append(len(rows))
        return {name: values[rows] for name, values in traces.items()}

    values = rff.compute(150, export, TOLERANCE_PPM)

    sample = rff.sample_rows(150)
    precision = rff.fit_precision(
        *rff.fragment_errors(traces["xic"][sample], traces["mz_error_ppm"][sample]),
        TOLERANCE_PPM,
    )
    expected = rff.chunk_features(traces, precision, TOLERANCE_PPM)
    # the sample in chunks of 64, then all rows in chunks of 64
    assert calls == [64, 36, 64, 64, 22]
    for name in rff.FEATURES:
        np.testing.assert_allclose(values[name], expected[name], rtol=1e-6, atol=1e-6)

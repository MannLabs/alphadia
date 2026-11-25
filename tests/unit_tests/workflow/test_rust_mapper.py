"""Unit tests for the alphadia-search-rs mapper."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from alphadia.workflow.peptidecentric.rust.mapper import to_features_df


def test_to_features_df_sets_the_library_mz_as_observed_mz():
    # given: scored candidates and their library precursors
    candidate_features = MagicMock()
    candidate_features.to_dict_arrays.return_value = {
        "precursor_idx": [2, 1],
        "rank": [0, 0],
        "score": [1.0, 2.0],
    }
    spectral_library = MagicMock()
    spectral_library.precursor_df = pd.DataFrame(
        {
            "precursor_idx": [1, 2],
            "decoy": [0, 1],
            "elution_group_idx": [0, 0],
            "channel": [0, 0],
            "proteins": ["P1", "P1"],
            "mz_library": [501.0, 502.0],
            "mz_calibrated": [501.1, 502.1],
        }
    )

    # when
    features_df = to_features_df(candidate_features, spectral_library)

    # then: like the quantified precursors, the observed m/z is the library m/z
    assert features_df["mz_observed"].tolist() == [502.0, 501.0]
    assert features_df["mz_library"].tolist() == [502.0, 501.0]


def _flat_library(n_precursors: int = 40, n_fragments: int = 5):
    """Flat library in memory with library columns only, in a random precursor order."""
    import numpy as np
    from alphabase.spectral_library.flat import SpecLibFlat

    rng = np.random.default_rng(0)
    order = rng.permutation(n_precursors)
    precursor_df = pd.DataFrame(
        {
            "precursor_idx": order,
            "mz_library": rng.uniform(400.0, 1000.0, n_precursors),
            "rt_library": rng.uniform(0.0, 1.0, n_precursors),
            "nAA": rng.integers(7, 30, n_precursors),
            "flat_frag_start_idx": np.arange(n_precursors) * n_fragments,
            "flat_frag_stop_idx": np.arange(1, n_precursors + 1) * n_fragments,
            "decoy": rng.integers(0, 2, n_precursors),
        }
    )
    n = n_precursors * n_fragments
    fragment_df = pd.DataFrame(
        {
            "mz_library": rng.uniform(150.0, 1500.0, n).astype(np.float32),
            "intensity": rng.uniform(0.0, 1.0, n).astype(np.float32),
            "cardinality": np.ones(n, dtype=np.uint8),
            "charge": rng.integers(1, 3, n).astype(np.int8),
            "loss_type": np.zeros(n, dtype=np.int16),
            "number": rng.integers(1, 20, n).astype(np.uint32),
            "position": rng.integers(0, 20, n).astype(np.uint32),
            "type": rng.choice([98, 121], n).astype(np.int8),
        }
    )
    speclib = SpecLibFlat()
    speclib._precursor_df = precursor_df
    speclib._fragment_df = fragment_df
    return speclib


def _dia_data_rust(rt_start: float, rt_stop: float, windows: list[tuple[float, float]]):
    """DIA data in the alphadia-search-rs format with one spectrum per window and cycle."""
    import numpy as np
    from alphadia_search_rs import DIAData

    n_windows = len(windows)
    n_cycles = 10
    n_spectra = n_windows * n_cycles
    cycle = np.zeros((1, n_windows, 1, 2), dtype=np.float32)
    cycle[0, :, 0, :] = windows
    lower = np.tile(cycle[0, :, 0, 0], n_cycles)
    return DIAData.from_arrays(
        np.tile(np.arange(n_windows), n_cycles).astype(np.int64),
        lower,
        np.tile(cycle[0, :, 0, 1], n_cycles),
        np.arange(n_spectra, dtype=np.int64),
        np.arange(1, n_spectra + 1, dtype=np.int64),
        np.repeat(np.arange(n_cycles), n_windows).astype(np.int64),
        np.repeat(np.linspace(rt_start, rt_stop, n_cycles), n_windows).astype(
            np.float32
        ),
        lower + 1.0,
        np.ones(n_spectra, dtype=np.float32),
        cycle,
    )


def _assert_same_rust_library(actual, expected, rtol=1e-6):
    import numpy as np

    for actual_dict, expected_dict in zip(
        actual.to_dict_arrays(), expected.to_dict_arrays(), strict=True
    ):
        assert actual_dict.keys() == expected_dict.keys()
        for key in actual_dict:
            np.testing.assert_allclose(
                actual_dict[key], expected_dict[key], rtol=rtol, err_msg=key
            )


def test_run_speclib_to_rust_is_the_same_as_init_spectral_library():
    import numpy as np

    from alphadia.workflow.peptidecentric.library_init import init_spectral_library
    from alphadia.workflow.peptidecentric.rust.mapper import (
        library_speclib_to_rust,
        run_speclib_to_rust,
    )

    # given: a library and DIA data that isolate only a part of the precursor m/z range
    speclib = _flat_library()
    dia_data = _dia_data_rust(30.0, 1800.0, [(500.0, 650.0), (650.0, 800.5)])
    speclib_rust = library_speclib_to_rust(speclib)
    num_precursors = speclib_rust.num_precursors

    # when: the Python library and the Rust library are prepared for the run
    run_speclib_rust = run_speclib_to_rust(speclib_rust, dia_data)
    init_spectral_library(dia_data.cycle, dia_data.rt_values, MagicMock(), speclib)

    # then: both have the same precursors with the same RT, and the source Rust library is not changed
    precursor_dict, _ = run_speclib_rust.to_dict_arrays()
    expected_df = speclib.precursor_df.sort_values("precursor_idx")
    assert 0 < len(expected_df) < num_precursors
    np.testing.assert_array_equal(
        precursor_dict["precursor_idx"], expected_df["precursor_idx"].to_numpy()
    )
    np.testing.assert_allclose(
        precursor_dict["rt_library"], expected_df["rt_library"].to_numpy(), rtol=1e-6
    )
    np.testing.assert_array_equal(precursor_dict["rt"], precursor_dict["rt_library"])
    assert speclib_rust.num_precursors == num_precursors


@pytest.mark.parametrize("is_calibrated", [True, False])
def test_calibrated_speclib_to_rust_is_the_same_as_speclib_to_rust(is_calibrated):
    import numpy as np

    from alphadia.calibration.estimator import CalibrationEstimator
    from alphadia.workflow.peptidecentric.library_init import init_spectral_library
    from alphadia.workflow.peptidecentric.rust.mapper import (
        calibrated_speclib_to_rust,
        library_speclib_to_rust,
        run_speclib_to_rust,
        speclib_to_rust,
    )

    # given: a library prepared for a run and a batch of it, with fitted calibration estimators
    rng = np.random.default_rng(1)
    speclib = _flat_library(n_precursors=400)
    dia_data = _dia_data_rust(30.0, 1800.0, [(400.0, 1000.0)])
    run_speclib_rust = run_speclib_to_rust(library_speclib_to_rust(speclib), dia_data)
    init_spectral_library(dia_data.cycle, dia_data.rt_values, MagicMock(), speclib)

    batch = speclib.precursor_df.sample(100, random_state=0)
    batch_fragment_df = speclib.fragment_df.copy()
    estimators = {}
    for name, input_column, n_kernels, transform, deviation in [
        (
            "rt",
            "rt_library",
            6,
            None,
            lambda x: 0.01 * x + 5 + rng.normal(0, 3, len(x)),
        ),
        (
            "mz",
            "mz_library",
            2,
            1e6,
            lambda x: x * (3e-6 + rng.normal(0, 2e-6, len(x))),
        ),
    ]:
        estimator = CalibrationEstimator(
            name,
            n_kernels,
            transform,
            input_column,
            "observed",
            input_column.replace("library", "calibrated"),
        )
        values = batch[input_column].to_numpy()
        estimator.fit(
            pd.DataFrame(
                {input_column: values, "observed": values + deviation(values)}
            ),
            plot=False,
        )
        estimators[name] = estimator

    # Python path: calibrated columns in the dataframes, as the calibration manager writes them
    batch_speclib = MagicMock()
    batch_speclib.precursor_df = batch.copy()
    batch_speclib.fragment_df = batch_fragment_df
    if is_calibrated:
        estimators["rt"].predict(batch_speclib.precursor_df)
        estimators["mz"].predict(batch_speclib.precursor_df)
        estimators["mz"].predict(batch_speclib.fragment_df)
    suffix = "calibrated" if is_calibrated else "library"
    column_name_handler = MagicMock()
    column_name_handler.get_rt_column.return_value = f"rt_{suffix}"
    column_name_handler.get_precursor_mz_column.return_value = f"mz_{suffix}"
    column_name_handler.get_fragment_mz_column.return_value = f"mz_{suffix}"
    calibration_manager = MagicMock()
    calibration_manager.get_estimator.side_effect = lambda group, name: estimators[name]

    # when
    speclib_rust = calibrated_speclib_to_rust(
        run_speclib_rust, batch_speclib, calibration_manager, column_name_handler
    )

    # then: the result is the same as the conversion of the calibrated columns (the fragments of the batch only)
    expected = speclib_to_rust(
        batch_speclib,
        rt_column=f"rt_{suffix}",
        precursor_mz_column=f"mz_{suffix}",
        fragment_mz_column=f"mz_{suffix}",
    ).subset(batch["precursor_idx"].to_numpy(dtype=np.uint64))
    _assert_same_rust_library(speclib_rust, expected)
    assert speclib_rust.num_precursors == 100

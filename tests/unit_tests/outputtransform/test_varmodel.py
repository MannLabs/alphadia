import logging
from unittest.mock import patch

import numpy as np
import pandas as pd

from alphadia.outputtransform.quantification import quant_builder
from alphadia.outputtransform.quantification.quant_builder import (
    INTENSITY_LIBRARY_COLUMN,
    QuantBuilder,
)

# the fit itself is tested in alphadia_search_rs (src/varmodel_quantification/tests*.rs), these tests
# cover the conversion of the fragment matrices to the precursor table

RUNS = ["run1", "run2", "run3"]


def _fragment_table(intensity_library, precursor, intensity):
    intensity_df = pd.DataFrame(intensity, columns=RUNS[: intensity.shape[1]])
    intensity_df.insert(0, "precursor_idx", precursor)
    intensity_df.insert(1, "ion", np.arange(len(intensity_library)))
    intensity_df["pg"] = [f"PG{p}" for p in precursor]
    intensity_df["mod_seq_hash"] = precursor
    intensity_df["mod_seq_charge_hash"] = precursor
    intensity_df[INTENSITY_LIBRARY_COLUMN] = intensity_library
    return intensity_df


def _correlation_table(intensity_df, correlation=1.0):
    return intensity_df.assign(
        **{run: correlation for run in RUNS if run in intensity_df}
    )


def test_precursor_table_has_one_sorted_row_per_precursor_with_its_runs():
    # Given two precursors in reversed order whose fragments follow the library pattern,
    # precursor 20 at twice the level in run2 and missing in run3
    intensity_library = np.array([1.0, 0.5, 0.25, 1.0, 0.5])
    precursor = np.array([20, 20, 20, 10, 10])
    totals = {20: [1000.0, 2000.0, 0.0], 10: [500.0, 500.0, 500.0]}
    intensity = np.array(
        [
            np.array(totals[p])
            * fragment_intensity_library
            / intensity_library[precursor == p].sum()
            for fragment_intensity_library, p in zip(
                intensity_library, precursor, strict=True
            )
        ]
    )
    intensity_df = _fragment_table(intensity_library, precursor, intensity)

    # When
    precursor_df = QuantBuilder(pd.DataFrame()).varmodel_precursor_table(
        intensity_df, _correlation_table(intensity_df)
    )

    # Then
    assert precursor_df["mod_seq_charge_hash"].tolist() == [10, 20]
    assert precursor_df["pg"].tolist() == ["PG10", "PG20"]
    np.testing.assert_allclose(
        precursor_df.loc[0, RUNS].astype(float), 500.0, rtol=1e-4
    )
    np.testing.assert_allclose(
        precursor_df.loc[1, RUNS].astype(float), [1000.0, 2000.0, 0.0], rtol=1e-4
    )


def test_fragments_without_library_intensity_are_ignored_with_a_warning(caplog):
    # Given one fragment with a zero library intensity and one missing from the library
    intensity_library = np.array([1.0, 0.5, 0.0, np.nan])
    intensity = np.array([[100.0, 200.0], [50.0, 100.0], [10.0, 99.0], [25.0, 51.0]])
    intensity_df = _fragment_table(intensity_library, np.zeros(4, dtype=int), intensity)

    # When
    with caplog.at_level(logging.WARNING):
        precursor_df = QuantBuilder(pd.DataFrame()).varmodel_precursor_table(
            intensity_df, _correlation_table(intensity_df)
        )

    # Then
    assert any(
        "2 fragment ions without library intensity" in r.message for r in caplog.records
    )
    np.testing.assert_allclose(
        precursor_df.loc[0, "run2"] / precursor_df.loc[0, "run1"], 2.0, rtol=1e-4
    )


def _fit_call(intensity_df, correlation_df, quality_exponent):
    """Positional arguments and keyword options the precursor table passes to the backend fit."""
    n_precursors = intensity_df["precursor_idx"].nunique()
    n_runs = len([c for c in RUNS if c in intensity_df])
    fit_result = {
        "quantity": np.zeros((n_runs, n_precursors)),
        "residual_scale": 0.0,
        "loss_scale": 0.0,
        "offset_shrinkage": 0.0,
    }
    with patch.object(quant_builder, "VarModelQuantification") as backend:
        backend.return_value.fit.return_value = fit_result
        QuantBuilder(pd.DataFrame()).varmodel_precursor_table(
            intensity_df, correlation_df, quality_exponent=quality_exponent
        )
    return backend.return_value.fit.call_args


def test_several_runs_censor_and_pass_the_correlation_by_run():
    # Given two runs with per-cell correlations that differ by run and fragment, one missing and one out of range
    intensity_library = np.array([1.0, 0.5, 0.25])
    intensity = np.array([[100.0, 200.0], [50.0, 0.0], [25.0, 50.0]])
    intensity_df = _fragment_table(intensity_library, np.zeros(3, dtype=int), intensity)
    correlation_df = intensity_df.assign(run1=[0.9, np.nan, 0.7], run2=[0.6, 0.5, -0.1])

    # When
    args, options = _fit_call(intensity_df, correlation_df, quality_exponent=1.5)

    # Then the correlation is laid out like the intensity, one row per run, within [0, 1]
    intensity_arg, correlation_arg = args[0], args[4]
    assert options == {"fragment_censoring_correction": True}
    assert correlation_arg.shape == intensity_arg.shape
    assert correlation_arg.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(
        correlation_arg, [[0.9, 0.0, 0.7], [0.6, 0.5, 0.0]], rtol=1e-6
    )


def test_a_single_run_uses_the_plain_fit():
    # Given one run, where there are no run ratios to protect
    intensity_library = np.array([1.0, 0.5])
    intensity = np.array([[100.0], [50.0]])
    intensity_df = _fragment_table(intensity_library, np.zeros(2, dtype=int), intensity)

    # When
    _, options = _fit_call(
        intensity_df, _correlation_table(intensity_df, 0.9), quality_exponent=1.5
    )

    # Then
    assert options == {"fragment_censoring_correction": False}

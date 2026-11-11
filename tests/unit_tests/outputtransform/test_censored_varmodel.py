import numpy as np
import pandas as pd
from alphadia_search_rs import VarModelQuantification

from alphadia.outputtransform.quantification.censored_varmodel import (
    censored_quantities,
)
from alphadia.outputtransform.quantification.quant_builder import (
    INTENSITY_LIBRARY_COLUMN,
    QuantBuilder,
)

N_PRECURSORS = 3000
N_FRAGMENTS = 8
N_RUNS = 6
NOISE_SD = 0.25
DETECTION_LIMIT = 9.0
SEED = 1


def simulate(log2_ratio: float, detection_limit: float | None):
    """Precursors at log2 abundance a in runs 0-2 and a + log2_ratio in runs 3-5; fragments below the limit missing."""
    rng = np.random.default_rng(SEED)
    abundance = rng.uniform(10.0, 18.0, N_PRECURSORS)
    shift = np.r_[np.zeros(N_RUNS // 2), np.full(N_RUNS // 2, log2_ratio)]
    library = rng.uniform(0.05, 1.0, (N_PRECURSORS, N_FRAGMENTS))
    log_share = np.log2(library / library.sum(1, keepdims=True))
    log_intensity = (
        abundance[:, None, None]
        + shift[None, None, :]
        + log_share[:, :, None]
        + rng.normal(0, NOISE_SD, (N_PRECURSORS, N_FRAGMENTS, N_RUNS))
    )
    intensity = np.exp2(log_intensity)
    if detection_limit is not None:
        intensity[log_intensity < detection_limit] = 0.0
    precursor_index = np.repeat(np.arange(N_PRECURSORS), N_FRAGMENTS)
    return (
        intensity.reshape(-1, N_RUNS),
        library.ravel(),
        precursor_index,
        abundance,
    )


def test_without_censoring_the_port_reproduces_the_backend_fit():
    intensity, library, precursor_index, _ = simulate(-1.0, DETECTION_LIMIT)

    port = censored_quantities(
        intensity, library, precursor_index, N_PRECURSORS, censored=False
    )
    backend = (
        VarModelQuantification()
        .fit(
            np.ascontiguousarray(intensity.T.astype(np.float32)),
            library.astype(np.float32),
            precursor_index.astype(np.uintp),
            N_PRECURSORS,
        )["quantity"]
        .T
    )

    both = np.isfinite(port) & np.isfinite(backend)
    assert both.mean() > 0.6
    np.testing.assert_allclose(port[both], backend[both], atol=0.02)


def test_censoring_removes_the_compression_near_the_detection_limit():
    # Given precursors whose fragments in the low condition straddle the detection limit
    log2_ratio = -2.0
    intensity, library, precursor_index, abundance = simulate(
        log2_ratio, DETECTION_LIMIT
    )
    quantities = {
        censored: censored_quantities(
            intensity, library, precursor_index, N_PRECURSORS, censored=censored
        )
        for censored in (False, True)
    }
    quantified = np.isfinite(quantities[False]).all(1) & np.isfinite(
        quantities[True]
    ).all(1)
    straddling = quantified & (abundance > 13.5) & (abundance < 14.5)

    # When
    bias = {
        censored: np.median(
            q[straddling, N_RUNS // 2 :].mean(1) - q[straddling, : N_RUNS // 2].mean(1)
        )
        - log2_ratio
        for censored, q in quantities.items()
    }

    # Then the plain fit reads the low runs too high (compression) and the censored fit does not
    assert bias[False] > 0.03
    assert abs(bias[True]) < bias[False]


def test_precursor_table_censored_option_quantifies_every_precursor():
    intensity, library, precursor_index, _ = simulate(-1.0, None)
    runs = [f"run{i}" for i in range(N_RUNS)]
    intensity_df = pd.DataFrame(intensity, columns=runs)
    intensity_df.insert(0, "precursor_idx", precursor_index)
    intensity_df.insert(1, "ion", np.arange(len(library)))
    intensity_df["pg"] = precursor_index
    intensity_df["mod_seq_hash"] = precursor_index
    intensity_df["mod_seq_charge_hash"] = precursor_index
    intensity_df[INTENSITY_LIBRARY_COLUMN] = library

    precursor_df = QuantBuilder(pd.DataFrame()).varmodel_precursor_table(
        intensity_df, missing_fragments="censored"
    )

    assert len(precursor_df) == N_PRECURSORS
    assert (precursor_df[runs].to_numpy() > 0).all()


def test_quality_exponent_downweights_interfered_values():
    # Given one fragment per precursor carrying a 2x interference in the last run, with a poor XIC correlation there
    # (larger outliers are already caught by the robust weights)
    intensity, library, precursor_index, _ = simulate(0.0, None)
    first_fragment = np.r_[True, np.diff(precursor_index) > 0]
    intensity[first_fragment, -1] *= 2.0
    correlation = np.full(intensity.shape, 0.95)
    correlation[first_fragment, -1] = 0.1

    # When
    shift = {
        gamma: np.median(
            (
                q := censored_quantities(
                    intensity,
                    library,
                    precursor_index,
                    N_PRECURSORS,
                    correlation=correlation,
                    quality_exponent=gamma,
                )
            )[:, -1]
            - q[:, :-1].mean(1)
        )
        for gamma in (0.0, 1.5)
    }

    # Then the interfered run reads less inflated with the quality weighting
    assert shift[0.0] > 0.05
    assert shift[1.5] < 0.75 * shift[0.0]


def test_precursor_table_of_a_single_run_uses_the_backend_fit():
    # given: one run, where the numpy fit has no fragments seen in several runs to estimate its noise curve from
    intensity, library, precursor_index, _ = simulate(0.0, None)
    intensity_df = pd.DataFrame({"run0": intensity[:, 0]})
    intensity_df.insert(0, "precursor_idx", precursor_index)
    intensity_df.insert(1, "ion", np.arange(len(library)))
    intensity_df["pg"] = precursor_index
    intensity_df["mod_seq_hash"] = precursor_index
    intensity_df["mod_seq_charge_hash"] = precursor_index
    intensity_df[INTENSITY_LIBRARY_COLUMN] = library
    correlation_df = intensity_df.assign(run0=0.9)

    # when
    precursor_df = QuantBuilder(pd.DataFrame()).varmodel_precursor_table(
        intensity_df,
        missing_fragments="censored",
        correlation_df=correlation_df,
        quality_exponent=1.5,
    )

    # then
    assert len(precursor_df) == N_PRECURSORS
    assert (precursor_df["run0"] > 0).all()

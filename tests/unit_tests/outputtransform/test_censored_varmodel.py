import numpy as np
import pandas as pd
from alphadia_search_rs import VarModelQuantification

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
CORRELATION_SEED = 7
REFERENCE_TOLERANCE = 2e-4
# quantities of the first precursors from the array implementation the censored fit was validated against (dropout
# curve on every measurement, all precursors swept in lockstep), on the simulation below with float32 inputs and
# correlations drawn with CORRELATION_SEED
REFERENCE_CENSORED_G15 = np.array(
    [
        [
            14.108416001952,
            13.874226813174,
            14.034860309895,
            13.026770541503,
            13.083449747145,
            13.2233685966,
        ],
        [
            17.737312000812,
            17.656306024715,
            17.512980086071,
            16.624840076002,
            16.629909130891,
            16.82865774444,
        ],
        [np.nan] * N_RUNS,
        [
            17.639847688761,
            17.594789333111,
            17.538307952362,
            16.620343315772,
            16.536957324443,
            16.597799885239,
        ],
    ]
)
REFERENCE_DROP_G1 = np.array(
    [
        [
            14.112237041889,
            13.906194732911,
            14.094810461036,
            13.033826736763,
            13.076673767097,
            13.22209950443,
        ],
        [
            17.729902937759,
            17.656766540045,
            17.513641607546,
            16.620774978435,
            16.641287230911,
            16.81776757159,
        ],
        [np.nan] * N_RUNS,
        [
            17.636630281078,
            17.59612940009,
            17.536601081281,
            16.622618532447,
            16.535514226902,
            16.592772722142,
        ],
    ]
)


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


def censored_quantities(
    intensity: np.ndarray,
    library: np.ndarray,
    precursor_index: np.ndarray,
    n_precursors: int,
    censored: bool = True,
    correlation: np.ndarray | None = None,
    quality_exponent: float = 0.0,
) -> np.ndarray:
    """(precursors, runs) log2 quantities of the backend fit on (fragments, runs) intensities."""
    options = {}
    if correlation is not None:
        options["correlation"] = np.ascontiguousarray(correlation.T.astype(np.float32))
    return (
        VarModelQuantification()
        .fit(
            np.ascontiguousarray(intensity.T.astype(np.float32)),
            library.astype(np.float32),
            precursor_index.astype(np.uintp),
            n_precursors,
            missing_fragments="censored" if censored else "drop",
            quality_exponent=quality_exponent,
            **options,
        )["quantity"]
        .T
    )


def test_without_censoring_the_extended_fit_reproduces_the_plain_fit():
    # Given a quality exponent too small to change any weight, which still runs the extended (f64) fit
    intensity, library, precursor_index, _ = simulate(-1.0, DETECTION_LIMIT)

    # When
    extended = censored_quantities(
        intensity,
        library,
        precursor_index,
        N_PRECURSORS,
        censored=False,
        correlation=np.full(intensity.shape, 0.5),
        quality_exponent=1e-12,
    )
    plain = censored_quantities(
        intensity, library, precursor_index, N_PRECURSORS, censored=False
    )

    # Then
    both = np.isfinite(extended) & np.isfinite(plain)
    assert both.mean() > 0.6
    np.testing.assert_allclose(extended[both], plain[both], atol=0.02)


def test_extended_fit_agrees_with_the_reference_implementation():
    # the binned dropout likelihood and the per-precursor convergence move the quantities within the convergence
    # tolerance only
    # Given
    intensity, library, precursor_index, _ = simulate(-1.0, DETECTION_LIMIT)
    correlation = np.random.default_rng(CORRELATION_SEED).uniform(
        0.3, 1.0, intensity.shape
    )

    for censored, gamma, reference in (
        (True, 1.5, REFERENCE_CENSORED_G15),
        (False, 1.0, REFERENCE_DROP_G1),
    ):
        # When
        quantities = censored_quantities(
            intensity,
            library,
            precursor_index,
            N_PRECURSORS,
            censored=censored,
            correlation=correlation,
            quality_exponent=gamma,
        )

        # Then
        np.testing.assert_allclose(
            quantities[: len(reference)], reference, rtol=0, atol=REFERENCE_TOLERANCE
        )


def test_a_run_without_missing_fragments_does_not_spoil_the_other_runs():
    # Given a run where every well-measured fragment is observed (its dropout curve never censors) and one weak
    # precursor with missing fragments in that run only
    intensity, library, precursor_index, abundance = simulate(0.0, DETECTION_LIMIT)
    intensity[:, 0] = np.where(intensity[:, 0] > 0, intensity[:, 0], 1.0)
    weak = precursor_index == 0
    intensity[weak, 0] = np.r_[2.0**DETECTION_LIMIT, np.zeros(N_FRAGMENTS - 1)]
    intensity[weak, 1:] = np.exp2(abundance[0] + np.log2(0.5 / N_FRAGMENTS))

    # when
    quantities = censored_quantities(intensity, library, precursor_index, N_PRECURSORS)

    # then
    assert np.isfinite(quantities[0, 1:]).all()
    assert np.abs(quantities[0, 1:] - abundance[0]).max() < 2.0


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


def test_precursor_table_of_a_single_run_uses_the_plain_fit():
    # given: one run, where the extended fit has no fragments seen in several runs to estimate its noise curve from
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

import numpy as np
from alphabase.spectral_library.base import SpecLibBase

from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config


def init_spectral_library(
    config: Config,
    dia_cycle: np.ndarray,
    dia_rt_values: np.ndarray,
    reporter: Pipeline,
    spectral_library: SpecLibBase,
) -> None:
    """Initialize the spectral library.

    Parameters
    ----------
    config : Config
        Configuration object containing search and calibration parameters.
    dia_cycle : np.ndarray
        Array of DIA cycle values, used to determine the mz limits for filtering.
    dia_rt_values : np.ndarray
        Array of observed retention time values from the DIA data.
    reporter : Pipeline
        Reporter object for logging messages.
    spectral_library : SpecLibBase
        Spectral library object containing precursor information, will be modified in place.


    Normalizes the normalized retention time values form the spectral library to the observed RT values.
    Filters the spectral library based on the observed mz values.
    Optionally filters the spectral library to only contain precursors from selected channels (if set in config).

    Returns
    -------
    None
        The spectral library is modified in place:
            - precursor_df attribute is updated
            - precursor_df_unfiltered attribute is set to the original precursor dataframe.
    """
    # normalize RT
    spectral_library.precursor_df["rt_library"] = _norm_to_rt(
        config, dia_rt_values, spectral_library.precursor_df["rt_library"].values
    )

    # filter based on precursor observability
    lower_mz_limit = dia_cycle[dia_cycle > 0].min()
    upper_mz_limit = dia_cycle[dia_cycle > 0].max()

    n_precursor_before = np.sum(spectral_library.precursor_df["decoy"] == 0)
    spectral_library.precursor_df = spectral_library.precursor_df[
        (spectral_library.precursor_df["mz_library"] >= lower_mz_limit)
        & (spectral_library.precursor_df["mz_library"] <= upper_mz_limit)
    ]
    n_precursors_after = np.sum(spectral_library.precursor_df["decoy"] == 0)
    reporter.log_string(
        f"Initializing spectral library: {n_precursors_after:,} target precursors potentially observable ({n_precursor_before - n_precursors_after:,} removed)",
        verbosity="progress",
    )

    # filter spectral library to only contain precursors from allowed channels
    spectral_library.precursor_df_unfiltered = spectral_library.precursor_df.copy()

    if config["search"]["channel_filter"] != "":
        selected_channels = [
            int(c) for c in config["search"]["channel_filter"].split(",")
        ]

        spectral_library.precursor_df = spectral_library.precursor_df_unfiltered[
            spectral_library.precursor_df_unfiltered["channel"].isin(selected_channels)
        ].copy()

        reporter.log_string(
            f"Initializing spectral library: applied channel filter using only {selected_channels}, {len(spectral_library.precursor_df):,} precursors left",
        )


def _norm_to_rt(
    config: Config,
    dia_rt_values: np.ndarray,
    norm_values: np.ndarray,
    active_gradient_start: float | None = None,
    active_gradient_stop: float | None = None,
    mode: str | None = None,
) -> np.ndarray:
    """Convert normalized retention time values to absolute retention time values.

    Parameters
    ----------
    dia_rt_values :  np.ndarray
       RT values of the DIA data.

    norm_values : np.ndarray
        Array of normalized retention time values from the spectral library.

    active_gradient_start : float, optional
        Start of the active gradient in seconds, by default None.
        If None, it is set to the first retention time value plus half the configured `initial_rt_tolerance`.

    active_gradient_stop : float, optional
        End of the active gradient in seconds, by default None.
        If None, it is set to the last retention time value minus half the configured `initial_rt_tolerance`.

    mode : str, optional
        Mode of the gradient, by default None.
        If None, the value from the config is used which should be 'linear' by default

    Returns
    -------
    np.ndarray
        Array of absolute retention time values corresponding to the normalized values.
    """
    # TODO: "initial retention time tolerance in seconds if > 1, or a proportion of the total gradient length if < 1" not reflected here!
    # TODO: would expect the signs turned around?
    lower_rt = (
        (dia_rt_values[0] + config["search_initial"]["initial_rt_tolerance"] / 2)
        if active_gradient_start is None
        else active_gradient_start
    )

    upper_rt = (
        dia_rt_values[-1] - (config["search_initial"]["initial_rt_tolerance"] / 2)
        if active_gradient_stop is None
        else active_gradient_stop
    )

    # make sure values are really norm values
    norm_values = np.interp(norm_values, [norm_values.min(), norm_values.max()], [0, 1])

    # determine the mode based on the config or the function parameter
    mode = (
        config["calibration"].get("norm_rt_mode", "linear")
        if mode is None
        else mode.lower()
    )

    if mode == "linear":
        return np.interp(norm_values, [0, 1], [lower_rt, upper_rt])

    if mode == "tic":
        raise NotImplementedError("tic mode is not implemented yet")

    raise ValueError(f"Unknown norm_rt_mode {mode}")

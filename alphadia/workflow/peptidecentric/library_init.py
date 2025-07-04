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
    # apply channel filter
    if config["search"]["channel_filter"] == "":
        allowed_channels = spectral_library.precursor_df["channel"].unique()
    else:
        allowed_channels = [
            int(c) for c in config["search"]["channel_filter"].split(",")
        ]
        reporter.log_string(
            f"Applying channel filter using only: {allowed_channels}",
            verbosity="progress",
        )

    # normalize spectral library rt to file specific TIC profile
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
    # self.spectral_library.remove_unused_fragmen
    n_precursors_after = np.sum(spectral_library.precursor_df["decoy"] == 0)
    reporter.log_string(
        f"Initializing spectral library: {n_precursors_after:,} target precursors potentially observable ({n_precursor_before - n_precursors_after:,} removed)",
        verbosity="progress",
    )

    # filter spectral library to only contain precursors from allowed channels
    # save original precursor_df for later use
    spectral_library.precursor_df_unfiltered = spectral_library.precursor_df.copy()
    spectral_library.precursor_df = spectral_library.precursor_df_unfiltered[
        spectral_library.precursor_df_unfiltered["channel"].isin(allowed_channels)
    ].copy()


def _norm_to_rt(
    config: Config,
    dia_rt_values: np.ndarray,
    norm_values: np.ndarray,
    active_gradient_start: float | None = None,
    active_gradient_stop: float | None = None,
    mode=None,
) -> np.ndarray:
    """Convert normalized retention time values to absolute retention time values.

    Parameters
    ----------
    dia_rt_values :  np.ndarray
       RT values of the DIA data.

    norm_values : np.ndarray
        Array of normalized retention time values.

    active_gradient_start : float, optional
        Start of the active gradient in seconds, by default None.
        If None, the value from the config is used.
        If not defined in the config, it is set to zero.

    active_gradient_stop : float, optional
        End of the active gradient in seconds, by default None.
        If None, the value from the config is used.
        If not defined in the config, it is set to the last retention time value.

    mode : str, optional
        Mode of the gradient, by default None.
        If None, the value from the config is used which should be 'tic' by default

    """

    # determine if the gradient start and stop are defined in the config
    if active_gradient_start is None:
        if "active_gradient_start" in config["calibration"]:
            lower_rt = config["calibration"]["active_gradient_start"]
        else:
            lower_rt = (
                dia_rt_values[0] + config["search_initial"]["initial_rt_tolerance"] / 2
            )
    else:
        lower_rt = active_gradient_start

    if active_gradient_stop is None:
        if "active_gradient_stop" in config["calibration"]:
            upper_rt = config["calibration"]["active_gradient_stop"]
        else:
            upper_rt = dia_rt_values[-1] - (
                config["search_initial"]["initial_rt_tolerance"] / 2
            )
    else:
        upper_rt = active_gradient_stop

    # make sure values are really norm values
    norm_values = np.interp(norm_values, [norm_values.min(), norm_values.max()], [0, 1])

    # determine the mode based on the config or the function parameter
    if mode is None:
        mode = config["calibration"].get("norm_rt_mode", "tic")
    else:
        mode = mode.lower()

    if mode == "linear":
        return np.interp(norm_values, [0, 1], [lower_rt, upper_rt])

    elif mode == "tic":
        raise NotImplementedError("tic mode is not implemented yet")

    else:
        raise ValueError(f"Unknown norm_rt_mode {mode}")

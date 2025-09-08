import numpy as np
from alphabase.spectral_library.base import SpecLibBase

from alphadia.constants.keys import CalibCols
from alphadia.reporting.reporting import Pipeline


def init_spectral_library(
    dia_cycle: np.ndarray,
    dia_rt_values: np.ndarray,
    reporter: Pipeline,
    spectral_library: SpecLibBase,
    channel_filter: str | None = None,
) -> None:
    """Initialize the spectral library.

    Normalizes the normalized retention time values form the spectral library to the observed RT values.
    Filters the spectral library based on the observed mz values.
    Optionally filters the spectral library to only contain precursors from selected channels (if set in config).

    Parameters
    ----------
    dia_cycle : np.ndarray
        Array of DIA cycle values, used to determine the mz limits for filtering.
    dia_rt_values : np.ndarray
        Array of observed retention time values from the DIA data.
    reporter : Pipeline
        Reporter object for logging messages.
    spectral_library : SpecLibBase
        Spectral library object containing precursor information, will be modified in place.
    channel_filter : str, optional
        Comma-separated string of channel numbers (integers) to filter the spectral library (column "channel") by.
        If empty, no filtering is applied.

    Returns
    -------
    None
        The spectral library is modified in place:
            - precursor_df attribute is updated
            - precursor_df_unfiltered attribute is set to the original precursor dataframe.
    """
    # normalize RT
    spectral_library._precursor_df[CalibCols.RT_LIBRARY] = _norm_to_rt(
        dia_rt_values, spectral_library._precursor_df[CalibCols.RT_LIBRARY].values
    )

    # filter based on precursor observability
    lower_mz_limit = dia_cycle[dia_cycle > 0].min()
    upper_mz_limit = dia_cycle[dia_cycle > 0].max()

    # TODO using spectral_library.precursor_df (no underscore) here will trigger the setter method, which will additionally call refine_precursor_df()

    n_precursor_before = np.sum(spectral_library._precursor_df["decoy"] == 0)
    spectral_library._precursor_df = spectral_library._precursor_df[
        (spectral_library._precursor_df[CalibCols.MZ_LIBRARY] >= lower_mz_limit)
        & (spectral_library._precursor_df[CalibCols.MZ_LIBRARY] <= upper_mz_limit)
    ]
    n_precursors_after = np.sum(spectral_library._precursor_df["decoy"] == 0)
    reporter.log_string(
        f"Initializing spectral library: {n_precursors_after:,} target precursors potentially observable ({n_precursor_before - n_precursors_after:,} removed)",
        verbosity="progress",
    )

    # filter spectral library to only contain precursors from allowed channels
    spectral_library.precursor_df_unfiltered = spectral_library._precursor_df.copy()

    if channel_filter:
        selected_channels = [int(c) for c in channel_filter.split(",")]

        spectral_library._precursor_df = spectral_library.precursor_df_unfiltered[
            spectral_library.precursor_df_unfiltered["channel"].isin(selected_channels)
        ].copy()

        reporter.log_string(
            f"Initializing spectral library: applied channel filter using only {selected_channels}, {len(spectral_library._precursor_df):,} precursors left",
        )


def _norm_to_rt(
    dia_rt_values: np.ndarray,
    norm_values: np.ndarray,
) -> np.ndarray:
    """Convert normalized retention time values to absolute retention time values.

    Parameters
    ----------
    dia_rt_values :  np.ndarray
       RT values of the DIA data.

    norm_values : np.ndarray
        Array of normalized retention time values from the spectral library.

    Returns
    -------
    np.ndarray
        Array of absolute retention time values corresponding to the normalized values.
    """

    # make sure values are really norm values
    norm_values = np.interp(norm_values, [norm_values.min(), norm_values.max()], [0, 1])

    lower_rt = dia_rt_values[0]
    upper_rt = dia_rt_values[-1]

    return np.interp(norm_values, [0, 1], [lower_rt, upper_rt])

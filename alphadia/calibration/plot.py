"""Plotting functionality for the Calibration class."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from alphadia.plotting.utils import density_scatter

if TYPE_CHECKING:
    import pandas as pd

    from alphadia.calibration.estimator import CalibrationEstimator


def plot_calibration(
    calibration: CalibrationEstimator,
    df: pd.DataFrame,
    figure_path: str | None = None,
) -> None:
    """Plot the data and calibration model.

    Parameters
    ----------
    calibration : CalibrationEstimator
        Calibration object.

    df : pd.DataFrame
        Dataframe containing the input and target columns

    figure_path : str, default=None
        If set, the figure is saved to the given path.

    """
    deviation = calibration.calc_deviation(df)

    n_input_properties = deviation.shape[1] - 3
    input_property = None
    if n_input_properties <= 0:
        logging.warning("No input properties found for plotting calibration")
        return

    transform_unit = _get_transform_unit(calibration.transform_deviation)

    fig, axs = plt.subplots(
        n_input_properties,
        2,
        figsize=(6.5, 3.5 * n_input_properties),
        squeeze=False,
    )

    for input_property in range(n_input_properties):
        # plot the relative observed deviation
        density_scatter(
            deviation[:, 3 + input_property],
            deviation[:, 0],
            axis=axs[input_property, 0],
            s=1,
        )

        # plot the calibration model
        x_values = deviation[:, 3 + input_property]
        y_values = deviation[:, 1]
        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]

        axs[input_property, 0].plot(x_values, y_values, color="red")

        # plot the calibrated deviation

        density_scatter(
            deviation[:, 3 + input_property],
            deviation[:, 2],
            axis=axs[input_property, 1],
            s=1,
        )

        for ax, dim in zip(axs[input_property, :], [0, 2], strict=True):
            ax.set_xlabel(calibration.input_columns[input_property])
            ax.set_ylabel(f"observed deviation {transform_unit}")

            # get absolute y value and set limits to plus minus absolute y
            y = deviation[:, dim]
            y_abs = np.abs(y)
            ax.set_ylim(-y_abs.max() * 1.05, y_abs.max() * 1.05)

    fig.tight_layout()

    if figure_path is not None:
        figure_path_ = Path(figure_path)
        i = 0
        figure_file_path = (
            figure_path_
            / f"calibration_{calibration.input_columns[input_property]}_{i}.pdf"
        )

        while figure_file_path.exists():
            figure_file_path = (
                figure_path_
                / f"calibration_{calibration.input_columns[input_property]}_{i}.pdf"
            )

            i += 1
        fig.savefig(figure_file_path)
    else:
        plt.show()

    plt.close()


def _get_transform_unit(transform_deviation: None | float) -> str:
    """Get the unit of the deviation based on the transform deviation.

    Parameters
    ----------
    transform_deviation : typing.Union[None, float]
        If set to a valid float, the deviation is expressed as a fraction of the input value e.g. 1e6 for ppm.

    Returns
    -------
    str
        The unit of the deviation

    """
    if transform_deviation is not None:
        if np.isclose(transform_deviation, 1e6):
            return "(ppm)"
        if np.isclose(transform_deviation, 1e2):
            return "(%)"
        return f"({transform_deviation})"
    return "(absolute)"

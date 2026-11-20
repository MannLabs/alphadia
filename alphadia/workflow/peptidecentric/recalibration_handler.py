from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphadia.constants.settings import PeakGroupScoreCutoff
from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import (
    CalibrationGroups,
    CalibrationManager,
)


@dataclass(frozen=True)
class PeakParameters:
    """Peak widths and the minimum peak group score that candidate selection takes from the identifications."""

    fwhm_rt: float
    fwhm_mobility: float
    score_cutoff: float


def peak_parameters(
    identifications: pd.DataFrame, memory_saving: bool, cutoff_scale: float = 1.0
) -> PeakParameters:
    """Return the median peak widths of the identifications and the minimum peak group score.

    The score cutoff is a factor times a low percentile of the peak group scores of the identifications, see
    `PeakGroupScoreCutoff`, scaled by `cutoff_scale`. `memory_saving` takes the higher cutoff of
    `search.optimized_peak_group_score`.
    """
    if memory_saving:
        factor = PeakGroupScoreCutoff.MEMORY_SAVING_FACTOR
        percentile = PeakGroupScoreCutoff.MEMORY_SAVING_PERCENTILE
    else:
        factor = PeakGroupScoreCutoff.FACTOR
        percentile = PeakGroupScoreCutoff.PERCENTILE
    return PeakParameters(
        fwhm_rt=identifications["fwhm_rt"].median(),
        fwhm_mobility=identifications["fwhm_mobility"].median(),
        score_cutoff=cutoff_scale
        * factor
        * np.percentile(identifications["score"], percentile),
    )


class RecalibrationHandler:
    """Calibrates the library on identifications and derives the peak parameters of the next search."""

    def __init__(
        self,
        config: Config,
        calibration_manager: CalibrationManager,
        reporter: Pipeline,
        figure_path: str,
    ):
        self._config = config
        self._calibration_manager = calibration_manager
        self._reporter = reporter
        self._figure_path = figure_path

    def recalibrate(
        self, identifications: pd.DataFrame, fragments: pd.DataFrame
    ) -> PeakParameters:
        """Fit the MS1, MS2, RT and mobility calibration on the identifications and return their peak parameters.

        Parameters
        ----------
        identifications : pd.DataFrame
            Identified precursors, see `OptimizationHandler._filter_dfs`.

        fragments : pd.DataFrame
            Fragments of the identifications used for calibration, see `OptimizationHandler._filter_dfs`.

        """
        self._calibration_manager.fit(
            identifications,
            CalibrationGroups.PRECURSOR,
            figure_path=self._figure_path,
        )
        self._calibration_manager.fit(
            fragments,
            CalibrationGroups.FRAGMENT,
            figure_path=self._figure_path,
        )
        parameters = peak_parameters(
            identifications,
            self._config["search"]["optimized_peak_group_score"],
            self._config["search"]["peak_group_score_cutoff_scale"],
        )
        self._reporter.log_string(f"Using score_cutoff {parameters.score_cutoff}")
        return parameters

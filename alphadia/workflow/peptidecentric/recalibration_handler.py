import numpy as np
import pandas as pd

from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager


class RecalibrationHandler:
    """
    Handles recalibration of peptide-centric data.
    """

    def __init__(
        self,
        config: Config,
        reporter: Pipeline,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        figure_path: str,
        dia_data_has_ms1: bool,
    ):
        self._config = config
        self._reporter = reporter
        self._optimization_manager = optimization_manager
        self._calibration_manager = calibration_manager
        self._figure_path = figure_path
        self._dia_data_has_ms1 = dia_data_has_ms1

    def recalibrate(
        self, precursor_df_filtered: pd.DataFrame, fragments_df_filtered: pd.DataFrame
    ) -> None:
        """Performs recalibration of the MS1, MS2, RT and mobility properties. Also fits the convolution kernel and the score cutoff.
        The calibration manager is used to fit the data and predict the calibrated values.

        Parameters
        ----------
        precursor_df_filtered : pd.DataFrame
            Filtered precursor dataframe (see filter_dfs)

        fragments_df_filtered : pd.DataFrame
            Filtered fragment dataframe (see filter_dfs)

        """
        self._calibration_manager.fit(
            precursor_df_filtered,
            "precursor",
            figure_path=self._figure_path,
            skip=["mz"] if not self._dia_data_has_ms1 else [],
        )

        self._calibration_manager.fit(
            fragments_df_filtered,
            "fragment",
            figure_path=self._figure_path,
        )

        self._optimization_manager.fit(
            {
                "column_type": "calibrated",
                "num_candidates": self._config["search"]["target_num_candidates"],
            }
        )

        score = precursor_df_filtered["score"]
        if self._config["search"]["optimized_peak_group_score"]:
            # these values give benefits on max memory and runtime, with a small precursor penalty
            fac, q = 0.95, 3
        else:
            fac, q = 0.99, 1

        score_cutoff = fac * np.percentile(score, q)

        self._reporter.log_string(f"Using score_cutoff {score_cutoff} ({fac=}, {q=})")

        self._optimization_manager.fit(
            {
                "fwhm_rt": precursor_df_filtered["cycle_fwhm"].median(),
                "fwhm_mobility": precursor_df_filtered["mobility_fwhm"].median(),
                "score_cutoff": score_cutoff,
            }
        )

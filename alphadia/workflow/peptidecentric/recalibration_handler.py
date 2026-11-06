import numpy as np
import pandas as pd

from alphadia.reporting.reporting import Pipeline
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import (
    CalibrationGroups,
    CalibrationManager,
)
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.peptidecentric.candidate_prefilter import fit_prefilter


class RecalibrationHandler:
    """
    Handles recalibration of peptide-centric data.
    """

    DEFAULT_FAC = 0.95
    DEFAULT_Q = 3

    # The calibration identifications are the targets at 1 % FDR, so about 1 % of them are
    # false and fill the low-score tail: a 1st percentile is set by the false ones, drifts
    # between runs of the same file and, when low, floods the final search with low-score
    # candidates of which false targets pass far more often than decoys. The 3rd percentile
    # lies above that tail.
    # The percentile falls the longer the calibration classifier trains on the same data (about
    # 0.3 score units per further 10-epoch step), so the factor sets the cutoff that a long
    # calibration used to reach instead of the number of calibration steps.
    OPTIMIZED_FAC = 0.85
    OPTIMIZED_Q = 3

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        reporter: Pipeline,
        figure_path: str,
        dia_data_has_ms1: bool,
    ):
        self._config = config
        self._optimization_manager = optimization_manager
        self._calibration_manager = calibration_manager
        self._reporter = reporter
        self._figure_path = figure_path
        self._dia_data_has_ms1 = dia_data_has_ms1

    def recalibrate(
        self,
        precursor_df_filtered: pd.DataFrame,
        fragments_df_filtered: pd.DataFrame,
        candidates_df: pd.DataFrame | None = None,
    ) -> None:
        """Performs recalibration of the MS1, MS2, RT and mobility properties and fits the convolution kernel and the score cutoff.

        The calibration manager is used to fit the data and predict the calibrated values.

        Parameters
        ----------
        precursor_df_filtered : pd.DataFrame
            Filtered precursor dataframe (see filter_dfs)

        fragments_df_filtered : pd.DataFrame
            Filtered fragment dataframe (see filter_dfs)

        candidates_df : pd.DataFrame, optional
            The scored candidates of the calibration batches (targets and decoys). With the selection features of the
            rust backend, the linear candidate prefilter is fitted on them and replaces the raw-score cutoff.

        """
        self._calibration_manager.fit(
            precursor_df_filtered,
            CalibrationGroups.PRECURSOR,
            figure_path=self._figure_path,
        )

        self._calibration_manager.fit(
            fragments_df_filtered,
            CalibrationGroups.FRAGMENT,
            figure_path=self._figure_path,
        )

        self._optimization_manager.update(
            num_candidates=self._config["search"]["target_num_candidates"],
        )

        score = precursor_df_filtered["score"]
        if self._config["search"]["optimized_peak_group_score"]:
            # these values give benefits on max memory and runtime, with a small precursor penalty
            fac, q = self.DEFAULT_FAC, self.DEFAULT_Q
        else:
            fac, q = self.OPTIMIZED_FAC, self.OPTIMIZED_Q

        score_cutoff = fac * np.percentile(score, q)

        self._reporter.log_string(f"Using score_cutoff {score_cutoff} ({fac=}, {q=})")

        prefilter = None
        if candidates_df is not None:
            kept_share = float((score > score_cutoff).mean())
            prefilter = fit_prefilter(candidates_df, precursor_df_filtered, kept_share)
            if prefilter is not None:
                kept = prefilter.keep(candidates_df)
                self._reporter.log_string(
                    f"Candidate prefilter replaces the score cutoff: coefficients {prefilter.coef} on standardized "
                    f"score / detected library share / detected fragments, threshold {prefilter.threshold:.3f}, keeps "
                    f"{kept_share:.3f} of the calibration identifications and {kept.mean():.3f} of the calibration "
                    f"candidates (score cutoff {(candidates_df['score'] > score_cutoff).mean():.3f})"
                )

        self._optimization_manager.update(
            fwhm_rt=precursor_df_filtered["fwhm_rt"].median(),
            fwhm_mobility=precursor_df_filtered["fwhm_mobility"].median(),
            score_cutoff=score_cutoff,
            candidate_prefilter=prefilter,
        )

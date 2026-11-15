from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

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


def decoy_probit_score_cutoff(
    identification_scores: np.ndarray, decoy_scores: np.ndarray, accepted_loss: float
) -> float | None:
    """Estimate the peak group score below which `accepted_loss` of the identifications of the main search lie.

    The identifications of a calibration are the stronger part of those of the main search, so a low percentile of
    their scores cuts too high. On the scale of the decoy scores of the same search their distribution is close to
    normal: z = Phi^-1(F_decoy(score)). The cutoff is the decoy score at the `accepted_loss` quantile of a normal fitted
    to z, which extrapolates below the weakest identifications.

    Returns None without decoys or with fewer than two identifications.
    """
    identification_scores = np.asarray(identification_scores, dtype=np.float64)
    decoy_scores = np.sort(np.asarray(decoy_scores, dtype=np.float64))
    n_decoys = len(decoy_scores)
    if n_decoys == 0 or len(identification_scores) < 2:
        return None
    decoy_cdf = np.searchsorted(decoy_scores, identification_scores, side="right")
    # The +1 keeps the CDF of identifications above all decoys below 1.
    z = stats.norm.ppf(
        np.clip(
            decoy_cdf / (n_decoys + 1), 1 / (n_decoys + 1), n_decoys / (n_decoys + 1)
        )
    )
    cutoff_cdf = stats.norm.cdf(z.mean() + z.std() * stats.norm.ppf(accepted_loss))
    return float(np.quantile(decoy_scores, cutoff_cdf))


def peak_parameters(
    identifications: pd.DataFrame,
    memory_saving: bool,
    decoy_scores: np.ndarray | None = None,
) -> PeakParameters:
    """Return the median peak widths of the identifications and the minimum peak group score.

    With `decoy_scores`, the decoy scores of the search of the identifications, the score cutoff is the decoy-probit
    estimate for `PeakGroupScoreCutoff.ACCEPTED_LOSS`, see `decoy_probit_score_cutoff`. Otherwise, and with
    `memory_saving`, the higher cutoff of `search.optimized_peak_group_score`, it is a factor times a low percentile of
    the peak group scores of the identifications.
    """
    if memory_saving:
        factor = PeakGroupScoreCutoff.MEMORY_SAVING_FACTOR
        percentile = PeakGroupScoreCutoff.MEMORY_SAVING_PERCENTILE
    else:
        factor = PeakGroupScoreCutoff.FACTOR
        percentile = PeakGroupScoreCutoff.PERCENTILE
    score_cutoff = factor * np.percentile(identifications["score"], percentile)
    if decoy_scores is not None and not memory_saving:
        estimate = decoy_probit_score_cutoff(
            identifications["score"].to_numpy(),
            decoy_scores,
            PeakGroupScoreCutoff.ACCEPTED_LOSS,
        )
        if estimate is not None:
            score_cutoff = estimate
    return PeakParameters(
        fwhm_rt=identifications["fwhm_rt"].median(),
        fwhm_mobility=identifications["fwhm_mobility"].median(),
        score_cutoff=score_cutoff,
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
        self,
        identifications: pd.DataFrame,
        fragments: pd.DataFrame,
        decoy_scores: np.ndarray | None = None,
    ) -> PeakParameters:
        """Fit the MS1, MS2, RT and mobility calibration on the identifications and return their peak parameters.

        Parameters
        ----------
        identifications : pd.DataFrame
            Identified precursors, see `OptimizationHandler._filter_dfs`.

        fragments : pd.DataFrame
            Fragments of the identifications used for calibration, see `OptimizationHandler._filter_dfs`.

        decoy_scores : np.ndarray | None
            Peak group scores of the decoy candidates of the search of the identifications, see `peak_parameters`.

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
            decoy_scores,
        )
        self._reporter.log_string(f"Using score_cutoff {parameters.score_cutoff}")
        return parameters

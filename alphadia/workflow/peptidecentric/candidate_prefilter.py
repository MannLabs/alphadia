"""Linear prefilter of the selected candidates before scoring.

The selection score is an intensity-scaled sum over fragments, so a cutoff on it alone removes faint peptides whose
fragments are mostly present at the apex. The prefilter instead ranks candidates by a logistic regression on the
selection score and the fragment detection at the apex (both computed for every selected candidate by the rust
selection), fitted at the end of calibration on the calibration identifications against the decoy candidates of the
same batches. Its threshold keeps the same share of the calibration identifications as the raw-score cutoff did, so
it changes which candidates pass rather than how many of the known good ones.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SELECTION_FEATURES = ["score", "selection_det_share", "selection_n_detected"]
# a degenerate feature (constant over the training rows) keeps a unit scale instead of dividing by zero
MIN_SCALE = 1e-6
# decoy candidates of the calibration batches used to fit; more add nothing to a three-coefficient model
MAX_DECOYS = 200_000
MIN_ROWS_PER_CLASS = 50


@dataclass
class CandidatePrefilter:
    """A fitted linear prefilter: keep candidates whose standardized linear score exceeds `threshold`."""

    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    threshold: float

    def linear_score(self, candidates_df: pd.DataFrame) -> np.ndarray:
        x = (
            candidates_df[SELECTION_FEATURES].to_numpy(np.float64) - self.mean
        ) / self.scale
        return x @ np.asarray(self.coef) + self.intercept

    def keep(self, candidates_df: pd.DataFrame) -> np.ndarray:
        return self.linear_score(candidates_df) > self.threshold


def fit_prefilter(
    candidates_df: pd.DataFrame,
    identifications_df: pd.DataFrame,
    kept_share: float,
    random_state: int | None = None,
) -> CandidatePrefilter | None:
    """Fit the prefilter on the identified targets against the decoy candidates.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        The scored candidates of the calibration batches, targets and decoys, with `SELECTION_FEATURES`, `decoy`,
        `precursor_idx` and `rank`.
    identifications_df : pd.DataFrame
        The calibration identifications (targets at the calibration FDR), with `precursor_idx` and `rank`.
    kept_share : float
        The share of the identifications the threshold keeps.
    random_state : int, optional
        Seed of the decoy subsample.

    Returns
    -------
    CandidatePrefilter | None
        None when a class has too few rows or the selection features are missing.

    """
    if not set(SELECTION_FEATURES) <= set(candidates_df.columns):
        return None
    identified = candidates_df.merge(
        identifications_df[["precursor_idx", "rank"]].drop_duplicates(),
        on=["precursor_idx", "rank"],
    )
    identified = identified[identified["decoy"] == 0]
    decoys = candidates_df[candidates_df["decoy"] == 1]
    if len(decoys) > MAX_DECOYS:
        decoys = decoys.sample(MAX_DECOYS, random_state=random_state)
    if min(len(identified), len(decoys)) < MIN_ROWS_PER_CLASS:
        return None

    x = np.vstack(
        [
            identified[SELECTION_FEATURES].to_numpy(np.float64),
            decoys[SELECTION_FEATURES].to_numpy(np.float64),
        ]
    )
    y = np.r_[np.ones(len(identified)), np.zeros(len(decoys))]
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), MIN_SCALE)
    model = LogisticRegression(class_weight="balanced").fit((x - mean) / scale, y)
    prefilter = CandidatePrefilter(
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]),
        threshold=0.0,
    )
    prefilter.threshold = float(
        np.quantile(prefilter.linear_score(identified), 1.0 - kept_share)
    )
    return prefilter

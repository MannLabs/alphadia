"""Decoy weights that make the decoys look like the false targets.

Target-decoy competition assumes that false targets and decoys are exchangeable. The decoys
carry a signature of their generation instead: on HeLa a stage-1 score tells known-false
(entrapment) targets from decoys with an AUC of 0.58, and 0.72 on the gated set. A flexible
classifier learns that signature on top of the true-versus-false boundary. Candidates of
rank 2 and above are false whatever the sample, so a model of P(target | x) fitted on them
estimates the density ratio of false targets to decoys; weighting every decoy by it makes
the weighted decoys match the false targets in feature space (importance weighting under
covariate shift, Shimodaira 2000).
"""

import logging

import numpy as np

from alphadia.fdr.classifiers import LightGBMClassifier

logger = logging.getLogger()

# candidates from this rank on are false matches whatever the sample
JUNK_RANK = 2
# density ratios beyond this are more noise than signal
_MAX_WEIGHT = 5.0
_N_FOLDS = 2
_MIN_JUNK_ROWS = 1_000
_LGBM_PARAMS = {
    "n_estimators": 200,
    "final_n_estimators": 200,
    "num_leaves": 15,
    "learning_rate_start": 0.1,
    "learning_rate_end": 0.1,
    "learning_rate_decay_rounds": 1,
}


def decoy_weights(
    x: np.ndarray,
    y: np.ndarray,
    rank: np.ndarray,
    random_state: int | None = None,
    num_threads: int = 1,
) -> np.ndarray:
    """Return one weight per row: 1 for targets, the estimated density ratio for decoys.

    Parameters
    ----------
    x : np.ndarray
        Features of shape (n_samples, n_features).

    y : np.ndarray
        Decoy labels, 1 for decoys.

    rank : np.ndarray
        Candidate rank of every row, 0 for the best candidate of a precursor.

    random_state : int, optional
        Seed of the fold split and the models.

    num_threads : int, default=1
        Threads for the LightGBM fits.

    """
    is_decoy = y == 1
    junk = rank >= JUNK_RANK
    weight = np.ones(len(y))
    if (junk & is_decoy).sum() < _MIN_JUNK_ROWS or (
        junk & ~is_decoy
    ).sum() < _MIN_JUNK_ROWS:
        logger.warning(
            "Too few rank>=2 candidates to estimate decoy weights; weighting every decoy the same"
        )
        return weight

    rng = np.random.default_rng(random_state)
    fold = rng.integers(0, _N_FOLDS, size=len(y))
    # out-of-fold on the junk rows the models were fitted on, the fold average elsewhere
    p_decoy_junk = np.zeros(len(y))
    p_decoy_rest = np.zeros(len(y))
    for k in range(_N_FOLDS):
        model = LightGBMClassifier(
            **_LGBM_PARAMS,
            num_threads=num_threads,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        train = junk & (fold != k)
        model.fit(x[train], y[train])
        p = model.predict_proba(x)[:, 1]
        held_out = junk & (fold == k)
        p_decoy_junk[held_out] = p[held_out]
        p_decoy_rest += p / _N_FOLDS
    p_decoy = np.where(junk, p_decoy_junk, p_decoy_rest)

    # P(target | x) / P(decoy | x) on false rows is the density ratio up to the class ratio,
    # which the normalisation to a mean weight of 1 removes
    ratio = np.clip(
        (1 - p_decoy) / np.clip(p_decoy, 1e-6, None), 1 / _MAX_WEIGHT, _MAX_WEIGHT
    )
    weight[is_decoy] = ratio[is_decoy] / ratio[is_decoy].mean()
    q = np.quantile(weight[is_decoy], [0.05, 0.5, 0.95])
    logger.info(
        f"Decoy weights from {int((junk & ~is_decoy).sum()):,} rank>={JUNK_RANK} targets and "
        f"{int((junk & is_decoy).sum()):,} decoys: 5/50/95 % quantiles {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}"
    )
    return weight

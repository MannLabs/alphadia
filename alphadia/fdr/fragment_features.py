"""Final-round classifier features from the dense fragment arrays that the riding removal quantifies."""

import numpy as np
import pandas as pd

# The XIC correlation of a candidate's fragments with its profile, averaged with the fragments' library intensity
# shares and with their observed intensity shares as weights. A real peptide's correlating fragments are the ones its
# library predicts strong and the ones that carry its signal.
LIBRARY_WEIGHTED = "xic_libweighted"
OBSERVED_WEIGHTED = "xic_obsweighted"
FEATURES = (LIBRARY_WEIGHTED, OBSERVED_WEIGHTED)


def _shares(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    v = np.where(present, values, 0.0).astype(np.float64)
    total = v.sum(1, keepdims=True)
    return np.divide(v, total, out=np.zeros_like(v), where=total > 0)


def weighted_correlations(
    mz: np.ndarray, library: np.ndarray, observed: np.ndarray, correlation: np.ndarray
) -> dict[str, np.ndarray]:
    """FEATURES of each row of the (rows, fragments) library m/z (0: empty slot), library intensity, observed
    intensity and correlation arrays; 0 for a row without observed signal."""
    present = mz > 0
    corr = np.where(present, correlation, 0.0)
    return {
        LIBRARY_WEIGHTED: (_shares(library, present) * corr).sum(1).astype(np.float32),
        OBSERVED_WEIGHTED: (_shares(observed, present) * corr)
        .sum(1)
        .astype(np.float32),
    }


def serving_provider(candidate_idx: np.ndarray, fragments: tuple[np.ndarray, ...]):
    """A riding provider that serves the rows of already quantified `fragments` (in the order of `candidate_idx`) to any
    PSM frame by its `_candidate_idx`."""
    index = pd.Index(candidate_idx)

    def provider(psm_df: pd.DataFrame) -> tuple[np.ndarray, ...]:
        rows = index.get_indexer(psm_df["_candidate_idx"].to_numpy())
        if (rows < 0).any():
            raise KeyError("PSMs without quantified fragments")
        return tuple(a[rows] for a in fragments)

    return provider

"""Utility functions for FDR classification tasks."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from alphadia.exceptions import TooFewPSMError

logger = logging.getLogger()

# Torch threads for the FDR classifier. The network is small: a training step on 4096 rows saturates
# at about 8 threads, one on 16384 rows (the batch size above 4 M rows) at about 16. Measured on a
# hybrid 8+16-core laptop, where 24 threads were slower again.
MAX_FDR_TORCH_THREADS = 16

# The raw selection score of a candidate minus the best score among the other candidates of its precursor. The raw
# score depends on how much signal a precursor collects anywhere, which varies between precursors, and between targets
# and decoys of a decoy scheme. Its gap to the precursor's own next-best peak does not. On the pass-1 dumps of the
# paired-entrapment runs this gave +2-5 % real IDs at a matched paired FDP, with the label-free leak AUC up by
# at most 0.004.
SCORE_TO_NEXT_CANDIDATE = "score_to_next_candidate"


def train_test_split_(
    X: np.ndarray,
    y: np.ndarray,
    *,
    exception: type[Exception] = TooFewPSMError,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper around `sklearn.model_selection.train_test_split` to handle exceptions.

    Parameters
    ----------
    X : np.ndarray
        The input features.
    y : np.ndarray
        The target values.
    exception : type[Exception], default=TooFewPSMError
        The exception to raise if train_test_split fails.

    **kwargs
        Additional arguments passed to sklearn's train_test_split.

    Returns
    -------
    X_train, X_test, y_train, y_test, indices_train, indices_test : np.ndarray
        The split data, including the indices.

    """
    try:
        indices = np.arange(len(X))

        X_train, X_test, y_train, y_test, indices_train, indices_test = (
            train_test_split(X, y, indices, **kwargs)
        )
    except ValueError as e:
        raise exception(str(e)) from e
    else:
        return X_train, X_test, y_train, y_test, indices_train, indices_test


def train_test_split_indices(
    n_samples: int,
    *,
    exception: type[Exception] = TooFewPSMError,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the split `train_test_split_` makes for `n_samples` rows with the same arguments.

    Parameters
    ----------
    n_samples : int
        The number of rows.
    exception : type[Exception], default=TooFewPSMError
        The exception to raise if train_test_split fails.

    **kwargs
        Additional arguments passed to sklearn's train_test_split.

    Returns
    -------
    indices_train, indices_test : np.ndarray
        The row indices of the training and the test set.

    """
    try:
        indices_train, indices_test = train_test_split(np.arange(n_samples), **kwargs)
    except ValueError as e:
        raise exception(str(e)) from e
    else:
        return indices_train, indices_test


def manage_torch_threads(max_threads: int = 2) -> Callable[..., Any]:
    """Decorator to manage torch thread count during method execution.

    Parameters
    ----------
    max_threads : int, default=2
        Maximum number of threads to use during method execution

    Returns
    -------
    Callable
        Decorated function that manages torch thread count

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            is_threads_changed = False
            original_threads = torch.get_num_threads()

            # Restrict threads if needed
            if original_threads > max_threads:
                torch.set_num_threads(max_threads)
                is_threads_changed = True
                logger.info(
                    f"Setting torch num_threads to {max_threads} for FDR classification task"
                )

            try:
                # Execute the wrapped function
                return func(*args, **kwargs)
            finally:
                # Reset threads if we changed them
                if is_threads_changed:
                    logger.info(f"Resetting torch num_threads to {original_threads}")
                    torch.set_num_threads(original_threads)

        return wrapper

    return decorator


def add_score_to_next_candidate(df: pd.DataFrame) -> None:
    """Add `SCORE_TO_NEXT_CANDIDATE` in place: `score` minus the best `score` among the other candidates of the precursor.

    The best candidate is compared with the runner-up, every other candidate with the best one, so the value is
    positive only for the best candidate. Tied best candidates both get 0. A precursor with a single candidate is
    compared with the lowest score of the frame, as its other peaks fell below the raw-score cutoff.
    """
    score = df["score"].to_numpy(dtype=np.float64)
    precursor = df["precursor_idx"].to_numpy()
    if len(score) == 0:
        df[SCORE_TO_NEXT_CANDIDATE] = np.zeros(0, dtype=np.float32)
        return

    order = np.lexsort((-score, precursor))
    sorted_precursor, sorted_score = precursor[order], score[order]
    is_first = np.r_[True, sorted_precursor[1:] != sorted_precursor[:-1]]
    group = np.cumsum(is_first) - 1
    best = sorted_score[is_first][group]

    has_runner_up = (
        is_first & np.r_[sorted_precursor[1:] == sorted_precursor[:-1], False]
    )
    runner_up = np.full(group[-1] + 1, np.nanmin(score))
    runner_up[group[has_runner_up]] = sorted_score[np.flatnonzero(has_runner_up) + 1]

    other = np.where(is_first, runner_up[group], best)
    relative = np.empty(len(score), dtype=np.float32)
    relative[order] = sorted_score - other
    df[SCORE_TO_NEXT_CANDIDATE] = relative

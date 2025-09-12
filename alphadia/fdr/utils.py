"""Utility functions for FDR classification tasks."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from alphadia.exceptions import TooFewPSMError

logger = logging.getLogger()


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

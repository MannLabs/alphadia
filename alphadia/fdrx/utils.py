"""Utility functions for FDR classification tasks."""

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger()


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

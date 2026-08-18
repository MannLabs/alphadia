"""Some utility functions for the whole alphadia package."""

import logging
import os
import platform
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:  # not available on windows
    resource = None

logger = logging.getLogger()


def get_peak_memory_gb() -> float:
    """Return the peak resident set size of this process in GB, 0.0 if unavailable."""
    if resource is None:
        return 0.0

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # linux reports kilobytes, macOS bytes
    return max_rss / 1e6 if platform.system() == "Linux" else max_rss / 1e9


def log_lfq_step(message: str) -> None:
    """Log a step of the label-free quantification, tagged for grepping."""
    logger.info(f"[lfq] {message} | peak memory {get_peak_memory_gb():.2f} GB")


USE_NUMBA_CACHING = os.environ.get("USE_NUMBA_CACHING", "0") == "1"


def expand_path(path: str) -> str | None:
    """Expand ~ in a path to the user's home directory."""

    if path is None:
        return path

    return str(Path(path).expanduser())


def get_torch_device(use_gpu: bool = False):
    """Get the torch device to be used.

    Parameters
    ----------

    use_gpu : bool, optional
        If True, use GPU if available, by default False

    Returns
    -------
    str
        Device to be used, either 'cpu', 'gpu' or 'mps'

    """
    import torch  # deliberately importing lazily to decouple utils from the heavy torch dependency

    device = "cpu"
    if use_gpu:
        if platform.system() == "Darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = "gpu" if torch.cuda.is_available() else "cpu"

    logger.info(f"Device set to {device}")

    return device


# TODO find a better place for this
def get_isotope_columns(colnames):
    isotopes = []
    for col in colnames:
        if col[:2] == "i_":
            try:
                isotopes.append(int(col[2:]))
            except Exception:
                logging.warning(
                    f"Column {col} does not seem to be a valid isotope column"
                )

    isotopes = np.array(sorted(isotopes))

    if not np.all(np.diff(isotopes) == 1):
        logging.warning("Isotopes are not consecutive")

    return isotopes

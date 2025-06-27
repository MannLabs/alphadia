import logging
import os
import platform

import numba as nb
import numpy as np

logger = logging.getLogger()


ISOTOPE_DIFF = 1.0032999999999674

USE_NUMBA_CACHING = os.environ.get("USE_NUMBA_CACHING", "0") == "1"


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


@nb.njit(cache=USE_NUMBA_CACHING)
def candidate_hash(precursor_idx, rank):
    # create a 64 bit hash from the precursor_idx, number and type
    # the precursor_idx is the lower 32 bits
    # the rank is the next 8 bits
    return precursor_idx + (rank << 32)


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

"""Some utility functions for the whole alphadia package."""

import logging
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger()


USE_NUMBA_CACHING = os.environ.get("USE_NUMBA_CACHING", "0") == "1"

# alphabase's position of the monoisotopic peak among the i_* columns
MONO_ISOTOPE_INDEX_COLUMN = "mono_isotope_idx"


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


def isotopes_from_monoisotope(
    precursor_df: pd.DataFrame, isotope_columns: list[str]
) -> np.ndarray:
    """Isotope abundances with the monoisotopic peak first, as rust expects them.

    alphabase may start the i_* columns below the monoisotopic peak (e.g. for heavy labels); its position is
    in `mono_isotope_idx`. Positions beyond the computed isotopes are 0.
    """
    isotopes = precursor_df[isotope_columns].to_numpy(dtype=np.float32)
    if MONO_ISOTOPE_INDEX_COLUMN not in precursor_df.columns:
        return np.ascontiguousarray(isotopes)
    mono_idx = precursor_df[MONO_ISOTOPE_INDEX_COLUMN].to_numpy()
    shifted = np.zeros_like(isotopes)
    for offset in np.unique(mono_idx):
        rows = mono_idx == offset
        shifted[rows, : isotopes.shape[1] - offset] = isotopes[rows, offset:]
    return shifted

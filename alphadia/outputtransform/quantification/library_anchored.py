"""Library-anchored precursor quantification.

Converts the fragment matrices to the one-row-per-run matrices of
`alphadia_search_rs.LibraryAnchoredFit`, which defines the model, the fit and the derivation
of its loss scale and offset shrinkage from the data.
"""

import logging

import numpy as np
import pandas as pd
from alphadia_search_rs import LibraryAnchoredFit

logger = logging.getLogger(__name__)


def library_anchored_estimate(
    intensity: np.ndarray,
    library: np.ndarray,
    group: np.ndarray,
    **fit_params,
) -> tuple[np.ndarray, np.ndarray]:
    """Group quantities from linear fragment intensities and their library intensities.

    Parameters
    ----------
    intensity : np.ndarray
        Fragments x runs linear intensities, 0 or NaN where missing
    library : np.ndarray
        Library intensity per fragment row, 0 or NaN where unknown (the fragment is ignored)
    group : np.ndarray
        Group id (e.g. precursor) per fragment row
    **fit_params
        Keyword arguments of `LibraryAnchoredFit`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted unique group ids and the log2 quantities (groups x runs), NaN where undefined
    """
    library = np.nan_to_num(np.asarray(library, dtype=np.float32))
    n_without_library = int((library <= 0).sum())
    if n_without_library:
        logger.warning(
            f"{n_without_library:,} fragment ions without library intensity are ignored"
        )
    group_codes, groups = pd.factorize(group, sort=True)

    result = LibraryAnchoredFit(**fit_params).fit(
        np.ascontiguousarray(np.nan_to_num(intensity).T, dtype=np.float32),
        library,
        group_codes.astype(np.uintp),
        len(groups),
    )
    logger.info(
        f"Library-anchored fit: residual scale {result['residual_scale']:.3f} log2, "
        f"loss scale {result['loss_scale']:.2f}, "
        f"offset shrinkage {result['offset_shrinkage']:.2f}"
    )
    return np.asarray(groups), result["quantity"].T

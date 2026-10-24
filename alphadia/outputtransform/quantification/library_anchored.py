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

# keys of the result returned by LibraryAnchoredFit.fit
_RESULT_QUANTITY = "quantity"
_RESULT_RESIDUAL_SCALE = "residual_scale"
_RESULT_LOSS_SCALE = "loss_scale"
_RESULT_OFFSET_SHRINKAGE = "offset_shrinkage"


def library_anchored_estimate(
    intensity: np.ndarray,
    library: np.ndarray,
    group: np.ndarray,
    tau: float | None = None,
    lam: float | None = None,
    max_sweeps: int | None = None,
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
    tau : float | None
        Scale of the Cauchy loss in log2 units, None to derive it from the data
    lam : float | None
        Shrinkage of the fragment offsets towards the library pattern, None to derive it from the data
    max_sweeps : int | None
        Maximum number of reweighting sweeps per group, None for the default of `LibraryAnchoredFit`

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

    fit_kwargs = {} if max_sweeps is None else {"max_sweeps": max_sweeps}
    result = LibraryAnchoredFit(tau=tau, lam=lam, **fit_kwargs).fit(
        np.ascontiguousarray(np.nan_to_num(intensity).T, dtype=np.float32),
        library,
        group_codes.astype(np.uintp),
        len(groups),
    )
    logger.info(
        f"Library-anchored fit: residual scale {result[_RESULT_RESIDUAL_SCALE]:.3f} log2, "
        f"loss scale {result[_RESULT_LOSS_SCALE]:.2f}, "
        f"offset shrinkage {result[_RESULT_OFFSET_SHRINKAGE]:.2f}"
    )
    return np.asarray(groups), result[_RESULT_QUANTITY].T

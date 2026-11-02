"""Final-round classifier features from the top library fragments of each PSM.

Every PSM carries its TOP_K_FRAGMENTS most intense library fragments (no y1, m/z 0 = empty slot) with library
intensity, observed intensity, XIC correlation to the candidate's profile, ion type, ion number and fragment charge.
A decoy's slots are its target's (copied intensities, ion types and numbers) with the m/z of the mutated sequence, so
the features describe the evidence of a target and of its decoy alike.
"""

import numpy as np

# the fragments the backend quantifies for the features: as many as the rust scoring scores
TOP_K_FRAGMENTS = 12

# The XIC correlation of a candidate's fragments with its profile, averaged with the fragments' library intensity
# shares and with their observed intensity shares as weights. A real peptide's correlating fragments are the ones its
# library predicts strong and the ones that carry its signal.
LIBRARY_WEIGHTED = "xic_libweighted"
OBSERVED_WEIGHTED = "xic_obsweighted"
ION_COHERENT = tuple(
    f"ion_coherent_{t}_{size}" for t in ("b", "y") for size in ("short", "mid", "long")
)
FEATURES = (
    LIBRARY_WEIGHTED,
    OBSERVED_WEIGHTED,
    "ion_fisher",
    "ion_fisher_lib",
    "ion_coherent_2plus",
    *ION_COHERENT,
    "ent_sim",
    "ent_sim_coherent",
    *(f"z2_{name}" for name in ("libw_corr", "explained", "n_over_50")),
)

B_ION, Y_ION = ord("b"), ord("y")
# a fragment co-elutes with its candidate above this profile correlation
COHERENT_CORRELATION = 0.5
# ion classes of the noise model: ion type x number bucket x fragment charge x group of SLOT_GROUP library slots
NULL_NUMBER_EDGES = (3, 5, 8)
SLOT_GROUP = 3
MAX_FRAGMENT_CHARGE = 2
COUNT_NUMBER_EDGES = (4, 8)
CORRELATION_BINS = np.linspace(0.0, 1.0, 101)
# the chance of a hit is floored here: one fragment adds at most -log(MIN_CHANCE) surprise
MIN_CHANCE = 1e-4


def _shares(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    v = np.where(present, values, 0.0).astype(np.float64)
    total = v.sum(1, keepdims=True)
    return np.divide(v, total, out=np.zeros_like(v), where=total > 0)


def weighted_correlations(
    mz: np.ndarray, library: np.ndarray, observed: np.ndarray, correlation: np.ndarray
) -> dict[str, np.ndarray]:
    """LIBRARY_WEIGHTED and OBSERVED_WEIGHTED of each row; 0 for a row without observed signal."""
    present = mz > 0
    corr = np.where(present, correlation, 0.0)
    return {
        LIBRARY_WEIGHTED: (_shares(library, present) * corr).sum(1),
        OBSERVED_WEIGHTED: (_shares(observed, present) * corr).sum(1),
    }


def _ion_class(
    ion_type: np.ndarray, number: np.ndarray, charge: np.ndarray
) -> np.ndarray:
    kind = np.where(ion_type == B_ION, 0, np.where(ion_type == Y_ION, 1, 2))
    bucket = np.searchsorted(np.array(NULL_NUMBER_EDGES), number, side="right")
    slot_group = np.broadcast_to(
        np.arange(ion_type.shape[1]) // SLOT_GROUP, ion_type.shape
    )
    n_groups = int(np.ceil(ion_type.shape[1] / SLOT_GROUP))
    fragment_charge = np.clip(charge, 0, MAX_FRAGMENT_CHARGE)
    return (
        (
            (kind * (len(NULL_NUMBER_EDGES) + 1) + bucket) * (MAX_FRAGMENT_CHARGE + 1)
            + fragment_charge
        )
        * n_groups
        + slot_group
    ).astype(np.int64)


def ion_features(  # noqa: PLR0913 # one array per fragment attribute
    present: np.ndarray,
    hit: np.ndarray,
    correlation: np.ndarray,
    library_share: np.ndarray,
    ion_type: np.ndarray,
    number: np.ndarray,
    charge: np.ndarray,
    null_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fisher sums of each fragment's chance under a label-blind noise model, and coherent counts per ion class.

    The noise model of each ion class is fitted on the rows `null_rows`, the rank >= 1 candidates of both classes.
    """
    cls = _ion_class(ion_type, number, charge)
    n_classes = int(cls.max()) + 1
    c, h, r = (
        cls[null_rows][present[null_rows]],
        hit[null_rows][present[null_rows]],
        correlation[null_rows][present[null_rows]],
    )
    hit_rate = np.bincount(c, weights=h, minlength=n_classes) / np.maximum(
        np.bincount(c, minlength=n_classes), 1
    )
    positive = h & (r > 0)
    last_bin = len(CORRELATION_BINS) - 2
    bins = np.clip(
        np.searchsorted(CORRELATION_BINS, r[positive], side="right") - 1, 0, last_bin
    )
    counts = np.zeros((n_classes, last_bin + 1))
    np.add.at(counts, (c[positive], bins), 1)
    survival = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1] / np.maximum(
        counts.sum(1, keepdims=True), 1
    )
    row_bins = np.clip(
        np.searchsorted(CORRELATION_BINS, correlation, side="right") - 1, 0, last_bin
    )
    chance = np.where(
        hit & (correlation > 0), hit_rate[cls] * survival[cls, row_bins], 1.0
    )
    surprise = -np.log(np.maximum(chance, MIN_CHANCE))
    coherent = hit & (correlation > COHERENT_CORRELATION)
    count_bucket = np.searchsorted(np.array(COUNT_NUMBER_EDGES), number, side="right")
    out = {
        "ion_fisher": surprise.sum(1),
        "ion_fisher_lib": (library_share * surprise).sum(1),
        "ion_coherent_2plus": (coherent & (charge == MAX_FRAGMENT_CHARGE)).sum(1),
    }
    for kind, name in ((B_ION, "b"), (Y_ION, "y")):
        for bucket, size in enumerate(("short", "mid", "long")):
            out[f"ion_coherent_{name}_{size}"] = (
                coherent & (ion_type == kind) & (count_bucket == bucket)
            ).sum(1)
    return out


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(1)


def entropy_similarity(
    library: np.ndarray, observed: np.ndarray, present: np.ndarray
) -> np.ndarray:
    """Spectral entropy similarity (Li et al. 2021) of library and observed intensities over the present slots.

    A present fragment with no signal counts as a missing peak, a row without observed signal gets 0.
    """
    p, q = _shares(library, present), _shares(observed, present)
    similarity = 1 - (2 * _entropy((p + q) / 2) - _entropy(p) - _entropy(q)) / np.log(4)
    return np.where(np.where(present, observed, 0.0).sum(1) > 0, similarity, 0.0)


def split_agreement(
    mask: np.ndarray, library: np.ndarray, observed: np.ndarray, correlation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Library-weighted correlation, coherent library share and coherent count of the fragments in `mask`."""
    library_mask = np.where(mask, library, 0.0)
    total = library_mask.sum(1)
    coherent = mask & (observed > 0) & (correlation > COHERENT_CORRELATION)
    zeros = np.zeros(len(mask))
    return (
        np.divide(
            (library_mask * np.where(mask, correlation, 0.0)).sum(1),
            total,
            out=zeros.copy(),
            where=total > 0,
        ),
        np.divide(
            np.where(coherent, library, 0.0).sum(1),
            total,
            out=zeros.copy(),
            where=total > 0,
        ),
        coherent.sum(1),
    )


def compute(
    fragments: tuple[np.ndarray, ...], rank: np.ndarray
) -> dict[str, np.ndarray]:
    """FEATURES of every row of the (rows, TOP_K_FRAGMENTS) fragment arrays.

    The arrays are library m/z, library intensity, observed intensity, correlation, ion type, ion number and fragment
    charge; `rank` is each row's candidate rank.
    """
    # the arrays are stored compactly (float32, uint8) and computed on in float64 and int64
    mz, library, observed, correlation = (a.astype(np.float64) for a in fragments[:4])
    ion_type, number, charge = (a.astype(np.int64) for a in fragments[4:7])
    present = mz > 0
    hit = present & (observed > 0)
    corr = np.where(present, correlation, 0.0)
    library_share = _shares(library, present)
    out = weighted_correlations(mz, library, observed, correlation)
    out |= ion_features(
        present, hit, corr, library_share, ion_type, number, charge, rank >= 1
    )
    out["ent_sim"] = entropy_similarity(library, observed, present)
    out["ent_sim_coherent"] = entropy_similarity(
        library, observed * np.clip(corr, 0, None), present
    )
    out["z2_libw_corr"], out["z2_explained"], out["z2_n_over_50"] = split_agreement(
        present & (charge == MAX_FRAGMENT_CHARGE), library, observed, corr
    )
    return {name: np.asarray(out[name], dtype=np.float32) for name in FEATURES}

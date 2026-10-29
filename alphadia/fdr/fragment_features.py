"""Final-round classifier features from the dense fragment arrays that the riding removal quantifies.

Every PSM carries its top library fragments (riding.TOP_K_FRAGMENTS slots, by library intensity, m/z 0 = empty) with
library intensity, observed intensity, XIC correlation to the candidate's profile, ion type, ion number and fragment
charge, plus the precursor charge. A decoy's slots are its target's (copied intensities, ion types and numbers) with
the m/z of the mutated sequence, so the features below describe the evidence of a target and of its decoy alike.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    "ion_fisher_lib",
    "ion_coherent_2plus",
    *(name for name in ION_COHERENT if name != "ion_coherent_b_mid"),
    "ent_sim",
    "ent_sim_coherent",
    "z2_n_over_50",
)
# Computed but not classifier features. Missingness, charge and the above-precursor split raised the paired entrapment
# ratio from 1.07 to 1.18 in the offline replays at about 1 % more real identifications; the others moved the real
# identifications by less than 1 % when permuted, and a refit without them kept them level.
UNUSED = (
    "miss_ll",
    "miss_strong",
    "charge",
    *(f"above_{name}" for name in ("libw_corr", "explained", "n_over_50")),
    "ion_fisher",
    "ion_coherent_b_mid",
    "z2_libw_corr",
    "z2_explained",
)
# Rust and python scoring columns left out of the final-round classifier: a permutation of each moved the real
# identifications of the offline replays by less than 1 %, and a refit without all of them kept the real identifications
# level at an equal or lower entrapment ratio. The bins above rank 12 never change: only the top 12 fragments are scored.
UNUSED_SCORE_COLUMNS = (
    "num_over_0_rank_12_17",
    "num_over_0_rank_18_23",
    "num_over_50_rank_12_17",
    "num_over_50_rank_18_23",
    "num_scans",
    "num_over_0",
    "num_over_50",
    "num_over_90",
    "num_over_95",
    "num_over_0_rank_6_11",
    "num_over_50_rank_6_11",
    "num_over_50_rank_0_5",
    "num_over_50_short_ions",
    "num_over_50_long_ions",
    "num_fragments",
    "num_profiles",
    "median_profile_sum_filtered",
    "intensity_correlation",
    "fwhm_rt",
)

B_ION, Y_ION = ord("b"), ord("y")
# a fragment co-elutes with its candidate above this profile correlation (as riding.COHERENT_CORRELATION)
COHERENT_CORRELATION = 0.5
# ion classes of the noise model: ion type x number bucket x fragment charge x group of SLOT_GROUP library slots
NULL_NUMBER_EDGES = (3, 5, 8)
SLOT_GROUP = 3
MAX_FRAGMENT_CHARGE = 2
COUNT_NUMBER_EDGES = (4, 8)
CORRELATION_BINS = np.linspace(0.0, 1.0, 101)
# the chance of a hit is floored here: one fragment adds at most -log(MIN_CHANCE) surprise
MIN_CHANCE = 1e-4
# missingness: the detection floor is this quantile of the coherent fragments' observed intensities, a fragment
# "should be seen" above STRONG_FACTOR times the floor, the scale needs MIN_COHERENT_FOR_SCALE coherent fragments
FLOOR_QUANTILE = 0.05
STRONG_FACTOR = 3.0
MIN_COHERENT_FOR_SCALE = 2
LOGISTIC_ROWS = 2_000_000
ROW_CHUNK = 1_000_000
SEED = 20261028


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


def ion_features(
    present,
    hit,
    correlation,
    library_share,
    ion_type,
    number,
    charge,
    null_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fisher sums of each fragment's chance under a label-blind noise model of its ion class (fitted on the rows
    `null_rows`, the rank >= 1 candidates of both classes) and coherent fragment counts per ion class."""
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


def missingness(
    present, observed, correlation, library_share, rank: np.ndarray
) -> dict[str, np.ndarray]:
    """Whether the absent fragments are consistent with the intensity the coherent ones imply: the mean
    log-likelihood of hits and misses under a label-blind detection curve, and the number of absent fragments the
    implied intensity puts well above the detection floor."""
    coherent = (
        present
        & (observed > 0)
        & (correlation > COHERENT_CORRELATION)
        & (library_share > 0)
    )
    floor = np.quantile(observed[coherent], FLOOR_QUANTILE) if coherent.any() else 1.0
    ratio = np.where(coherent, observed / np.maximum(library_share, 1e-12), np.nan)
    enough = coherent.sum(1) >= MIN_COHERENT_FOR_SCALE
    scale = np.full(len(observed), np.nan)
    for start in range(0, len(observed), ROW_CHUNK):
        rows = np.flatnonzero(enough[start : start + ROW_CHUNK]) + start
        scale[rows] = np.nanmedian(ratio[rows], axis=1)
    expected = scale[:, None] * library_share
    log_ratio = np.log(np.maximum(expected, 1e-12) / floor)
    hit = (observed > 0).astype(float)
    fit_rows = np.flatnonzero(enough & (rank == 0))
    if len(fit_rows) == 0:
        return {
            "miss_ll": np.zeros(len(observed)),
            "miss_strong": np.zeros(len(observed)),
        }
    fit_rows = np.random.default_rng(SEED).choice(
        fit_rows, min(LOGISTIC_ROWS, len(fit_rows)), replace=False
    )
    fit_present = present[fit_rows]
    model = LogisticRegression().fit(
        log_ratio[fit_rows][fit_present][:, None], hit[fit_rows][fit_present]
    )
    q = 1.0 / (1.0 + np.exp(-(model.coef_[0, 0] * log_ratio + model.intercept_[0])))
    q = np.clip(q, 1e-6, 1 - 1e-6)
    log_likelihood = np.where(present, hit * np.log(q) + (1 - hit) * np.log(1 - q), 0.0)
    miss_ll = log_likelihood.sum(1) / np.maximum(present.sum(1), 1)
    strong_missing = present & (observed <= 0) & (expected > STRONG_FACTOR * floor)
    return {
        "miss_ll": np.where(enough, miss_ll, 0.0),
        "miss_strong": np.where(enough, strong_missing.sum(1), 0),
    }


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(1)


def entropy_similarity(
    library: np.ndarray, observed: np.ndarray, present: np.ndarray
) -> np.ndarray:
    """Spectral entropy similarity (Li et al. 2021) of library and observed intensities over the present slots; a
    present fragment with no signal counts as a missing peak, a row without observed signal gets 0."""
    p, q = _shares(library, present), _shares(observed, present)
    similarity = 1 - (2 * _entropy((p + q) / 2) - _entropy(p) - _entropy(q)) / np.log(4)
    return np.where(np.where(present, observed, 0.0).sum(1) > 0, similarity, 0.0)


def split_agreement(
    mask, library, observed, correlation
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
    fragments: tuple[np.ndarray, ...], rank: np.ndarray, precursor_mz: np.ndarray
) -> dict[str, np.ndarray]:
    """FEATURES of every row of the fragment arrays (m/z, library, observed, correlation, ion type, ion number,
    fragment charge per slot, and the precursor charge per row)."""
    mz, library, observed, correlation, ion_type, number, charge, precursor_charge = (
        fragments
    )
    present = mz > 0
    hit = present & (observed > 0)
    corr = np.where(present, correlation, 0.0)
    library_share = _shares(library, present)
    out = weighted_correlations(mz, library, observed, correlation)
    out |= ion_features(
        present, hit, corr, library_share, ion_type, number, charge, rank >= 1
    )
    out |= missingness(present, observed, corr, library_share, rank)
    out["charge"] = precursor_charge
    out["ent_sim"] = entropy_similarity(library, observed, present)
    out["ent_sim_coherent"] = entropy_similarity(
        library, observed * np.clip(corr, 0, None), present
    )
    for name, mask in (
        ("z2", present & (charge == MAX_FRAGMENT_CHARGE)),
        ("above", present & (mz > precursor_mz[:, None])),
    ):
        out[f"{name}_libw_corr"], out[f"{name}_explained"], out[f"{name}_n_over_50"] = (
            split_agreement(mask, library, observed, corr)
        )
    return {name: np.asarray(out[name], dtype=np.float32) for name in FEATURES}


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

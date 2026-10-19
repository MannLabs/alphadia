"""Conversion of AlphaDIA to NG data structure and back."""

import numba as nb
import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import (
    CandidateCollection,
    CandidateContext,
    CandidateFeatureCollection,
    set_num_threads,
)
from alphadia_search_rs import (
    DIAData as DiaDataNG,
)
from alphadia_search_rs import SpecLibFlat as SpecLibFlatNG

from alphadia.raw_data import DiaData

CANDIDATE_KEY_COLUMNS = ["precursor_idx", "rank"]
# How much better a candidate is than the best other candidate (peak) of the same precursor. A real peptide usually
# has one peak that stands out; a false target several similar ones. In alphaDIA's uncertain band on hela1ft_entr these
# separate real from false targets beyond every per-candidate feature (+0.03 AUC each).
RANK_DELTA_SOURCE_FEATURES = (
    "score",
    "hyperscore_intensity_observation",
    "mean_correlation",
)
RANK_DELTA_PREFIX = "rank_delta_"
N_OTHER_CANDIDATES = "n_other_candidates"
# Other charge states of the same peptide that peak at the same time. A real peptide often shows up in several charge
# states at one apex, a false target only by chance. In alphaDIA's uncertain band on hela1ft_entr 18.7 % of HUMAN vs
# 1.1 % of ARATH targets have such a sibling; decoys get one slightly more often than false targets (A / D_A 0.84-0.89),
# so the feature errs on the conservative side.
SIBLING_RT_TOLERANCE_SECONDS = 3.0
SIBLING_SOURCE_FEATURES = ("score", "mean_correlation")
SIBLING_PREFIX = "sibling_best_"
N_SIBLINGS_COELUTING = "n_sibling_coeluting"
PEPTIDE_KEY_COLUMNS = ["sequence", "mods", "mod_sites", "decoy", "channel"]
MAX_SIBLINGS = 16


def set_ng_thread_count(thread_count: int) -> None:
    """Set the number of threads for NG computations."""
    set_num_threads(thread_count)


def dia_data_to_ng(dia_data: DiaData) -> "DiaDataNG":  # noqa: F821
    """Convert DIA data from classic to ng format."""

    spectrum_df = dia_data.spectrum_df
    peak_df = dia_data.peak_df

    cycle_len = dia_data.cycle.shape[1]
    spectrum_df_len = len(dia_data.spectrum_df)

    delta_scan_idx = np.tile(
        np.arange(cycle_len), int(spectrum_df_len / cycle_len + 1)
    )[:spectrum_df_len]
    cycle_idx = np.repeat(np.arange(int(spectrum_df_len / cycle_len + 1)), cycle_len)[
        :spectrum_df_len
    ]

    return DiaDataNG.from_arrays(
        delta_scan_idx.astype(np.int64),
        spectrum_df["isolation_lower_mz"].values.astype(np.float32),
        spectrum_df["isolation_upper_mz"].values.astype(np.float32),
        spectrum_df["peak_start_idx"].values.astype(np.int64),
        spectrum_df["peak_stop_idx"].values.astype(np.int64),
        cycle_idx.astype(np.int64),
        spectrum_df["rt"].values.astype(np.float32) * 60,
        peak_df["mz"].values.astype(np.float32),
        peak_df["intensity"].values.astype(np.float32),
        dia_data.cycle.astype(np.float32),
    )


def speclib_to_ng(
    speclib: SpecLibFlat,
    *,
    rt_column: str,
    precursor_mz_column: str,
    fragment_mz_column: str,
) -> "SpecLibFlatNG":  # noqa: F821
    """Convert speclib from classic to ng format."""

    precursor_df = speclib.precursor_df
    fragment_df = speclib.fragment_df

    return SpecLibFlatNG.from_arrays(
        precursor_df["precursor_idx"].values.astype(np.uint64),
        precursor_df["mz_library"].values.astype(np.float32),
        precursor_df[precursor_mz_column].values.astype(np.float32),
        precursor_df["rt_library"].values.astype(np.float32),
        precursor_df[rt_column].values.astype(np.float32),
        precursor_df["nAA"].values.astype(np.uint8),
        precursor_df["flat_frag_start_idx"].values.astype(np.uint64),
        precursor_df["flat_frag_stop_idx"].values.astype(np.uint64),
        fragment_df["mz_library"].values.astype(np.float32),
        fragment_df[fragment_mz_column].values.astype(np.float32),
        fragment_df["intensity"].values.astype(np.float32),
        fragment_df["cardinality"].values.astype(np.uint8),
        fragment_df["charge"].values.astype(np.uint8),
        fragment_df["loss_type"].values.astype(np.uint8),
        fragment_df["number"].values.astype(np.uint8),
        fragment_df["position"].values.astype(np.uint8),
        fragment_df["type"].values.astype(np.uint8),
    )


def get_feature_names() -> list[str]:
    """Get feature names from NG CandidateFeatureCollection."""
    return [f for f in CandidateFeatureCollection.get_feature_names()]


def get_context_feature_names() -> list[str]:
    """Get the competition and context feature names from NG CandidateContext."""
    return list(CandidateContext.get_feature_names())


def get_rank_delta_feature_names() -> list[str]:
    """Names of the columns `add_rank_delta_features` adds."""
    return [RANK_DELTA_PREFIX + name for name in RANK_DELTA_SOURCE_FEATURES] + [
        N_OTHER_CANDIDATES
    ]


def get_final_round_feature_names() -> list[str]:
    """Features the FDR classifier uses only in the final round, not in the calibration rounds."""
    return get_rank_delta_feature_names()


def add_rank_delta_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add each source feature minus its best value among the other candidates of the same precursor, and their number.

    A precursor without other candidates gets the feature minus 0.
    """
    precursor_idx = features_df["precursor_idx"].to_numpy()
    group_size = features_df.groupby("precursor_idx")["precursor_idx"].transform("size")
    features_df[N_OTHER_CANDIDATES] = (group_size - 1).to_numpy(dtype=np.float32)
    for name in RANK_DELTA_SOURCE_FEATURES:
        values = features_df[name].to_numpy(dtype=np.float64)
        order = np.lexsort((-values, precursor_idx))
        sorted_precursor = precursor_idx[order]
        sorted_values = values[order]
        is_first = np.ones(len(order), dtype=bool)
        is_first[1:] = sorted_precursor[1:] != sorted_precursor[:-1]
        first_position = np.maximum.accumulate(
            np.where(is_first, np.arange(len(order)), 0)
        )
        best = sorted_values[first_position]
        next_position = np.minimum(first_position + 1, len(order) - 1)
        has_second = (first_position + 1 < len(order)) & (
            sorted_precursor[next_position] == sorted_precursor
        )
        second = np.where(has_second, sorted_values[next_position], 0.0)
        best_other = np.where(is_first, second, best)
        delta = np.empty(len(order))
        delta[order] = sorted_values - best_other
        features_df[RANK_DELTA_PREFIX + name] = delta.astype(np.float32)
    return features_df


def get_sibling_feature_names() -> list[str]:
    """Names of the columns `add_sibling_features` adds."""
    return [N_SIBLINGS_COELUTING] + [
        SIBLING_PREFIX + name for name in SIBLING_SOURCE_FEATURES
    ]


@nb.njit(cache=True)
def _sibling_scan(
    peptide: np.ndarray,
    precursor_idx: np.ndarray,
    rt: np.ndarray,
    values: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """For rows sorted by (peptide, rt): the number of distinct other precursors of the peptide with a row within
    `tolerance`, and the best value of each column among those rows (0 without any)."""
    n, n_columns = values.shape
    count = np.zeros(n, dtype=np.float32)
    best = np.zeros((n, n_columns), dtype=np.float32)
    seen = np.empty(MAX_SIBLINGS, dtype=np.int64)
    for i in range(n):
        n_seen = 0
        row_best = np.full(n_columns, -np.inf, dtype=np.float32)
        for step in (-1, 1):
            j = i + step
            while (
                0 <= j < n
                and peptide[j] == peptide[i]
                and abs(rt[j] - rt[i]) <= tolerance
            ):
                if precursor_idx[j] != precursor_idx[i]:
                    known = False
                    for k in range(n_seen):
                        known = known or seen[k] == precursor_idx[j]
                    if not known and n_seen < MAX_SIBLINGS:
                        seen[n_seen] = precursor_idx[j]
                        n_seen += 1
                    for c in range(n_columns):
                        row_best[c] = max(row_best[c], values[j, c])
                j += step
        count[i] = n_seen
        if n_seen > 0:
            best[i] = row_best
    return count, best


def add_sibling_features(
    features_df: pd.DataFrame, precursor_df: pd.DataFrame
) -> pd.DataFrame:
    """Add the number of the peptide's other charge states with a candidate within SIBLING_RT_TOLERANCE_SECONDS of
    this candidate, and the best source features among those candidates (0 without any)."""
    present = precursor_df[
        precursor_df["precursor_idx"].isin(features_df["precursor_idx"].unique())
    ]
    keys = present[PEPTIDE_KEY_COLUMNS[0]].astype(str)
    for column in PEPTIDE_KEY_COLUMNS[1:]:
        keys = keys + "|" + present[column].astype(str)
    peptide_of = pd.Series(
        pd.factorize(keys)[0], index=present["precursor_idx"].to_numpy()
    )
    peptide = peptide_of.reindex(features_df["precursor_idx"].to_numpy()).to_numpy()
    rt = features_df["rt_observed"].to_numpy(dtype=np.float64)
    order = np.lexsort((rt, peptide))
    values = features_df[list(SIBLING_SOURCE_FEATURES)].to_numpy(dtype=np.float32)
    count, best = _sibling_scan(
        peptide[order],
        features_df["precursor_idx"].to_numpy(dtype=np.int64)[order],
        rt[order],
        values[order],
        SIBLING_RT_TOLERANCE_SECONDS,
    )
    unsorted = np.empty((len(order), 1 + best.shape[1]), dtype=np.float32)
    unsorted[order] = np.column_stack([count, best])
    for c, name in enumerate(get_sibling_feature_names()):
        features_df[name] = unsorted[:, c]
    return features_df


def merge_context_features(
    features_df: pd.DataFrame, context_features: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Merge the output of `CandidateContext.compute()` into the scored candidates."""
    context_df = pd.DataFrame(context_features)

    # checked before the merge, so that the error can name the offending candidates
    duplicates = context_df.loc[
        context_df.duplicated(CANDIDATE_KEY_COLUMNS), CANDIDATE_KEY_COLUMNS
    ]
    if not duplicates.empty:
        raise ValueError(
            f"Context features contain duplicate candidates:\n"
            f"{duplicates.to_string(index=False)}"
        )

    merged_df = features_df.merge(
        context_df, on=CANDIDATE_KEY_COLUMNS, validate="one_to_one"
    )

    num_missing = len(features_df) - len(merged_df)
    if num_missing > 0:
        raise ValueError(f"Context features are missing for {num_missing} candidates")

    return merged_df


def parse_candidates(
    candidates: CandidateCollection, spectral_library: SpecLibFlat, dia_data: DiaDataNG
) -> pd.DataFrame:
    """Parse candidates from NG to classic format."""

    cycle_len = dia_data.cycle.shape[1]

    result = candidates.to_arrays()

    precursor_idx = result[0]
    rank = result[1]
    score = result[2]
    scan_center = result[3]
    scan_start = result[4]
    scan_stop = result[5]
    frame_center = result[6]
    frame_start = result[7]
    frame_stop = result[8]

    candidates_df = pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "rank": rank,
            "score": score,
            "scan_center": scan_center,
            "scan_start": scan_start,
            "scan_stop": scan_stop,
            "frame_center": frame_center,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )

    candidates_df = candidates_df.merge(
        spectral_library.precursor_df[["precursor_idx", "elution_group_idx", "decoy"]],
        on="precursor_idx",
        how="left",
    )

    candidates_df["frame_start"] = candidates_df["frame_start"] * cycle_len
    candidates_df["frame_stop"] = candidates_df["frame_stop"] * cycle_len
    candidates_df["frame_center"] = candidates_df["frame_center"] * cycle_len

    candidates_df["scan_start"] = 0
    candidates_df["scan_stop"] = 1
    candidates_df["scan_center"] = 0

    return candidates_df


def candidates_to_ng(
    candidates_df: pd.DataFrame, dia_data: DiaDataNG
) -> CandidateCollection:
    """Convert candidates from classic to NG format."""

    cycle_len = dia_data.cycle.shape[1]

    candidates = CandidateCollection.from_arrays(
        candidates_df["precursor_idx"].values.astype(np.uint64),
        candidates_df["rank"].values.astype(np.uint64),
        candidates_df["score"].values.astype(np.float32),
        candidates_df["scan_center"].values.astype(np.uint64),
        candidates_df["scan_start"].values.astype(np.uint64),
        candidates_df["scan_stop"].values.astype(np.uint64),
        candidates_df["frame_center"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_start"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_stop"].values.astype(np.uint64) // cycle_len,
    )
    return candidates


def to_features_df(
    candidate_features: CandidateFeatureCollection, spectral_library: SpecLibFlat
) -> pd.DataFrame:
    """Convert NG candidate features to classic format."""

    features_dict = candidate_features.to_dict_arrays()

    features_df = pd.DataFrame(features_dict)

    features_df = features_df.merge(
        spectral_library.precursor_df[
            [
                "precursor_idx",
                "decoy",
                "elution_group_idx",
                "channel",
                "proteins",
            ]
        ],
        on="precursor_idx",
        how="left",
    )

    return features_df


def parse_quantification(
    quantified_speclib: "SpecLibFlatQuantified",  # noqa: F821
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert NG quantified spectral library to classic precursor and fragments DataFrame."""

    precursor_dict, fragment_dict = quantified_speclib.to_dict_arrays()

    precursor_df = pd.DataFrame(precursor_dict).rename(
        columns={"idx": "precursor_idx"}
    )  # TODO: remove when #96 is merged

    fragments_df = pd.DataFrame(fragment_dict).rename(
        columns={
            "correlation_observed": "correlation",
            "mass_error_observed": "mass_error",
        }
    )

    return precursor_df, fragments_df

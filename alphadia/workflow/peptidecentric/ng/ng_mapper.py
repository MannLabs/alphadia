"""Conversion of AlphaDIA to NG data structure and back."""

import logging

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

logger = logging.getLogger()

CANDIDATE_KEY_COLUMNS = ["precursor_idx", "rank"]

# A decoy and its target share elution group, channel and charge.
DECOY_TARGET_KEY_COLUMNS = ["elution_group_idx", "channel", "charge"]

# A decoy fragment corresponds to the target fragment with the same annotation.
FRAGMENT_ANNOTATION_COLUMNS = ["type", "number", "charge", "loss_type"]


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
    """Convert speclib from classic to ng format.

    The NG library uses the library fragment m/z only to look up the fragment IDF. Every decoy fragment is passed with
    the library m/z of the corresponding fragment of its target, so a decoy gets exactly the IDF weights of its target
    while it is still matched at its own m/z.
    """

    precursor_df = speclib.precursor_df
    fragment_df = speclib.fragment_df

    fragment_mz_library = target_fragment_mz_for_decoys(precursor_df, fragment_df)

    return SpecLibFlatNG.from_arrays(
        precursor_df["precursor_idx"].values.astype(np.uint64),
        precursor_df["mz_library"].values.astype(np.float32),
        precursor_df[precursor_mz_column].values.astype(np.float32),
        precursor_df["rt_library"].values.astype(np.float32),
        precursor_df[rt_column].values.astype(np.float32),
        precursor_df["nAA"].values.astype(np.uint8),
        precursor_df["flat_frag_start_idx"].values.astype(np.uint64),
        precursor_df["flat_frag_stop_idx"].values.astype(np.uint64),
        fragment_mz_library,
        fragment_df[fragment_mz_column].values.astype(np.float32),
        fragment_df["intensity"].values.astype(np.float32),
        fragment_df["cardinality"].values.astype(np.uint8),
        fragment_df["charge"].values.astype(np.uint8),
        fragment_df["loss_type"].values.astype(np.uint8),
        fragment_df["number"].values.astype(np.uint8),
        fragment_df["position"].values.astype(np.uint8),
        fragment_df["type"].values.astype(np.uint8),
    )


def target_fragment_mz_for_decoys(
    precursor_df: pd.DataFrame, fragment_df: pd.DataFrame
) -> np.ndarray:
    """The library fragment m/z, with each decoy fragment replaced by that of the corresponding target fragment.

    The IDF weights a fragment by how rare its m/z is in the library. The decoy mutation moves decoy fragments onto
    masses of their own, so with their own m/z decoys get different weights than any target and the IDF features tell
    them apart from the library alone. A decoy is a copy of its target with mutated residues, so its fragments come in the
    same order; a fragment takes the target's m/z where the annotation at the same offset agrees. Decoys without a target
    of the same elution group, channel and charge, or with a different fragment count, keep their own m/z.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Library precursors with `decoy`, `elution_group_idx`, `channel`, `charge`, `flat_frag_start_idx` and
        `flat_frag_stop_idx`.

    fragment_df : pd.DataFrame
        Library fragments with `mz_library` and the annotation columns.

    Returns
    -------
    np.ndarray
        Fragment m/z of dtype float32, one per row of `fragment_df`.

    """
    fragment_mz = fragment_df["mz_library"].to_numpy(dtype=np.float32, copy=True)
    span = ["flat_frag_start_idx", "flat_frag_stop_idx"]
    precursors = precursor_df[["decoy", *DECOY_TARGET_KEY_COLUMNS, *span]]

    targets = precursors[precursors["decoy"] == 0].drop_duplicates(
        DECOY_TARGET_KEY_COLUMNS
    )
    pairs = precursors[precursors["decoy"] == 1].merge(
        targets, on=DECOY_TARGET_KEY_COLUMNS, suffixes=("_decoy", "_target")
    )
    decoy_start = pairs["flat_frag_start_idx_decoy"].to_numpy(dtype=np.int64)
    target_start = pairs["flat_frag_start_idx_target"].to_numpy(dtype=np.int64)
    lengths = pairs["flat_frag_stop_idx_decoy"].to_numpy(dtype=np.int64) - decoy_start
    same_length = (
        pairs["flat_frag_stop_idx_target"].to_numpy(dtype=np.int64) - target_start
        == lengths
    )
    decoy_start, target_start, lengths = (
        decoy_start[same_length],
        target_start[same_length],
        lengths[same_length],
    )

    offsets = np.arange(lengths.sum()) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    decoy_idx = np.repeat(decoy_start, lengths) + offsets
    target_idx = np.repeat(target_start, lengths) + offsets

    same_fragment = np.ones(len(decoy_idx), dtype=bool)
    for column in FRAGMENT_ANNOTATION_COLUMNS:
        values = fragment_df[column].to_numpy()
        same_fragment &= values[decoy_idx] == values[target_idx]

    fragment_mz[decoy_idx[same_fragment]] = fragment_mz[target_idx[same_fragment]]

    num_decoy_fragments = int(
        (
            precursor_df.loc[precursor_df["decoy"] == 1, "flat_frag_stop_idx"]
            - precursor_df.loc[precursor_df["decoy"] == 1, "flat_frag_start_idx"]
        ).sum()
    )
    logger.info(
        f"IDF of decoys: {same_fragment.sum():,} of {num_decoy_fragments:,} decoy fragments take the m/z of their target fragment"
    )
    return fragment_mz


def get_feature_names() -> list[str]:
    """Get feature names from NG CandidateFeatureCollection."""
    return [f for f in CandidateFeatureCollection.get_feature_names()]


def get_context_feature_names() -> list[str]:
    """Get the competition and context feature names from NG CandidateContext."""
    return list(CandidateContext.get_feature_names())


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

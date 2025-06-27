"""Utility Functions for Candidate Scoring."""

import logging

import alphatims.utils
import numba as nb
import numpy as np
import pandas as pd

from alphadia import validate
from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger()


@nb.njit(cache=USE_NUMBA_CACHING)
def tile(a, n):
    return np.repeat(a, n).reshape(-1, n).T.flatten()


@nb.njit(cache=USE_NUMBA_CACHING)
def frame_profile_2d(x):
    return np.sum(x, axis=2)


@nb.njit(cache=USE_NUMBA_CACHING)
def frame_profile_1d(x):
    return np.sum(x, axis=1)


@nb.njit(cache=USE_NUMBA_CACHING)
def scan_profile_2d(x):
    return np.sum(x, axis=3)


@nb.njit(cache=USE_NUMBA_CACHING)
def scan_profile_1d(x):
    return np.sum(x, axis=2)


@nb.njit(cache=USE_NUMBA_CACHING)
def or_envelope_1d(x):
    res = x.copy()
    for a0 in range(x.shape[0]):
        for i in range(1, x.shape[1] - 1):
            if (x[a0, i] < x[a0, i - 1]) or (x[a0, i] < x[a0, i + 1]):
                res[a0, i] = (x[a0, i - 1] + x[a0, i + 1]) / 2
    return res


@nb.njit(cache=USE_NUMBA_CACHING)
def or_envelope_2d(x):
    res = x.copy()
    for a0 in range(x.shape[0]):
        for a1 in range(x.shape[1]):
            for i in range(1, x.shape[2] - 1):
                if (x[a0, a1, i] < x[a0, a1, i - 1]) or (
                    x[a0, a1, i] < x[a0, a1, i + 1]
                ):
                    res[a0, a1, i] = (x[a0, a1, i - 1] + x[a0, a1, i + 1]) / 2
    return res


def candidate_features_to_candidates(
    candidate_features_df: pd.DataFrame,
    optional_columns: list[str] | None = None,
):
    """create candidates_df from candidate_features_df

    Parameters
    ----------

    candidate_features_df : pd.DataFrame
        candidate_features_df

    Returns
    -------

    candidate_df : pd.DataFrame
        candidates_df
    """

    # validate candidate_features_df input
    if optional_columns is None:
        optional_columns = ["proba"]
    validate.candidate_features_df(candidate_features_df.copy())

    required_columns = [
        "elution_group_idx",
        "precursor_idx",
        "rank",
        "scan_start",
        "scan_stop",
        "scan_center",
        "frame_start",
        "frame_stop",
        "frame_center",
    ]

    # select required columns
    candidate_df = candidate_features_df[required_columns + optional_columns].copy()
    # validate candidate_df output
    validate.candidates_df(candidate_df)

    return candidate_df


def multiplex_candidates(
    candidates_df: pd.DataFrame,
    precursors_flat_df: pd.DataFrame,
    remove_decoys: bool = True,
    channels: list[int] | None = None,
):
    """Takes a candidates dataframe and a precursors dataframe and returns a multiplexed candidates dataframe.
    All original candidates will be retained. For missing candidates, the best scoring candidate in the elution group will be used and multiplexed across all missing channels.

    Parameters
    ----------

    candidates_df : pd.DataFrame
        Candidates dataframe as returned by `hybridselection.HybridCandidateSelection`

    precursors_flat_df : pd.DataFrame
        Precursors dataframe

    remove_decoys : bool, optional
        If True, remove decoys from the precursors dataframe, by default True

    channels : typing.List[int], optional
        List of channels to include in the multiplexed candidates dataframe, by default [0,4,8,12]

    Returns
    -------

    pd.DataFrame
        Multiplexed candidates dataframe

    """
    if channels is None:
        channels = [0, 4, 8, 12]
    precursors_flat_view = precursors_flat_df.copy()
    best_candidate_view = candidates_df.copy()

    validate.precursors_flat(precursors_flat_view)
    validate.candidates_df(best_candidate_view)

    # remove decoys if requested
    if remove_decoys:
        precursors_flat_view = precursors_flat_df[precursors_flat_df["decoy"] == 0]
        if "decoy" in best_candidate_view.columns:
            best_candidate_view = best_candidate_view[best_candidate_view["decoy"] == 0]

    # original precursors are forbidden as they will be concatenated in the end
    # the candidate used for multiplexing is the best scoring candidate in each elution group
    best_candidate_view = (
        best_candidate_view.sort_values("proba")
        .groupby("elution_group_idx")
        .first()
        .reset_index()
    )

    # get all candidate elution group
    candidate_elution_group_idxs = best_candidate_view["elution_group_idx"].unique()

    # restrict precursors to channels and candidate elution groups
    precursors_flat_view = precursors_flat_view[
        precursors_flat_view["channel"].isin(channels)
    ]

    precursors_flat_view = precursors_flat_view[
        precursors_flat_view["elution_group_idx"].isin(candidate_elution_group_idxs)
    ]
    # remove original precursors
    precursors_flat_view = precursors_flat_view[
        ["elution_group_idx", "precursor_idx", "channel"]
    ]
    # reduce precursors to the elution group level
    best_candidate_view = best_candidate_view.drop(columns=["precursor_idx"])
    if "channel" in best_candidate_view.columns:
        best_candidate_view = best_candidate_view.drop(columns=["channel"])

    # merge candidates and precursors
    multiplexed_candidates_df = precursors_flat_view.merge(
        best_candidate_view, on="elution_group_idx", how="left"
    )

    # append original candidates
    # multiplexed_candidates_df = pd.concat([multiplexed_candidates_df, candidates_view]).sort_values('precursor_idx')
    validate.candidates_df(multiplexed_candidates_df)

    return multiplexed_candidates_df


@alphatims.utils.pjit(cache=USE_NUMBA_CACHING)
def transfer_feature(  # TODO: unused?
    idx, score_group_container, feature_array, precursor_idx_array, rank_array
):
    feature_array[idx] = score_group_container[idx].candidates[0].feature_array
    precursor_idx_array[idx] = score_group_container[idx].candidates[0].precursor_idx
    rank_array[idx] = score_group_container[idx].candidates[0].rank

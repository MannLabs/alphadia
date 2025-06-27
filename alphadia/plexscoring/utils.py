"""Utility Functions for Candidate Scoring."""

import logging

import alphatims.utils
import numba as nb
import numpy as np
import pandas as pd

from alphadia.utils import USE_NUMBA_CACHING
from alphadia.validation.schemas import (
    candidates_schema,
    features_schema,
    precursors_flat_schema,
)

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

    features_schema.validate(candidate_features_df, warn_on_critical_values=True)

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
    candidates_schema.validate(candidate_df, warn_on_critical_values=True)

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

    precursors_flat_schema.validate(precursors_flat_view, warn_on_critical_values=True)
    candidates_schema.validate(best_candidate_view, warn_on_critical_values=True)

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

    candidates_schema.validate(multiplexed_candidates_df, warn_on_critical_values=True)

    return multiplexed_candidates_df


@alphatims.utils.pjit(cache=USE_NUMBA_CACHING)
def transfer_feature(  # TODO: unused?
    idx, score_group_container, feature_array, precursor_idx_array, rank_array
):
    feature_array[idx] = score_group_container[idx].candidates[0].feature_array
    precursor_idx_array[idx] = score_group_container[idx].candidates[0].precursor_idx
    rank_array[idx] = score_group_container[idx].candidates[0].rank


def merge_missing_columns(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    right_columns: list,
    on: list = None,
    how: str = "left",
):
    """Merge missing columns from right_df into left_df.

    Merging is performed only for columns not yet present in left_df.

    Parameters
    ----------

    left_df : pandas.DataFrame
        Left dataframe

    right_df : pandas.DataFrame
        Right dataframe

    right_columns : list
        List of columns to merge from right_df into left_df

    on : list, optional
        List of columns to merge on, by default None

    how : str, optional
        How to merge, by default 'left'

    Returns
    -------
    pandas.DataFrame
        Merged left dataframe

    """
    if isinstance(on, str):
        on = [on]

    if isinstance(right_columns, str):
        right_columns = [right_columns]

    missing_from_left = list(set(right_columns) - set(left_df.columns))
    missing_from_right = list(set(missing_from_left) - set(right_df.columns))

    if len(missing_from_left) == 0:
        return left_df

    if missing_from_right:
        raise ValueError(f"Columns {missing_from_right} must be present in right_df")

    if on is None:
        raise ValueError("Parameter on must be specified")

    if not all([col in left_df.columns for col in on]):
        raise ValueError(f"Columns {on} must be present in left_df")

    if not all([col in right_df.columns for col in on]):
        raise ValueError(f"Columns {on} must be present in right_df")

    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError("Parameter how must be one of left, right, inner, outer")

    # merge
    return left_df.merge(right_df[on + missing_from_left], on=on, how=how)


def calculate_score_groups(
    input_df: pd.DataFrame,
    group_channels: bool = False,
):
    """
    Calculate score groups for DIA multiplexing.

    On the candidate selection level, score groups are used to group ions across channels.
    On the scoring level, score groups are used to group channels of the same precursor and rank together.

    This function makes sure that all precursors within a score group have the same `elution_group_idx`, `decoy` status and `rank` if available.
    If `group_channels` is True, different channels of the same precursor will be grouped together.

    Parameters
    ----------

    input_df : pandas.DataFrame
        Precursor dataframe. Must contain columns 'elution_group_idx' and 'decoy'. Can contain 'rank' column.

    group_channels : bool
        If True, precursors from the same elution group will be grouped together while seperating different ranks and decoy status.

    Returns
    -------

    score_groups : pandas.DataFrame
        Updated precursor dataframe with score_group_idx column.

    Example
    -------

    A precursor with the same `elution_group_idx` might be grouped with other precursors if only the `channel` is different.
    Different `rank` and `decoy` status will always lead to different score groups.

    .. list-table::
        :widths: 25 25 25 25 25 25
        :header-rows: 1

        * - elution_group_idx
          - rank
          - decoy
          - channel
          - group_channels = False
          - group_channels = True

        * - 0
          - 0
          - 0
          - 0
          - 0
          - 0

        * - 0
          - 0
          - 0
          - 4
          - 1
          - 0

        * - 0
          - 1
          - 0
          - 0
          - 2
          - 1

        * - 0
          - 1
          - 1
          - 0
          - 3
          - 2

    """

    @nb.njit(cache=USE_NUMBA_CACHING)
    def channel_score_groups(elution_group_idx, decoy, rank):
        """
        Calculate score groups for channel grouping.

        Parameters
        ----------

        elution_group_idx : numpy.ndarray
            Elution group indices.

        decoy : numpy.ndarray
            Decoy status.

        rank : numpy.ndarray
            Rank of precursor.

        Returns
        -------

        score_groups : numpy.ndarray
            Score groups.
        """
        score_groups = np.zeros(len(elution_group_idx), dtype=np.uint32)
        current_group = 0
        current_eg = elution_group_idx[0]
        current_decoy = decoy[0]
        current_rank = rank[0]

        for i in range(len(elution_group_idx)):
            # if elution group, decoy status or rank changes, increase score group
            if (
                (elution_group_idx[i] != current_eg)
                or (decoy[i] != current_decoy)
                or (rank[i] != current_rank)
            ):
                current_group += 1
                current_eg = elution_group_idx[i]
                current_decoy = decoy[i]
                current_rank = rank[i]

            score_groups[i] = current_group
        return score_groups

    # sort by elution group, decoy and rank
    # if no rank is present, pretend rank 0
    if "rank" in input_df.columns:
        input_df = input_df.sort_values(by=["elution_group_idx", "decoy", "rank"])
        rank_values = input_df["rank"].values
    else:
        input_df = input_df.sort_values(by=["elution_group_idx", "decoy"])
        rank_values = np.zeros(len(input_df), dtype=np.uint32)

    if group_channels:
        input_df["score_group_idx"] = channel_score_groups(
            input_df["elution_group_idx"].values, input_df["decoy"].values, rank_values
        )
    else:
        input_df["score_group_idx"] = np.arange(len(input_df), dtype=np.uint32)

    return input_df.sort_values(by=["score_group_idx"]).reset_index(drop=True)

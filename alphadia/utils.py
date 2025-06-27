import logging
import math
import os
import platform

import numba as nb
import numpy as np
import pandas as pd

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


def get_isotope_column_names(colnames):
    return [f"i_{i}" for i in get_isotope_columns(colnames)]


@nb.njit(cache=USE_NUMBA_CACHING)
def mass_range(mz_list, ppm_tolerance):
    out_mz = np.zeros((len(mz_list), 2), dtype=mz_list.dtype)
    out_mz[:, 0] = mz_list - ppm_tolerance * mz_list / (10**6)
    out_mz[:, 1] = mz_list + ppm_tolerance * mz_list / (10**6)
    return out_mz


@nb.njit(cache=USE_NUMBA_CACHING)
def tile(a, n):
    return np.repeat(a, n).reshape(-1, n).T.flatten()


@nb.njit(cache=USE_NUMBA_CACHING)
def make_slice_1d(start_stop):
    """Numba helper function to create a 1D slice object from a start and stop value.

        e.g. make_slice_1d([0, 10]) -> np.array([[0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (2,) containing the start and stop value.

    Returns
    -------
    np.ndarray
        Array of shape (1,3) containing the start, stop and step value.

    """
    return np.array([[start_stop[0], start_stop[1], 1]], dtype=start_stop.dtype)


@nb.njit(cache=USE_NUMBA_CACHING)
def make_slice_2d(start_stop):
    """Numba helper function to create a 2D slice object from multiple start and stop value.

        e.g. make_slice_2d([[0, 10], [0, 10]]) -> np.array([[0, 10, 1], [0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (N, 2) containing the start and stop value for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing the start, stop and step value for each dimension.

    """

    out = np.ones((start_stop.shape[0], 3), dtype=start_stop.dtype)
    out[:, 0] = start_stop[:, 0]
    out[:, 1] = start_stop[:, 1]
    return out


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


@nb.njit(inline="always", cache=USE_NUMBA_CACHING)
def get_frame_indices(
    rt_values: np.ndarray,
    rt_values_array: np.ndarray,
    zeroth_frame: int,
    cycle_len: int,
    precursor_cycle_max_index: int,
    optimize_size: int = 16,
    min_size: int = 32,
) -> np.ndarray:
    """
    Convert an interval of two rt values to a frame index interval.
    The length of the interval is rounded up so that a multiple of `optimize_size` cycles are included.

    Parameters
    ----------
    rt_values : np.ndarray, shape = (2,), dtype = float32
        Array of rt values.
    rt_values_array : np.ndarray
        Array containing all rt values for searching.
    zeroth_frame : int
        Indicator if the first frame is zero.
    cycle_len : int
        The size of the cycle dimension.
    precursor_cycle_max_index : int
        Maximum index for precursor cycles.
    optimize_size : int, default = 16
        Optimize for FFT efficiency by using multiples of this size.
    min_size : int, default = 32
        Minimum number of DIA cycles to include.

    Returns
    -------
    np.ndarray, shape = (1, 3), dtype = int64
        Array of frame indices (start, stop, 1)
    """
    if rt_values.shape != (2,):
        raise ValueError("rt_values must be a numpy array of shape (2,)")

    frame_index = np.searchsorted(rt_values_array, rt_values, "left")

    precursor_cycle_limits = (frame_index + zeroth_frame) // cycle_len
    precursor_cycle_len = precursor_cycle_limits[1] - precursor_cycle_limits[0]

    # Apply minimum size
    optimal_len = max(precursor_cycle_len, min_size)
    # Round up to the next multiple of `optimize_size`
    optimal_len = int(optimize_size * math.ceil(optimal_len / optimize_size))

    # By default, extend the precursor cycle to the right
    optimal_cycle_limits = np.array(
        [precursor_cycle_limits[0], precursor_cycle_limits[0] + optimal_len],
        dtype=np.int64,
    )

    # If the cycle is too long, extend it to the left
    if optimal_cycle_limits[1] > precursor_cycle_max_index:
        optimal_cycle_limits[1] = precursor_cycle_max_index
        optimal_cycle_limits[0] = precursor_cycle_max_index - optimal_len

        if optimal_cycle_limits[0] < 0:
            optimal_cycle_limits[0] = 0 if precursor_cycle_max_index % 2 == 0 else 1

    # Convert back to frame indices
    frame_limits = optimal_cycle_limits * cycle_len + zeroth_frame
    return make_slice_1d(frame_limits)

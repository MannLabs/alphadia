import logging
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

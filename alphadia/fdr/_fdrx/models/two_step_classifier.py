"""Implements the Two Step Classifier for use within the Alphadia framework."""

import logging

import numpy as np
import pandas as pd

from alphadia.fdr.fdr import get_q_values, keep_best

logger = logging.getLogger()


def get_target_count(df: pd.DataFrame) -> int:
    """Counts the number of target (non-decoy) entries in a DataFrame."""
    return len(df[(df["decoy"] == 0)])


def compute_q_values(
    df: pd.DataFrame, group_columns: list[str] | None = None
) -> pd.DataFrame:
    """Compute q-values for each entry after keeping only best entries per group."""
    df.sort_values(
        ["proba", "precursor_idx"], ascending=True, inplace=True
    )  # last sort to break ties
    df = keep_best(df, group_columns=group_columns)
    return get_q_values(df, "proba", "decoy")


def filter_by_qval(df: pd.DataFrame, fdr_cutoff: float) -> pd.DataFrame:
    """Filter dataframe by q-value threshold. If no entries pass the threshold, return the single target entry with lowest q-value."""
    df_filtered = df[df["qval"] < fdr_cutoff]

    if len(df_filtered) == 0:
        df_targets = df[df["decoy"] == 0]
        df_filtered = df_targets.loc[[df_targets["qval"].idxmin()]]

    return df_filtered


def compute_and_filter_q_values(
    df: pd.DataFrame,
    fdr: float,
    group_columns: list[str] | None = None,
    *,  # This line makes all following arguments keyword-only
    remove_decoys: bool = True,
) -> pd.DataFrame:
    """Returns entries in the DataFrame based on the FDR threshold and optionally removes decoy entries.

    If no entries are found below the FDR threshold after filtering, returns the single best entry based on the q-value.
    """
    df = compute_q_values(df, group_columns)
    if remove_decoys:
        df = df[df["decoy"] == 0]
    return filter_by_qval(df, fdr)


def get_target_decoy_partners(
    reference_df: pd.DataFrame, full_df: pd.DataFrame, group_by: list[str] | None = None
) -> pd.DataFrame:
    """Identifies and returns the corresponding target and decoy partner rows in full_df given the subset reference_df.

    This function is typically used to find target-decoy partners based on certain criteria like rank and elution group index.

    Parameters
    ----------
    reference_df : pd.DataFrame
        A subset DataFrame that contains reference values for matching.
    full_df : pd.DataFrame
        The main DataFrame from which rows will be matched against reference_df.
    group_by : list[str] | None, optional
        The columns to group by when performing the match. Defaults to ['rank', 'elution_group_idx'] if None is provided.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing rows from full_df that match the grouping criteria.

    """
    if group_by is None:
        group_by = ["rank", "elution_group_idx"]
    valid_tuples = reference_df[group_by]

    return full_df.merge(valid_tuples, on=group_by, how="inner")


def apply_absolute_transformations(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Applies absolute value transformations to predefined columns in a DataFrame inplace.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be transformed.
    columns : list of str, optional
        List of column names to transform. Defaults to ['delta_rt', 'top_3_ms2_mass_error', 'mean_ms2_mass_error'].

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.

    """
    if columns is None:
        columns = ["delta_rt", "top_3_ms2_mass_error", "mean_ms2_mass_error"]

    for col in columns:
        if col in df.columns:
            df[col] = np.abs(df[col])
        else:
            logger.warning(
                f"column '{col}' is not present in df, therefore abs() was not applied."
            )

    return df

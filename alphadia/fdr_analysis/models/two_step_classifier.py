import logging

import numpy as np
import pandas as pd

from alphadia.fdr import get_q_values, keep_best
from alphadia.fdrexperimental import Classifier

logger = logging.getLogger()


class TwoStepClassifier:
    def __init__(
        self,
        first_classifier: Classifier,
        second_classifier: Classifier,
        train_on_top_n: int = 1,
        first_fdr_cutoff: float = 0.6,
        second_fdr_cutoff: float = 0.01,
    ):
        """
        A two-step classifier, designed to refine classification results by applying a stricter second-stage classification after an initial filtering stage.

        Parameters
        ----------
        first_classifier : Classifier
            The first classifier used to initially filter the data.
        second_classifier : Classifier
            The second classifier used to further refine or confirm the classification based on the output from the first classifier.
        train_on_top_n : int, default=1
            The number of top candidates that are considered for training of the second classifier.
        first_fdr_cutoff : float, default=0.6
            The fdr threshold for the first classifier, determining how selective the first classification step is.
        second_fdr_cutoff : float, default=0.01
            The fdr threshold for the second classifier, typically set stricter to ensure high confidence in the final classification results.

        """
        self.first_classifier = first_classifier
        self.second_classifier = second_classifier
        self.first_fdr_cutoff = first_fdr_cutoff
        self.second_fdr_cutoff = second_fdr_cutoff

        self.train_on_top_n = train_on_top_n

    def fit_predict(
        self,
        df: pd.DataFrame,
        x_cols: list[str],
        y_col: str = "decoy",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Train the two-step classifier and predict resulting precursors, returning a DataFrame of only the predicted precursors.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame from which predictions are to be made.
        x_cols : list[str]
            List of column names representing the features to be used for prediction.
        y_col : str, optional
            The name of the column that denotes the target variable, by default 'decoy'.
        group_columns : list[str] | None, optional
            List of column names to group by for fdr calculations;. If None, fdr calculations will not be grouped.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the predicted precursors.

        """
        df.dropna(subset=x_cols, inplace=True)
        df = apply_absolute_transformations(df)

        if self.first_classifier.fitted:
            X = df[x_cols].to_numpy()
            df["proba"] = self.first_classifier.predict_proba(X)[:, 1]
            df_subset = get_entries_below_fdr(
                df, self.first_fdr_cutoff, group_columns, remove_decoys=False
            )

            self.second_classifier.epochs = 50

            df_train = df_subset
            df_predict = df_subset

        else:
            df_train = df[df["rank"] < self.train_on_top_n]
            df_predict = df

        self.second_classifier.fit(
            df_train[x_cols].to_numpy().astype(np.float32),
            df_train[y_col].to_numpy().astype(np.float32),
        )
        X = df_predict[x_cols].to_numpy()
        df_predict["proba"] = self.second_classifier.predict_proba(X)[:, 1]
        df_predict = get_entries_below_fdr(
            df_predict, self.second_fdr_cutoff, group_columns, remove_decoys=False
        )

        df_targets = df_predict[df_predict["decoy"] == 0]

        self.update_first_classifier(
            df=get_target_decoy_partners(df_predict, df),
            x_cols=x_cols,
            y_col=y_col,
            group_columns=group_columns,
        )

        return df_targets

    def update_first_classifier(
        self,
        df: pd.DataFrame,
        x_cols: list[str],
        y_col: str,
        group_columns: list[str],
    ) -> None:
        """
        Update the first classifier only if it improves upon the previous version or if it has not been previously fitted.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the features and target.
        x_cols : list[str]
            List of column names representing the features.
        y_col : str
            Name of the column representing the target variable.
        group_columns : list[str]
            Columns used to group data for FDR calculation.

        """
        X = df[x_cols].to_numpy()
        y = df[y_col].to_numpy()

        previous_n_precursors = -1

        if self.first_classifier.fitted:
            df["proba"] = self.first_classifier.predict_proba(X)[:, 1]
            df_targets = get_entries_below_fdr(df, self.first_fdr_cutoff, group_columns)
            previous_n_precursors = len(df_targets)
            previous_state_dict = self.first_classifier.to_state_dict()

        self.first_classifier.fit(X, y)

        df["proba"] = self.first_classifier.predict_proba(X)[:, 1]
        df_targets = get_entries_below_fdr(df, self.first_fdr_cutoff, group_columns)
        current_n_precursors = len(df_targets)

        if previous_n_precursors > current_n_precursors:
            self.first_classifier.from_state_dict(previous_state_dict)

    @property
    def fitted(self) -> bool:
        """Return whether both classifiers have been fitted."""
        return self.second_classifier.fitted

    def to_state_dict(self) -> dict:
        """Save classifier state.

        Returns
        -------
        dict
            State dictionary containing both classifiers
        """
        return {
            "first_classifier": self.first_classifier.to_state_dict(),
            "second_classifier": self.second_classifier.to_state_dict(),
            "first_fdr_cutoff": self.first_fdr_cutoff,
            "second_fdr_cutoff": self.second_fdr_cutoff,
            "train_on_top_n": self.train_on_top_n,
        }

    def from_state_dict(self, state_dict: dict) -> None:
        """Load classifier state.

        Parameters
        ----------
        state_dict : dict
            State dictionary containing both classifiers
        """
        self.first_classifier.from_state_dict(state_dict["first_classifier"])
        self.second_classifier.from_state_dict(state_dict["second_classifier"])
        self.first_fdr_cutoff = state_dict["first_fdr_cutoff"]
        self.second_fdr_cutoff = state_dict["second_fdr_cutoff"]
        self.train_on_top_n = state_dict["train_on_top_n"]


def get_entries_below_fdr(
    df: pd.DataFrame, fdr: float, group_columns: list[str], remove_decoys: bool = True
) -> pd.DataFrame:
    """
    Returns entries in the DataFrame based on the FDR threshold and optionally removes decoy entries.
    If no entries are found below the FDR threshold after filtering, returns the single best entry based on the q-value.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns 'proba', 'decoy', and any specified group columns.
    fdr : float
        The false discovery rate threshold for filtering entries.
    group_columns : list
        List of columns to group by when determining the best entries per group.
    remove_decoys : bool, optional
        Specifies whether decoy entries should be removed from the final result. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing entries below the specified FDR threshold, optionally excluding decoys.
    """
    df.sort_values("proba", ascending=True, inplace=True)
    df = keep_best(df, group_columns=group_columns)
    df = get_q_values(df, "proba", "decoy")

    df_subset = df[df["qval"] < fdr]
    if remove_decoys:
        df_subset = df_subset[df_subset["decoy"] == 0]

    # Handle case where no entries are below the FDR threshold
    if len(df_subset) == 0:
        df = df[df["decoy"] == 0]
        df_subset = df.loc[[df["qval"].idxmin()]]

    return df_subset


def get_target_decoy_partners(
    reference_df: pd.DataFrame, full_df: pd.DataFrame, group_by: list[str] | None = None
) -> pd.DataFrame:
    """
    Identifies and returns the corresponding target and decoy wartner rows in full_df given the subset reference_df/
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
    matching_rows = full_df.merge(valid_tuples, on=group_by, how="inner")

    return matching_rows


def apply_absolute_transformations(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Applies absolute value transformations to predefined columns in a DataFrame inplace.

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
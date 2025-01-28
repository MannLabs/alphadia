"""Implements the Two Step Classifier for use within the Alphadia framework."""

import logging

import numpy as np
import pandas as pd

from alphadia.fdr import get_q_values, keep_best
from alphadia.fdrexperimental import Classifier

logger = logging.getLogger()


class TwoStepClassifier:
    """A two-step classifier, designed to refine classification results by applying a stricter second-stage classification after an initial filtering stage."""

    def __init__(  # noqa: PLR0913 Too many arguments in function definition (> 5)
        self,
        first_classifier: Classifier,
        second_classifier: Classifier,
        first_fdr_cutoff: float = 0.6,
        second_fdr_cutoff: float = 0.01,
        min_precursors_for_update: int = 5000,
        max_iterations: int = 5,
        train_on_top_n: int = 1,
    ):
        """Initializing a two-step classifier.

        Parameters
        ----------
        first_classifier : Classifier
            The first classifier used to initially filter the data.
        second_classifier : Classifier
            The second classifier used to further refine or confirm the classification based on the output from the first classifier.
        first_fdr_cutoff : float, default=0.6
            The fdr threshold for the first classifier, determining how selective the first classification step is.
        second_fdr_cutoff : float, default=0.01
            The fdr threshold for the second classifier, typically set stricter to ensure high confidence in the final classification results.
        min_precursors_for_update : int, default=5000
            The minimum number of precursors required to update the first classifier.
        max_iterations : int
            Maximum number of refinement iterations during training.
        train_on_top_n : int
            Use candidates up to this rank for training. During inference, all ranks are used.

        """
        self.first_classifier = first_classifier
        self.second_classifier = second_classifier
        self.first_fdr_cutoff = first_fdr_cutoff
        self.second_fdr_cutoff = second_fdr_cutoff

        self._min_precursors_for_update = min_precursors_for_update
        self._max_iterations = max_iterations
        self._train_on_top_n = train_on_top_n

    def fit_predict(
        self,
        df: pd.DataFrame,
        x_cols: list[str],
        y_col: str = "decoy",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Train the two-step classifier and predict precursors using an iterative approach.

        1. First iteration: Train neural network on top-n candidates.
        2. Subsequent iterations: Use linear classifier to filter data, then refine with neural network.
        3. Update linear classifier if enough high-confidence predictions are found, else break.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing features and target variable
        x_cols : list[str]
            Feature column names
        y_col : str, optional
            Target variable column name, defaults to 'decoy'
        group_columns : list[str] | None, optional
            Columns to group by for FDR calculations

        Returns
        -------
        pd.DataFrame
            DataFrame containing predictions and q-values

        """
        df = self._preprocess_data(df, x_cols)
        best_result = None
        best_precursor_count = -1

        for i in range(self._max_iterations):
            if self.first_classifier.fitted and i > 0:
                df_train, df_predict = self._apply_filtering_with_first_classifier(
                    df, x_cols, group_columns
                )
                self.second_classifier.epochs = 50
            else:
                df_train = df[df["rank"] < self._train_on_top_n]
                df_predict = df
                self.second_classifier.epochs = 10

            predictions = self._train_and_apply_second_classifier(
                df_train, df_predict, x_cols, y_col, group_columns
            )

            # Filter results and check for improvement
            df_filtered = filter_by_qval(predictions, self.second_fdr_cutoff)
            current_target_count = len(df_filtered[df_filtered["decoy"] == 0])

            if current_target_count < best_precursor_count:
                logger.info(
                    f"Stopping training after iteration {i}, "
                    f"due to decreased target count ({current_target_count} < {best_precursor_count})"
                )
                return best_result

            best_precursor_count = current_target_count
            best_result = predictions

            # Update first classifier if enough confident predictions
            if current_target_count > self._min_precursors_for_update:
                self._update_first_classifier(
                    df_filtered, df, x_cols, y_col, group_columns
                )
            else:
                logger.info(
                    f"Stopping fitting after {i+1} / {self._max_iterations} iterations due to insufficient detected precursors to update the first classifier."
                )
                break
        else:
            logger.info(
                f"Stopping fitting after reaching the maximum number of iterations: {self._max_iterations} / {self._max_iterations}."
            )

        return best_result

    def _preprocess_data(self, df: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
        """Prepare data by removing NaN values and applying absolute transformations."""
        df.dropna(subset=x_cols, inplace=True)
        return apply_absolute_transformations(df)

    def _apply_filtering_with_first_classifier(
        self, df: pd.DataFrame, x_cols: list[str], group_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply first classifier to filter data for the training of the second classifier."""
        df["proba"] = self.first_classifier.predict_proba(df[x_cols].to_numpy())[:, 1]

        filtered_df = compute_and_filter_q_values(
            df, self.first_fdr_cutoff, group_columns, remove_decoys=False
        )

        return filtered_df, filtered_df

    def _train_and_apply_second_classifier(
        self,
        train_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        x_cols: list[str],
        y_col: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Train second_classifier and apply it to get predictions."""
        self.second_classifier.fit(
            train_df[x_cols].to_numpy().astype(np.float32),
            train_df[y_col].to_numpy().astype(np.float32),
        )

        x = predict_df[x_cols].to_numpy().astype(np.float32)
        predict_df["proba"] = self.second_classifier.predict_proba(x)[:, 1]

        return compute_q_values(predict_df, group_columns)

    def _update_first_classifier(
        self,
        subset_df: pd.DataFrame,
        full_df: pd.DataFrame,
        x_cols: list[str],
        y_col: str,
        group_columns: list[str],
    ) -> None:
        """Update first classifier by finding and using target/decoy pairs.

        First extracts the corresponding target/decoy partners from the full dataset
        for each entry in the subset, then uses these pairs to retrain the classifier.
        """
        df_train = get_target_decoy_partners(subset_df, full_df)
        x_train = df_train[x_cols].to_numpy()
        y_train = df_train[y_col].to_numpy()

        df_predict = full_df
        x_predict = full_df[x_cols].to_numpy()

        previous_n_precursors = -1

        if self.first_classifier.fitted:
            df_predict["proba"] = self.first_classifier.predict_proba(x_predict)[:, 1]
            df_targets = compute_and_filter_q_values(
                df_predict, self.first_fdr_cutoff, group_columns
            )
            previous_n_precursors = len(df_targets)
            previous_state_dict = self.first_classifier.to_state_dict()

        self.first_classifier.fit(x_train, y_train)

        df_predict["proba"] = self.first_classifier.predict_proba(x_predict)[:, 1]
        df_targets = compute_and_filter_q_values(
            df_predict, self.first_fdr_cutoff, group_columns
        )
        current_n_precursors = len(df_targets)

        if previous_n_precursors > current_n_precursors:
            logger.info(
                f"Reverted the first classifier back to the previous version "
                f"(prev: {previous_n_precursors}, curr: {current_n_precursors})"
            )
            self.first_classifier.from_state_dict(previous_state_dict)
        else:
            logger.info("Fitted the second classifier")

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
            "train_on_top_n": self._train_on_top_n,
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
        self._train_on_top_n = state_dict["train_on_top_n"]


def compute_q_values(
    df: pd.DataFrame, group_columns: list[str] | None = None
) -> pd.DataFrame:
    """Compute q-values for each entry after keeping only best entries per group."""
    df.sort_values("proba", ascending=True, inplace=True)
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

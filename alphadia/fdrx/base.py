"""This module implements a base class for semisupervised FDR estimation using targets and decoys.
It is flexible with regards to the features, type of classifier and type of identifications (precursors, peptides, proteins).
"""

import logging

import numpy as np
import pandas as pd
import sklearn.base

from alphadia.fdrx.plotting import (
    _plot_fdr_curve,
    _plot_roc_curve,
    _plot_score_distribution,
)
from alphadia.fdrx.stats import add_q_values, get_pep, keep_best
from alphadia.fragcomp import FragmentCompetition

logger = logging.getLogger()


class TargetDecoyFDR:
    def __init__(
        self,
        classifier: sklearn.base.BaseEstimator,
        feature_columns: list,
        decoy_column: str = "decoy",
        competition_columns: list | None = None,
    ):
        """Target Decoy FDR estimation using a classifier.

        This class supports target decoy competition as well as fragment competition.

        Parameters
        ----------

        classifier : sklearn.base.BaseEstimator
            The classifier to use for target decoy estimation.

        feature_columns : list
            The columns to use as features for the classifier.

        decoy_column : str, default='decoy'
            The column to use as decoy information.

        competition_columns : list, default=[]
            Perform target decoy competition on these columns. Only the best PSM for each group will be kept.
        """

        self._classifier = classifier
        self._feature_columns = feature_columns
        self._decoy_column = decoy_column
        self._competition_columns = competition_columns or []

    def fit_classifier(self, psm_df: pd.DataFrame):
        """Fit the classifier on the PSMs.

        Parameters
        ----------

        psm_df : pd.DataFrame
            The dataframe containing the PSMs.
        """

        is_na_row = psm_df[self._feature_columns].isna().any(axis=1)
        logger.info(f"Removing {is_na_row.sum()} rows with missing values")

        X = psm_df.loc[~is_na_row, self._feature_columns].values
        y = psm_df.loc[~is_na_row, self._decoy_column].values

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2
        )
        self._classifier.fit(X_train, y_train)

        # evaluate classifier
        y_test_proba = self._classifier.predict_proba(X_test)[:, 1]
        y_train_proba = self._classifier.predict_proba(X_train)[:, 1]

        _plot_score_distribution(y_train, y_train_proba, y_test, y_test_proba)
        _plot_roc_curve(y_train, y_train_proba, y_test, y_test_proba)

    def predict_classifier(self, psm_df: pd.DataFrame):
        """Predict the decoy probability for the PSMs.

        Parameters
        ----------

        psm_df : pd.DataFrame
            The dataframe containing the PSMs.

        Returns
        -------
        np.ndarray
            The decoy probabilities for the PSMs with same shape and order as the input dataframe.
        """

        is_na_row = psm_df[self._feature_columns].isna().any(axis=1)
        X = psm_df.loc[~is_na_row, self._feature_columns].values

        # Prediction should have the same shape of input, even for NaN rows
        # We are therefore assigning a decoy probability of 1 to all rows with NaN values
        y_proba_full = np.ones(len(psm_df))
        y_proba = self._classifier.predict_proba(X)[:, 1]
        y_proba_full[~is_na_row] = y_proba
        return y_proba_full

    def predict_qval(
        self,
        psm_df: pd.DataFrame,
        fragments_df: pd.DataFrame | None = None,
        dia_cycle: np.ndarray | None = None,
        competition_heuristic: float = 0.10,
    ) -> pd.DataFrame:
        """Calculate q-values for scored identifications.

        Parameters
        ----------

        psm_df : pd.DataFrame
            The dataframe containing the PSMs.

        fragments_df : pd.DataFrame, default=None
            The dataframe containing the fragments.

        dia_cycle : np.ndarray, default=None
            The DIA cycle for the fragments.

        competition_heuristic : float, default=0.10
            The q-value threshold for fragment competition.
            Only precursors with q-values below this threshold will be considered for fragment competition.

        Returns
        -------

        pd.DataFrame
            The input dataframe with q-values and PEPs added.
        """

        psm_df = psm_df.copy()
        psm_df["decoy_proba"] = self.predict_classifier(psm_df)
        # normalize to a 1:1 target decoy proportion
        r_target_decoy = (psm_df[self._decoy_column] == 0).sum() / (
            psm_df[self._decoy_column] == 1
        ).sum()

        # normalize q-values based on proportion before competition
        if dia_cycle is not None and fragments_df is not None:
            psm_df = add_q_values(
                psm_df,
                score_column="decoy_proba",
                decoy_column=self._decoy_column,
                r_target_decoy=r_target_decoy,
            )
            fragment_competition = FragmentCompetition()
            psm_df = fragment_competition(
                psm_df[psm_df["qval"] < competition_heuristic], fragments_df, dia_cycle
            )

        psm_df = keep_best(psm_df, group_columns=self._competition_columns)
        psm_df = add_q_values(
            psm_df, "decoy_proba", self._decoy_column, r_target_decoy=r_target_decoy
        )

        # calulate PEP
        psm_df["pep"] = get_pep(
            psm_df, score_column="decoy_proba", decoy_column=self._decoy_column
        )

        _plot_fdr_curve(psm_df["qval"])
        return psm_df

    def fit_predict_qval(
        self,
        psm_df: pd.DataFrame,
        fragments_df: pd.DataFrame | None = None,
        cycle: np.ndarray | None = None,
    ):
        """Fit the classifier, predict the decoy probabilities and calculate q-values.

        Parameters
        ----------

        psm_df : pd.DataFrame
            The dataframe containing the PSMs.

        fragments_df : pd.DataFrame, default=None
            The dataframe containing the fragments.

        cycle : np.ndarray, default=None
            The DIA cycle for the fragments.

        Returns
        -------

        pd.DataFrame
            The input dataframe with q-values and PEPs added.
        """

        self.fit_classifier(psm_df)
        return self.predict_qval(psm_df, fragments_df, cycle)

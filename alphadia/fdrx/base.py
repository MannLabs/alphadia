# make it an abc
import abc
import sklearn.base
from typing import List
import pandas as pd
import numpy as np
from alphadia.fragcomp import FragmentCompetition
from alphadia.fdrx.stats import _fdr_to_q_values, _get_pep, _get_q_values, _keep_best
from alphadia.fdrx.plotting import (
    _plot_fdr_curve,
    _plot_roc_curve,
    _plot_score_distribution,
)

import logging

logger = logging.getLogger()


class TargetDecoyFDR:
    def __init__(
        self,
        classifier: sklearn.base.BaseEstimator,
        feature_columns: list,
        decoy_column: str = "decoy",
        competition_columns: list = [],
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
        self._competition_columns = competition_columns

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

        y_proba_full = np.ones(len(psm_df))
        y_proba = self._classifier.predict_proba(X)[:, 1]
        y_proba_full[~is_na_row] = y_proba
        return y_proba_full

    def predict_qval(self, psm_df, fragments_df=None, dia_cycle=None):
        psm_df["decoy_proba"] = self.predict_classifier(psm_df)
        # normalize to a 1:1 target decoy proportion
        r_target_decoy = (psm_df["decoy"] == 0).sum() / (psm_df["decoy"] == 1).sum()

        # normalize q-values based on proportion before competition
        if dia_cycle is not None and fragments_df is not None:
            psm_df = _get_q_values(
                psm_df,
                score_column="decoy_proba",
                decoy_column="decoy",
                r_target_decoy=r_target_decoy,
            )
            fragment_competition = FragmentCompetition()
            psm_df = fragment_competition(
                psm_df[psm_df["qval"] < 0.10], fragments_df, dia_cycle
            )

        psm_df = _keep_best(psm_df, group_columns=self._competition_columns)
        psm_df = _get_q_values(
            psm_df, "decoy_proba", "decoy", r_target_decoy=r_target_decoy
        )

        # calulate PEP
        psm_df["pep"] = _get_pep(
            psm_df, score_column="decoy_proba", decoy_column="decoy"
        )

        _plot_fdr_curve(psm_df["qval"])
        return psm_df

    def fit_predict_qval(self, psm_df, fragments_df=None, cycle=None):
        self.fit_classifier(psm_df)
        return self.predict_qval(psm_df, fragments_df, cycle)

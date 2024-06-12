# make it an abc
import abc
import sklearn.base
from typing import List
from alphadia.fragcomp import FragmentCompetition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import sklearn.base

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


def _keep_best(
    df: pd.DataFrame,
    score_column: str = "decoy_proba",
    group_columns: List[str] = ["channel", "precursor_idx"],
):
    """Keep the best PSM for each group of PSMs with the same precursor_idx.
    This function is used to select the best candidate PSM for each precursor.
    if the group_columns is set to ['channel', 'elution_group_idx'] then its used for target decoy competition.

    Parameters
    ----------

    df : pd.DataFrame
        The dataframe containing the PSMs.

    score_column : str
        The name of the column containing the score to use for the selection.

    group_columns : list[str], default=['channel', 'precursor_idx']
        The columns to use for the grouping.

    Returns
    -------

    pd.DataFrame
        The dataframe containing the best PSM for each group.
    """
    temp_df = df.reset_index(drop=True)
    temp_df = temp_df.sort_values(score_column, ascending=True)
    temp_df = temp_df.groupby(group_columns).head(1)
    temp_df = temp_df.sort_index().reset_index(drop=True)
    return temp_df


def _fdr_to_q_values(fdr_values: np.ndarray):
    """Converts FDR values to q-values.
    Takes a ascending sorted array of FDR values and converts them to q-values.
    for every element the lowest FDR where it would be accepted is used as q-value.

    Parameters
    ----------
    fdr_values : np.ndarray
        The FDR values to convert.

    Returns
    -------
    np.ndarray
        The q-values.
    """
    fdr_values_flipped = np.flip(fdr_values)
    q_values_flipped = np.minimum.accumulate(fdr_values_flipped)
    q_vals = np.flip(q_values_flipped)
    return q_vals


def _get_q_values(
    _df: pd.DataFrame,
    score_column: str = "decoy_proba",
    decoy_column: str = "decoy",
    qval_column: str = "qval",
    r_target_decoy: float = 1.0,
):
    """Calculates q-values for a dataframe containing PSMs.

    Parameters
    ----------

    _df : pd.DataFrame
        The dataframe containing the PSMs.

    score_column : str, default='proba'
        The name of the column containing the score to use for the selection.
        Ascending sorted values are expected.

    decoy_column : str, default='_decoy'
        The name of the column containing the decoy information.
        Decoys are expected to be 1 and targets 0.

    qval_column : str, default='qval'
        The name of the column to store the q-values in.

    Returns
    -------

    pd.DataFrame
        The dataframe containing the q-values in column qval.

    """
    EPSILON = 1e-6
    _df = _df.sort_values([score_column, score_column], ascending=True)
    target_values = 1 - _df[decoy_column].values
    decoy_cumsum = np.cumsum(_df[decoy_column].values)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum / (target_cumsum + EPSILON)
    _df[qval_column] = _fdr_to_q_values(fdr_values) * r_target_decoy
    return _df


def _get_pep(
    psm_df: pd.DataFrame,
    score_column: str = "decoy_proba",
    decoy_column: str = "decoy",
    score_std: float = 0.01,
    pep_granularity: int = 1000,
    kernel_size: int = 20,
):
    """Implementation of a very simple nonparametric PEP estimation using a gaussian kernel.

    Parameters
    ----------

    psm_df : pd.DataFrame
        The dataframe containing the PSMs.

    score_column : str, default='decoy_proba'
        The name of the column containing the score to use for the selection.

    decoy_column : str, default='decoy'
        The name of the column containing the decoy information.

    score_std : float, default=0.01
        The standard deviation of the gaussian kernel.

    pep_granularity : int, default=1000
        The number of bins to use for the score histogram.

    kernel_size : int, default=20
        The size of the kernel to use for the convolution.

    Returns
    -------

    np.ndarray
        The PEP values with same shape and order as the input dataframe.

    """

    score_bins = np.linspace(0, 1, pep_granularity)
    target_decoy = psm_df[decoy_column].values
    score = psm_df[score_column].values

    target_hist, _ = np.histogram(score[target_decoy == 0], bins=score_bins)
    decoy_hist, _ = np.histogram(score[target_decoy == 1], bins=score_bins)

    std_norm = score_std / (score_bins[1] - score_bins[0])
    # create a gaussian kernel of 0.01 width with 5 elements
    kernel_gaussian = np.exp(
        -(np.arange(-kernel_size, kernel_size + 1) ** 2) / (2 * std_norm**2)
    )

    # convolve the target and decoy histograms with the kernel
    target_hist = np.convolve(target_hist, kernel_gaussian, mode="same")
    decoy_hist = np.convolve(decoy_hist, kernel_gaussian, mode="same")

    # numerical stability
    EPSILON = 1e-6
    pep = decoy_hist / (target_hist + decoy_hist + EPSILON)

    return pep[np.digitize(score, score_bins) - 1]


def _plot_score_distribution(y_train, y_train_proba, y_test, y_test_proba):
    fig, ax = plt.subplots(figsize=(5, 5))
    scores = np.hstack([y_train_proba, y_test_proba])
    labels = np.hstack([y_train, y_test])

    ax.hist(scores[labels == 0], bins=50, alpha=0.5, label="target")
    ax.hist(scores[labels == 1], bins=50, alpha=0.5, label="decoy")

    ax.set_xlabel("Decoy Probability")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def _plot_roc_curve(y_train, y_train_proba, y_test, y_test_proba):
    fpr_train, tpr_train, _ = sklearn.metrics.roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = sklearn.metrics.roc_curve(y_test, y_test_proba)

    test_auc = sklearn.metrics.auc(fpr_test, tpr_test)
    train_auc = sklearn.metrics.auc(fpr_train, tpr_train)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr_train, tpr_train, label="train")
    ax.plot(fpr_test, tpr_test, label="test")
    ax.plot([0, 1], [0, 1], "--", color="grey")

    ax.text(1, 0.05, f"Train AUC: {train_auc:.2f}", ha="right", va="bottom")
    ax.text(1, 0.01, f"Test AUC: {test_auc:.2f}", ha="right", va="bottom")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


def _plot_fdr_curve(qval, qval_treshold=0.05):
    qval_plot = qval[qval < qval_treshold]
    ids = np.arange(0, len(qval_plot), 1)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(qval_plot, ids)

    ax.set_xlim(-0.001, 0.05)
    ax.set_xlabel("q-value")
    ax.set_ylabel("Number of identifications")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def test_pep():
    # given
    # uniform distribution of decoy probabilities across target and decoy should result in 0.5 PEP on average
    df = pd.DataFrame(
        {
            "decoy_proba": np.random.rand(1000),
            "decoy": np.stack([np.zeros(500), np.ones(500)]).flatten(),
        }
    )

    # when
    pep = _get_pep(df)

    # then
    assert len(pep) == 1000
    assert np.all(pep >= 0)
    assert np.all(pep <= 1)
    assert (np.mean(pep) - 0.5) < 0.1

    plt.hist(pep)

# native imports
import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# third party imports
import pandas as pd
import sklearn

logger = logging.getLogger()


def keep_best(
    df: pd.DataFrame,
    score_column: str = "proba",
    group_columns: list[str] | None = None,
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
    if group_columns is None:
        group_columns = ["channel", "precursor_idx"]
    temp_df = df.reset_index(drop=True)
    temp_df = temp_df.sort_values(score_column, ascending=True)
    temp_df = temp_df.groupby(group_columns).head(1)
    temp_df = temp_df.sort_index().reset_index(drop=True)
    return temp_df


def fdr_to_q_values(fdr_values: np.ndarray):
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


def q_values(
    scores: np.ndarray,
    decoy_labels: np.ndarray,
    # score_column : str = 'proba',
    # decoy_column : str = '_decoy',
    # qval_column : str = 'qval'
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

    decoy_labels = decoy_labels[scores.argsort()]
    target_values = 1 - decoy_labels
    decoy_cumsum = np.cumsum(decoy_labels)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum / target_cumsum
    return fdr_to_q_values(fdr_values)


def get_q_values(
    _df: pd.DataFrame,
    score_column: str = "proba",
    decoy_column: str = "_decoy",
    qval_column: str = "qval",
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
    _df = _df.sort_values([score_column, score_column], ascending=True)
    target_values = 1 - _df[decoy_column].values
    decoy_cumsum = np.cumsum(_df[decoy_column].values)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum / target_cumsum
    _df[qval_column] = fdr_to_q_values(fdr_values)
    return _df


def plot_fdr(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier: sklearn.base.BaseEstimator,
    qval: np.ndarray,
    figure_path: str | None = None,
    neptune_run=None,
):
    """Plots statistics on the fdr corrected PSMs.

    Parameters
    ----------

    X_train : np.ndarray
        The training data.

    X_test : np.ndarray
        The test data.

    y_train : np.ndarray
        The training labels.

    y_test : np.ndarray
        The test labels.

    classifier : sklearn.base.BaseEstimator
        The classifier used for the prediction.

    qval : np.ndarray
        The q-values of the PSMs.
    """

    y_test_proba = classifier.predict_proba(X_test)[:, 1]

    y_train_proba = classifier.predict_proba(X_train)[:, 1]

    fpr_test, tpr_test, thresholds_test = sklearn.metrics.roc_curve(
        y_test, y_test_proba
    )
    fpr_train, tpr_train, thresholds_train = sklearn.metrics.roc_curve(
        y_train, y_train_proba
    )

    auc_test = sklearn.metrics.auc(fpr_test, tpr_test)
    auc_train = sklearn.metrics.auc(fpr_train, tpr_train)

    logger.info(f"Test AUC: {auc_test:.3f}")
    logger.info(f"Train AUC: {auc_train:.3f}")

    auc_difference_percent = np.abs((auc_test - auc_train) / auc_train * 100)
    logger.info(f"AUC difference: {auc_difference_percent:.2f}%")
    if auc_difference_percent > 5:
        logger.warning("AUC difference > 5%. This may indicate overfitting.")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(fpr_test, tpr_test, label=f"Test AUC: {auc_test:.3f}")
    ax[0].plot(fpr_train, tpr_train, label=f"Train AUC: {auc_train:.3f}")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].legend()

    ax[1].set_xlim(0, 1)
    ax[1].hist(
        np.concatenate([y_test_proba[y_test == 0], y_train_proba[y_train == 0]]),
        bins=50,
        alpha=0.5,
        label="target",
    )
    ax[1].hist(
        np.concatenate([y_test_proba[y_test == 1], y_train_proba[y_train == 1]]),
        bins=50,
        alpha=0.5,
        label="decoy",
    )
    ax[1].set_xlabel("decoy score")
    ax[1].set_ylabel("precursor count")
    ax[1].legend()

    qval_plot = qval[qval < 0.05]
    ids = np.arange(0, len(qval_plot), 1)
    ax[2].plot(qval_plot, ids)
    ax[2].set_xlim(-0.001, 0.05)
    ax[2].set_xlabel("q-value")
    ax[2].set_ylabel("number of precursors")

    for axs in ax:
        # remove top and right spines
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        axs.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )

    fig.tight_layout()
    plt.show()

    if figure_path is not None:
        fig.savefig(os.path.join(figure_path, "fdr.pdf"))

    if neptune_run is not None:
        neptune_run["eval/fdr"].log(fig)

    plt.close()

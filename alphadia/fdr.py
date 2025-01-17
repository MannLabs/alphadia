# native imports
import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# third party imports
import pandas as pd
import sklearn

# alphadia imports
# alpha family imports
from alphadia import fragcomp

logger = logging.getLogger()


def perform_fdr(
    classifier: sklearn.base.BaseEstimator,
    available_columns: list[str],
    df_target: pd.DataFrame,
    df_decoy: pd.DataFrame,
    competetive: bool = False,
    group_channels: bool = True,
    figure_path: str | None = None,
    neptune_run=None,
    df_fragments: pd.DataFrame | None = None,
    dia_cycle: np.ndarray = None,
    fdr_heuristic: float = 0.1,
    **kwargs,
):
    """Performs FDR calculation on a dataframe of PSMs

    Parameters
    ----------

    classifier : sklearn.base.BaseEstimator
        A classifier that implements the fit and predict_proba methods

    available_columns : list[str]
        A list of column names that are available for the classifier

    df_target : pd.DataFrame
        A dataframe of target PSMs

    df_decoy : pd.DataFrame
        A dataframe of decoy PSMs

    competetive : bool
        Whether to perform competetive FDR calculation where only the highest scoring PSM in a target-decoy pair is retained

    group_channels : bool
        Whether to group PSMs by channel before performing competetive FDR calculation

    figure_path : str, default=None
        The path to save the FDR plot to

    neptune_run : neptune.run.Run, default=None
        The neptune run to log the FDR plot to

    reuse_fragments : bool, default=True
        Whether to reuse fragments for different precursors

    dia_cycle : np.ndarray, default=None
        The DIA cycle as provided by alphatims

    fdr_heuristic : float, default=0.1
        The FDR heuristic to use for the initial selection of PSMs before fragment competition

    Returns
    -------

    psm_df : pd.DataFrame
        A dataframe of PSMs with q-values and probabilities.
        The columns `qval` and `proba` are added to the input dataframes.
    """
    target_len, decoy_len = len(df_target), len(df_decoy)
    df_target.dropna(subset=available_columns, inplace=True)
    df_decoy.dropna(subset=available_columns, inplace=True)
    target_dropped, decoy_dropped = (
        target_len - len(df_target),
        decoy_len - len(df_decoy),
    )

    if target_dropped > 0:
        logger.warning(f"dropped {target_dropped} target PSMs due to missing features")

    if decoy_dropped > 0:
        logger.warning(f"dropped {decoy_dropped} decoy PSMs due to missing features")

    if (
        np.abs(len(df_target) - len(df_decoy)) / ((len(df_target) + len(df_decoy)) / 2)
        > 0.1
    ):
        logger.warning(
            f"FDR calculation for {len(df_target)} target and {len(df_decoy)} decoy PSMs"
        )
        logger.warning(
            "FDR calculation may be inaccurate as there is more than 10% difference in the number of target and decoy PSMs"
        )

    X_target = df_target[available_columns].values
    X_decoy = df_decoy[available_columns].values
    y_target = np.zeros(len(X_target))
    y_decoy = np.ones(len(X_decoy))

    X = np.concatenate([X_target, X_decoy])
    y = np.concatenate([y_target, y_decoy])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2
    )

    classifier.fit(X_train, y_train)

    psm_df = pd.concat([df_target, df_decoy])

    psm_df["_decoy"] = y

    if competetive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]

    psm_df["proba"] = classifier.predict_proba(X)[:, 1]
    psm_df.sort_values("proba", ascending=True, inplace=True)

    psm_df = get_q_values(psm_df, "proba", "_decoy")

    if dia_cycle is not None and dia_cycle.shape[2] <= 2:
        # use a FDR of 10% as starting point
        # if there are no PSMs with a FDR < 10% use all PSMs
        start_idx = psm_df["qval"].searchsorted(fdr_heuristic, side="left")
        if start_idx == 0:
            start_idx = len(psm_df)

        # make sure fragments are not reused
        if df_fragments is not None:
            if dia_cycle is None:
                raise ValueError(
                    "dia_cycle must be provided if reuse_fragments is False"
                )
            fragment_competition = fragcomp.FragmentCompetition()
            psm_df = fragment_competition(
                psm_df.iloc[:start_idx], df_fragments, dia_cycle
            )

    psm_df = keep_best(psm_df, group_columns=group_columns)
    psm_df = get_q_values(psm_df, "proba", "_decoy")

    plot_fdr(
        X_train,
        X_test,
        y_train,
        y_test,
        classifier,
        psm_df["qval"],
        figure_path=figure_path,
        neptune_run=neptune_run,
    )

    return psm_df


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

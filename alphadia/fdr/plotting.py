"""Plotting functionality for FDR."""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn

auc_difference_percent_warning_threshold = 5

qval_threshold = 0.05

logger = logging.getLogger()


def plot_fdr(  # noqa: PLR0913 # Too many arguments
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier: sklearn.base.BaseEstimator,
    qval: np.ndarray,
    figure_path: str | None = None,
) -> None:
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

    figure_path: str | None
        The path to the folder to save the figure to.

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
    if auc_difference_percent > auc_difference_percent_warning_threshold:
        logger.warning(
            f"AUC difference > {auc_difference_percent_warning_threshold}%. This may indicate overfitting."
        )

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

    qval_plot = qval[qval < qval_threshold]
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
            mpl.ticker.FuncFormatter(lambda x, _p: format(int(x), ","))
        )

    fig.tight_layout()

    if figure_path is not None:
        figure_path_ = Path(figure_path)
        i = 0
        file_path = figure_path_ / f"fdr_{i}.pdf"
        while file_path.exists():
            i += 1
            file_path = figure_path_ / f"fdr_{i}.pdf"
        fig.savefig(file_path)
    else:
        plt.show()
        plt.close()

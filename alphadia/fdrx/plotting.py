import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics


def _plot_score_distribution(
    y_train: np.ndarray,
    y_train_proba: np.ndarray,
    y_test: np.ndarray,
    y_test_proba: np.ndarray,
):
    """Plot the distribution of the scores for the target and decoy classes.

    Parameters
    ----------

    y_train : np.ndarray
        The target/decoy labels for the training set. decoy=1, target=0.

    y_train_proba : np.ndarray
        The probability scores for the decoy class for the training set.

    y_test : np.ndarray
        The target/decoy labels for the test set. decoy=1, target=0.

    y_test_proba : np.ndarray
        The probability scores for the decoy class for the test set.

    Returns
    -------

    fig : matplotlib.figure.Figure
        The figure object.

    ax : matplotlib.axes.Axes
        The axes object.

    """

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
    return fig, ax


def _plot_roc_curve(
    y_train: np.ndarray,
    y_train_proba: np.ndarray,
    y_test: np.ndarray,
    y_test_proba: np.ndarray,
):
    """Plot the ROC curve for the training and test set.

    Parameters
    ----------

    y_train : np.ndarray
        The target/decoy labels for the training set. decoy=1, target=0.

    y_train_proba : np.ndarray
        The probability scores for the decoy class for the training set.

    y_test : np.ndarray
        The target/decoy labels for the test set. decoy=1, target=0.

    y_test_proba : np.ndarray
        The probability scores for the decoy class for the test set.

    Returns
    -------

    fig : matplotlib.figure.Figure
        The figure object.

    ax : matplotlib.axes.Axes
        The axes object.

    """
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
    return fig, ax


def _plot_fdr_curve(qval: np.ndarray, qval_treshold: float = 0.05):
    """Plot the FDR curve.

    Parameters
    ----------

    qval : np.ndarray
        The q-values for the identifications.

    qval_treshold : float, optional
        The q-value threshold to plot up to (default is 0.05).

    Returns
    -------

    fig : matplotlib.figure.Figure
        The figure object.

    ax : matplotlib.axes.Axes
        The axes object.

    """
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
    return fig, ax

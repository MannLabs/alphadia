import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from alphadia.fdrx.plotting import (
    _plot_score_distribution,
    _plot_roc_curve,
    _plot_fdr_curve,
)


@patch("matplotlib.pyplot.show")
def test_plot_score_distribution(mock_show):
    # given
    y_train = np.random.choice([0, 1], 1000)
    y_train_proba = np.random.rand(1000)
    y_test = np.random.choice([0, 1], 1000)
    y_test_proba = np.random.rand(1000)

    # when
    fig, ax = _plot_score_distribution(y_train, y_train_proba, y_test, y_test_proba)

    # then
    mock_show.assert_called_once()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@patch("matplotlib.pyplot.show")
def test_plot_roc_curve(mock_show):
    # given
    y_train = np.random.choice([0, 1], 1000)
    y_train_proba = np.random.rand(1000)
    y_test = np.random.choice([0, 1], 1000)
    y_test_proba = np.random.rand(1000)

    # when
    fig, ax = _plot_roc_curve(y_train, y_train_proba, y_test, y_test_proba)

    # then
    mock_show.assert_called_once()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@patch("matplotlib.pyplot.show")
def test_plot_fdr_curve(mock_show):
    # given
    qval = np.random.rand(1000).cumsum() / 1000

    # when
    fig, ax = _plot_fdr_curve(qval)

    # then
    mock_show.assert_called_once()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

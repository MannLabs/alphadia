"""
Test the finetunemetrics module.
"""

import numpy as np
from alphadia import finetunemetrics
from scipy import stats, linalg
from math import isclose
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score


def get_regression_test_input():
    np.random.seed(1337)
    y_true = np.random.rand(10)
    y_pred = np.random.rand(10)

    test_inp = {"predicted": y_pred, "target": y_true}

    return test_inp


def get_classification_test_input():
    np.random.seed(1337)
    y_true = np.random.randint(0, 2, 10)
    ohe = np.zeros((10, 2))
    ohe[np.arange(10), y_true] = 1

    y_pred = np.random.rand(10)
    y_pred = np.vstack([1 - y_pred, y_pred]).T
    test_inp = {"predicted": y_pred, "target": ohe}
    return test_inp


def test_MetricAccumulator():
    """
    Test the MetricAccumulator class
    """

    # Given
    metric_accumulator = finetunemetrics.MetricAccumulator(name="mse")

    metrics = np.random.rand(10)
    # When
    for i, metric in enumerate(metrics):
        metric_accumulator.accumulate(epoch=i, loss=metric)
    # Then
    assert np.all(metric_accumulator.stats.loc[:, "mse"].values == metrics)


def test_LinearRegressionTestMetric():
    """
    Test the LinearRegressionTestMetric class
    """

    # Given
    test_inp = get_regression_test_input()
    # When
    metric = finetunemetrics.LinearRegressionTestMetric()
    results = metric.calculate_test_metric(epoch=0, test_input=test_inp)

    # Then
    assert isclose(
        results.loc[0, "test_r_square"],
        stats.linregress(test_inp["predicted"], test_inp["target"]).rvalue ** 2,
        abs_tol=1e-3,
    )
    assert isclose(
        results.loc[0, "test_slope"],
        stats.linregress(test_inp["predicted"], test_inp["target"]).slope,
        abs_tol=1e-3,
    )
    assert isclose(
        results.loc[0, "test_intercept"],
        stats.linregress(test_inp["predicted"], test_inp["target"]).intercept,
        abs_tol=1e-3,
    )


def test_AbsErrorPercentileTestMetric():
    """
    Test the AbsErrorPercentileTestMetric class
    """

    # Given
    test_inp = get_regression_test_input()

    percentile = 95
    # When
    metric = finetunemetrics.AbsErrorPercentileTestMetric(percentile=percentile)
    results = metric.calculate_test_metric(epoch=0, test_input=test_inp)

    # Then
    assert isclose(
        results.loc[0, f"abs_error_{percentile}th_percentile"],
        np.percentile(np.abs(test_inp["target"] - test_inp["predicted"]), percentile),
        abs_tol=1e-3,
    )


def test_L1LossTestMetric():
    """
    Test the L1LossTestMetric class
    """

    # Given
    test_inp = get_regression_test_input()

    # When
    metric = finetunemetrics.L1LossTestMetric()
    results = metric.calculate_test_metric(epoch=0, test_input=test_inp)

    # Then
    assert isclose(
        results.loc[0, "test_loss"],
        linalg.norm(test_inp["target"] - test_inp["predicted"], 1)
        / len(test_inp["target"]),
        abs_tol=1e-3,
    )


def test_CELossTestMetric():
    """
    Test the CELossTestMetric class
    """

    # Given
    test_inp = get_classification_test_input()

    # When
    metric = finetunemetrics.CELossTestMetric()
    results = metric.calculate_test_metric(epoch=0, test_input=test_inp)

    # Then
    assert isclose(
        results.loc[0, "test_loss"],
        log_loss(np.argmax(test_inp["target"], axis=1), test_inp["predicted"]),
        abs_tol=1e-3,
    )


def test_AccuracyTestMetric():
    """
    Test the AccuracyTestMetric class
    """

    # Given
    test_inp = get_classification_test_input()

    # When
    metric = finetunemetrics.AccuracyTestMetric()
    results = metric.calculate_test_metric(epoch=0, test_input=test_inp)

    # Then
    assert isclose(
        results.loc[0, "test_accuracy"],
        accuracy_score(
            np.argmax(test_inp["target"], axis=1),
            np.argmax(test_inp["predicted"], axis=1),
        ),
        abs_tol=1e-3,
    )


def test_PrecisionRecallTestMetric():
    """
    Test the PrecisionRecallTestMetric class
    """

    # Given
    test_inp = get_classification_test_input()

    # When
    metric = finetunemetrics.PrecisionRecallTestMetric()
    results = metric.calculate_test_metric(epoch=0, test_input=test_inp)

    # Then
    assert isclose(
        results.loc[0, "test_precision"],
        precision_score(
            np.argmax(test_inp["target"], axis=1),
            np.argmax(test_inp["predicted"], axis=1),
            average="macro",
        ),
        abs_tol=1e-3,
    )
    assert isclose(
        results.loc[0, "test_recall"],
        recall_score(
            np.argmax(test_inp["target"], axis=1),
            np.argmax(test_inp["predicted"], axis=1),
            average="macro",
        ),
        abs_tol=1e-3,
    )


def test_MetricManager():
    """
    Test the MetricManager class
    """
    # Given
    metric_manager = finetunemetrics.MetricManager(
        model_name="test_model",
        test_interval=2,
        test_metrics=[
            finetunemetrics.LinearRegressionTestMetric(),
            finetunemetrics.L1LossTestMetric(),
        ],
    )

    test_inp = get_regression_test_input()
    # When
    for _ in range(10):
        metric_manager.calculate_test_metric(test_inp)

    # Then
    df = metric_manager.get_stats()

    assert df.columns.tolist() == [
        "epoch",
        "test_r_square",
        "test_r",
        "test_slope",
        "test_intercept",
        "test_loss",
    ]

    assert df.shape[0] == 10


def test_lrAccumulation():
    """
    Test the learning rate accumulation of the metric manager
    """
    # Given
    lr = np.random.rand(10)

    metric_manager = finetunemetrics.MetricManager(
        model_name="test_model",
        test_interval=1,
        test_metrics=[
            finetunemetrics.LinearRegressionTestMetric(),
            finetunemetrics.L1LossTestMetric(),
        ],
    )

    # When
    for i in range(10):
        metric_manager.accumulate_learning_rate(epoch=i, lr=lr[i])

    # Then
    df = metric_manager.get_stats()

    assert "learning_rate" in df.columns.tolist()

    assert np.all(df.loc[:, "learning_rate"].values == lr)


def test_trainLossAccumulation():
    """
    Test the training loss accumulation of the metric manager
    """
    # Given
    train_loss = np.random.rand(10)

    metric_manager = finetunemetrics.MetricManager(
        model_name="test_model",
        test_interval=1,
        test_metrics=[
            finetunemetrics.LinearRegressionTestMetric(),
            finetunemetrics.L1LossTestMetric(),
        ],
    )

    # When
    for i in range(10):
        metric_manager.accumulate_training_loss(epoch=i, loss=train_loss[i])

    # Then
    df = metric_manager.get_stats()

    assert "train_loss" in df.columns.tolist()

    assert np.all(df.loc[:, "train_loss"].values == train_loss)

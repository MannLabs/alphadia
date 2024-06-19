"""
Test the metrics module.
"""

import numpy as np
from alphadia.transferlearning.metrics import (
    LinearRegressionTestMetric,
    AbsErrorPercentileTestMetric,
    L1LossTestMetric,
    CELossTestMetric,
    AccuracyTestMetric,
    PrecisionRecallTestMetric,
    MetricManager,
)
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


def test_LinearRegressionTestMetric():
    """
    Test the LinearRegressionTestMetric class
    """

    # Given
    test_inp = get_regression_test_input()
    # When
    metric = LinearRegressionTestMetric()
    results = metric.calculate_test_metric(
        epoch=0, test_input=test_inp, data_split="test", property_name="rt"
    )
    # Then
    assert isclose(
        results[results["metric_name"] == "r_square"]["value"].values[0],
        stats.linregress(test_inp["predicted"], test_inp["target"]).rvalue ** 2,
        abs_tol=1e-3,
    )
    assert isclose(
        results[results["metric_name"] == "slope"]["value"].values[0],
        stats.linregress(test_inp["predicted"], test_inp["target"]).slope,
        abs_tol=1e-3,
    )
    assert isclose(
        results[results["metric_name"] == "intercept"]["value"].values[0],
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
    metric = AbsErrorPercentileTestMetric(percentile=percentile)
    results = metric.calculate_test_metric(
        epoch=0, test_input=test_inp, data_split="test", property_name="rt"
    )

    # Then
    assert isclose(
        results[results["metric_name"] == f"abs_error_{percentile}th_percentile"][
            "value"
        ].values[0],
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
    metric = L1LossTestMetric()
    results = metric.calculate_test_metric(
        epoch=0, test_input=test_inp, data_split="test", property_name="rt"
    )
    # Then
    assert isclose(
        results[results["metric_name"] == "l1_loss"]["value"].values[0],
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
    metric = CELossTestMetric()
    results = metric.calculate_test_metric(
        epoch=0, test_input=test_inp, data_split="test", property_name="charge"
    )
    # Then
    assert isclose(
        results[results["metric_name"] == "ce_loss"]["value"].values[0],
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
    metric = AccuracyTestMetric()
    results = metric.calculate_test_metric(
        epoch=0, test_input=test_inp, data_split="test", property_name="charge"
    )

    # Then
    assert isclose(
        results[results["metric_name"] == "accuracy"]["value"].values[0],
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
    metric = PrecisionRecallTestMetric()
    results = metric.calculate_test_metric(
        epoch=0, test_input=test_inp, data_split="test", property_name="charge"
    )
    # Then
    assert isclose(
        results[results["metric_name"] == "precision"]["value"].values[0],
        precision_score(
            np.argmax(test_inp["target"], axis=1),
            np.argmax(test_inp["predicted"], axis=1),
            average="macro",
        ),
        abs_tol=1e-3,
    )
    assert isclose(
        results[results["metric_name"] == "recall"]["value"].values[0],
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
    metric_manager = MetricManager(
        test_metrics=[
            LinearRegressionTestMetric(),
            L1LossTestMetric(),
        ],
    )

    test_inp = get_regression_test_input()
    # When
    for i in range(10):
        metric_manager.calculate_test_metric(
            test_inp, epoch=i, data_split="test", property_name="rt"
        )

    # Then
    results = metric_manager.get_stats()

    assert results["metric_name"].unique().tolist() == [
        "r_square",
        "r",
        "slope",
        "intercept",
        "l1_loss",
    ]

    assert results.shape[0] == 50


def test_metricAccumulation():
    """
    Test a metric accumulation
    """
    # Given
    lr = np.random.rand(10)

    metric_manager = MetricManager(
        test_metrics=[
            LinearRegressionTestMetric(),
            L1LossTestMetric(),
        ],
    )

    # When
    for i in range(10):
        metric_manager.accumulate_metrics(
            epoch=i,
            metric=lr[i],
            metric_name="learning_rate",
            data_split="train",
            property_name="rt",
        )

    # Then
    results = metric_manager.get_stats()

    assert "learning_rate" in results["metric_name"].unique().tolist()

    assert np.all(
        results[results["metric_name"] == "learning_rate"]["value"].values == lr
    )

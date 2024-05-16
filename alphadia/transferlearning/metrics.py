import numpy as np
import pandas as pd
from typing import List
from peptdeep.utils import linear_regression
from peptdeep.model.ms2 import calc_ms2_similarity


class MetricAccumulator:
    """
    Accumulator for any metric.
    """

    def __init__(self, name: str):
        self.name = name
        self.stats = None

    def accumulate(self, epoch: int, loss: float):
        """
        Accumulate a metric at a given epoch.

        Parameters
        ----------
        epoch : int
            The epoch at which the metric was calculated.
        loss : float
            The value of the metric.

        """

        new_stats = pd.DataFrame({self.name: [loss]})
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        self.stats = (
            new_stats if self.stats is None else pd.concat([self.stats, new_stats])
        )


class TestMetricBase:
    """
    An Base class for test metrics. Test metrics are classes that calculate a metric on the test set at a given epoch
    and accumulate the metric over time for reporting.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns  # a list of column names for the stats dataframe
        self.stats = (
            None  # Stats is a pandas dataframe that stores the test metric over time
        )

    def _update_stats(self, new_stats: pd.DataFrame, epoch: int):
        """
        Update the stats dataframe with new stats at a given epoch.

        Parameters
        ----------
        new_stats : pd.DataFrame
            A pandas dataframe containing the new stats.
        epoch : int
            The epoch at which the new stats were calculated.

        """
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        self.stats = (
            new_stats if self.stats is None else pd.concat([self.stats, new_stats])
        )

    def calculate_test_metric(self, test_input: dict, epoch: int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.

        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        raise NotImplementedError


class LinearRegressionTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(
            columns=["test_r_square", "test_r", "test_slope", "test_intercept"]
        )

    def calculate_test_metric(self, test_input: dict, epoch: int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.
        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """

        predictions = test_input["predicted"]
        targets = test_input["target"]
        new_stats = linear_regression(predictions, targets)
        new_stats = pd.DataFrame(new_stats)
        new_stats.columns = self.columns
        self._update_stats(new_stats, epoch)

        return new_stats


class AbsErrorPercentileTestMetric(TestMetricBase):
    def __init__(self, percentile: int):
        super().__init__(columns=[f"abs_error_{percentile}th_percentile"])
        self.percentile = percentile

    def calculate_test_metric(self, test_input: dict, epoch: int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.

        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        abs_error = np.abs(predictions - targets)
        new_stats = pd.DataFrame(
            [np.percentile(abs_error, self.percentile)], columns=self.columns
        )
        self._update_stats(new_stats, epoch)

        return new_stats


class L1LossTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["test_loss"])

    def calculate_test_metric(self, test_input: dict, epoch: int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.

        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        l1_loss = np.mean(np.abs(predictions - targets))
        new_stats = pd.DataFrame([l1_loss], columns=self.columns)
        self._update_stats(new_stats, epoch)

        return new_stats


class Ms2SimilarityTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(
            columns=["test_pcc_mean", "test_cos_mean", "test_sa_mean", "test_spc_mean"]
        )
        self.metrics = ["PCC", "COS", "SA", "SPC"]

    def calculate_test_metric(
        self,
        test_input: dict,
        epoch: int,
    ):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.
            - "psm_df": A pandas dataframe containing the PSMs for the test set.

        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        psm_df = test_input["psm_df"]
        predicted_fragments = test_input["predicted"]
        target_fragments = test_input["target"]
        psm_df, _ = calc_ms2_similarity(
            psm_df=psm_df,
            predict_intensity_df=predicted_fragments,
            fragment_intensity_df=target_fragments,
        )
        metrics = psm_df[self.metrics]
        new_stats = pd.DataFrame(
            {
                "PCC-mean": [metrics["PCC"].median()],
                "COS-mean": [metrics["COS"].median()],
                "SA-mean": [metrics["SA"].median()],
                "SPC-mean": [metrics["SPC"].median()],
            }
        )

        self._update_stats(new_stats, epoch)

        return new_stats


class CELossTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["test_loss"])

    def calculate_test_metric(self, test_input: dict, epoch: int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.

        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        ce_loss = np.mean(-np.sum(targets * np.log(predictions), axis=1))
        new_stats = pd.DataFrame([ce_loss], columns=self.columns)
        self._update_stats(new_stats, epoch)

        return new_stats


class AccuracyTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["test_accuracy"])

    def calculate_test_metric(self, test_input: dict, epoch: int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.

        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)

        accuracy = np.mean(predictions == targets)
        new_stats = pd.DataFrame([accuracy], columns=self.columns)

        self._update_stats(new_stats, epoch)

        return new_stats


class PrecisionRecallTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["test_precision", "test_recall"])

    def calculate_test_metric(self, test_input: dict, epoch: int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.
        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        n_classes = predictions.shape[1]
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)

        confusion_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                confusion_matrix[i, j] = np.sum((predictions == i) & (targets == j))

        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

        new_stats = pd.DataFrame(
            np.array([np.mean(precision), np.mean(recall)]).reshape(1, 2),
            columns=self.columns,
        )

        self._update_stats(new_stats, epoch)
        return new_stats


class MetricManager:
    """
    A class for managing metrics. The MetricManager class is used to accumulate training loss, learning rate, and calculate test metrics over time.

    parameters
    ----------
    model_name : str
        The name of the model being trained/tested.
    test_interval : int
        The interval of epochs between two test iterations.
    test_metrics : List[TestMetricBase]
        A list of test metrics to calculate at each epoch.
    """

    def __init__(
        self,
        model_name: str,
        test_interval: int = 1,
        test_metrics: List[TestMetricBase] = None,
    ):
        self.model_name = model_name
        self.test_metrics = test_metrics
        self.training_loss_accumulators = MetricAccumulator("train_loss")
        self.lr_accumulator = MetricAccumulator("learning_rate")
        self.epoch = 0
        self.test_interval = test_interval

    def calculate_test_metric(self, test_inp: dict) -> pd.DataFrame:
        """
        Calculate the test metrics at the current epoch by calling the test method of each test metric passed to the MetricManager
        during initialization.

        Parameters
        ----------
        test_inp : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.
            - [Optional] "psm_df": A pandas dataframe containing the PSMs for the test set. This is currently only required for MS2 similarity metrics.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metrics at the current epoch.

        """
        result = pd.DataFrame()
        for test_metric in self.test_metrics:
            result = pd.concat(
                [result, test_metric.calculate_test_metric(test_inp, self.epoch)],
                axis=1,
            )
        self.epoch += self.test_interval
        return result

    def accumulate_training_loss(self, epoch: int, loss: float):
        """
        Accumulate the training loss at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch at which the loss was calculated.
        loss : float
            The value of the loss.

        """

        self.training_loss_accumulators.accumulate(epoch, loss)

    def accumulate_learning_rate(self, epoch: int, lr: float):
        """
        Accumulate the learning rate at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch at which the learning rate was calculated.
        lr : float
            The value of the learning rate.

        """
        self.lr_accumulator.accumulate(epoch, lr)

    def get_stats(self) -> pd.DataFrame:
        """
        Get the stats for the training loss and test metrics accumulated so far.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the training loss, learning rate and test metrics accumulated so far.

        """

        result = (
            self.training_loss_accumulators.stats
            if self.training_loss_accumulators.stats is not None
            else pd.DataFrame()
        )
        if self.lr_accumulator.stats is not None:
            result = pd.concat([result, self.lr_accumulator.stats], axis=1)
        for test_metric in self.test_metrics:
            stats = test_metric.stats
            result = pd.concat([result, stats], axis=1)
        result.reset_index(inplace=True)

        return result

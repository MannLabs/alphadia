import numpy as np
import pandas as pd
from peptdeep.model.ms2 import calc_ms2_similarity
from peptdeep.utils import linear_regression


class TestMetricBase:
    """
    An Base class for test metrics. Test metrics are classes that calculate a metric on the test set at a given epoch.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns  # a list of column names for the stats dataframe

    def _to_long_format(
        self, stats: pd.DataFrame, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
        """
        Convert the stats dataframe to a long format.

        Parameters
        ----------
        stats : pd.DataFrame
            A pandas dataframe containing the stats.
        epoch : int
            The epoch at which the stats were calculated.
        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".
        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the stats in a long format.

        """
        long = []
        for row in stats.iterrows():
            epoch = epoch
            for column in stats.columns:
                if column != "epoch":
                    subject = {
                        "data_split": data_split,
                        "epoch": epoch,
                        "property": property_name,
                        "metric_name": column,
                        "value": row[1][column],
                    }
                    long.append(subject)
        return pd.DataFrame(long)

    def calculate_test_metric(
        self, test_input: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch in the long format.

        """
        raise NotImplementedError


class LinearRegressionTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["r_square", "r", "slope", "intercept"])

    def calculate_test_metric(
        self, test_input: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the data_split. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"
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
        stats = self._to_long_format(new_stats, epoch, data_split, property_name)

        return stats


class AbsErrorPercentileTestMetric(TestMetricBase):
    def __init__(self, percentile: int):
        super().__init__(columns=[f"abs_error_{percentile}th_percentile"])
        self.percentile = percentile

    def calculate_test_metric(
        self, test_input: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch in the long format.

        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        abs_error = np.abs(predictions - targets)
        new_stats = pd.DataFrame(
            [np.percentile(abs_error, self.percentile)], columns=self.columns
        )
        stats = self._to_long_format(new_stats, epoch, data_split, property_name)

        return stats


class L1LossTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["l1_loss"])

    def calculate_test_metric(
        self, test_input: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch in the long format.

        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        l1_loss = np.mean(np.abs(predictions - targets))
        new_stats = pd.DataFrame([l1_loss], columns=self.columns)
        stats = self._to_long_format(new_stats, epoch, data_split, property_name)

        return stats


class Ms2SimilarityTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["pcc_mean", "cos_mean", "sa_mean", "spc_mean"])
        self.metrics = ["PCC", "COS", "SA", "SPC"]

    def calculate_test_metric(
        self,
        test_input: dict,
        epoch: int,
        data_split: str,
        property_name: str,
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch in the long format.
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

        stats = self._to_long_format(new_stats, epoch, data_split, property_name)

        return stats


class CELossTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["ce_loss"])

    def calculate_test_metric(
        self, test_input: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch in the long format.
        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        ce_loss = np.mean(-np.sum(targets * np.log(predictions), axis=1))
        new_stats = pd.DataFrame([ce_loss], columns=self.columns)
        stats = self._to_long_format(new_stats, epoch, data_split, property_name)

        return stats


class AccuracyTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["accuracy"])

    def calculate_test_metric(
        self, test_input: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch in the long format.
        """
        predictions = test_input["predicted"]
        targets = test_input["target"]
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)

        accuracy = np.mean(predictions == targets)
        new_stats = pd.DataFrame([accuracy], columns=self.columns)

        stats = self._to_long_format(new_stats, epoch, data_split, property_name)

        return stats


class PrecisionRecallTestMetric(TestMetricBase):
    def __init__(self):
        super().__init__(columns=["precision", "recall"])

    def calculate_test_metric(
        self, test_input: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch in the long format.
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

        stats = self._to_long_format(new_stats, epoch, data_split, property_name)
        return stats


class MetricManager:
    """
    A class for managing metrics. The MetricManager class is used to accumulate training loss, learning rate, and calculate test metrics over time.

    parameters
    ----------

    test_metrics : List[TestMetricBase]
        A list of test metrics to calculate at each epoch.
    """

    def __init__(
        self,
        test_metrics: list[TestMetricBase] = None,
    ):
        self.test_metrics = test_metrics
        self.all_stats = pd.DataFrame()

    def calculate_test_metric(
        self, test_inp: dict, epoch: int, data_split: str, property_name: str
    ) -> pd.DataFrame:
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

        epoch : int
            The epoch at which the test metric is calculated.

        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".

        property_name : str
            The name of the property. e.g. "charge", "rt"
        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metrics at the given epoch in the long format.

        """
        current_result = pd.DataFrame()
        for test_metric in self.test_metrics:
            new_stats = test_metric.calculate_test_metric(
                test_inp, epoch, data_split, property_name
            )
            current_result = pd.concat([current_result, new_stats])
        current_result.reset_index(drop=True, inplace=True)
        self.all_stats = pd.concat([self.all_stats, current_result])
        return current_result

    def accumulate_metrics(
        self,
        epoch: int,
        metric: float,
        metric_name: str,
        data_split: str,
        property_name: str,
    ) -> None:
        """
        Accumulate a metric at a given epoch.

        Parameters
        ----------
        epoch : int
            The epoch at which the metric was calculated.
        metric : float
            The value of the metric.
        metric_name : str
            The name of the metric.
        data_split : str
            The name of the dataset. e.g. "train", "validation", "test".
        property_name : str
            The name of the property. e.g. "charge", "rt"

        """
        self.all_stats = pd.concat(
            [
                self.all_stats,
                pd.DataFrame(
                    {
                        "data_split": data_split,
                        "epoch": epoch,
                        "property": property_name,
                        "metric_name": metric_name,
                        "value": metric,
                    },
                    index=[0],
                ),
            ]
        )

    def get_stats(self) -> pd.DataFrame:
        """
        Get the stats for the training loss and test metrics accumulated so far. The stats are returned as pandas dataframe in the long format.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the stats for the training loss and test metrics accumulated so far.
        """
        return self.all_stats.reset_index(drop=True)

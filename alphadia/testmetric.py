import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from peptdeep.utils import logging,linear_regression
from peptdeep.model.ms2 import calc_ms2_similarity




    
    
class MetricAccumulator():
    """
    Accumulator for any metric. 
    """
    def __init__(self,name:str):
        self.name = name
        self.columns = [name]
        self.stats = None
    
    def accumulate(self, epoch:int, loss:float):
        """
        Accumulate a metric at a given epoch.

        Parameters
        ----------
        epoch : int
            The epoch at which the metric was calculated.
        loss : float
            The value of the metric.

        """
        
        new_stats = pd.DataFrame({
            self.name: [loss]
        })
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])



class TestMetricInterface():
    """
    An interface for test metrics. Test metrics are classes that calculate a metric on the test set at a given epoch
    and accumulate the metric over time for plotting and reporting.
    """
    def __init__(self, name:str):
        self.name = name
        self.columns = None # a list of column names for the stats dataframe
        # Stats is a pandas dataframe that stores the test metric over time
        self.stats = None


    def test(self,test_input:dict, epoch:int):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.
            - [Optional] "psm_df": A pandas dataframe containing the PSMs for the test set. This is currently only required for MS2 similarity metrics.

        epoch : int
            The epoch at which the test metric is calculated.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the test metric at the given epoch.

        """
        raise NotImplementedError
    
 

class LinearRegressionTestMetric(TestMetricInterface):
    def __init__(self):
        super().__init__("Linear regression")
        self.columns = ['test_r_square', 'test_r', 'test_slope', 'test_intercept']
    
    def test(self, test_input:dict, epoch:int):
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
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])

        return new_stats

    

class AbsErrorPercentileTestMetric(TestMetricInterface):
    def __init__(self, percentile:int):
        super().__init__(f"Test Absolute error {percentile}th percentile")
        self.percentile = percentile
        self.columns = [f"test_abs_error_{self.percentile}th_percentile"]

    def test(self, test_input:dict ,epoch:int):
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
        new_stats = pd.DataFrame({
            f"abs_error_{self.percentile}th_percentile": [np.percentile(abs_error, self.percentile)]
        })
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])

        return new_stats
    



class L1LossTestMetric(TestMetricInterface):
    def __init__(self):
        super().__init__("L1 loss")
        self.columns = ["test_loss"]
    
    def test(self, test_input:dict ,epoch:int):
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
        new_stats = pd.DataFrame({
            self.name: [l1_loss]
        })
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])

        return new_stats
    



class Ms2SimilarityTestMetric(TestMetricInterface):
    def __init__(self):
        super().__init__("MS2 similarity")
        self.metrics = ['PCC', 'COS', 'SA', 'SPC']
        self.columns = ["test_pcc_mean","test_cos_mean","test_sa_mean","test_spc_mean"]
    
    def test(self, test_input:dict, epoch:int, ):
        """
        Calculate the test metric at a given epoch.

        Parameters
        ----------
        test_input : dict
            A dictionary containing the test input data. The dictionary should contain the following keys:
            - "predicted": A numpy array of predicted values.
            - "target": A numpy array of target values.
            - "psm_df": A pandas dataframe containing the PSMs for the test set. This is currently only required for MS2 similarity metrics.

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
        psm_df, _ = calc_ms2_similarity(psm_df=psm_df, predict_intensity_df=predicted_fragments, fragment_intensity_df=target_fragments)
        metrics= psm_df[self.metrics]
        new_stats = pd.DataFrame({
            "PCC-mean": [metrics["PCC"].median()],
            "COS-mean": [metrics["COS"].median()],
            "SA-mean": [metrics["SA"].median()],
            "SPC-mean": [metrics["SPC"].median()]
        })
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])

        return new_stats

    


class CELossTestMetric(TestMetricInterface):
    def __init__(self):
        super().__init__("CE loss")
        self.columns = ["test_loss"]
    
    def test(self, test_input:dict ,epoch:int):
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
        new_stats = pd.DataFrame({
            self.name: [ce_loss]
        })
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])

        return new_stats


class AccuracyTestMetric(TestMetricInterface):
    def __init__(self):
        super().__init__("Accuracy")
        self.columns = ["test_accuracy"]
    
    def test(self, test_input:dict,epoch:int):
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
        #Predictions are probabilities, so we need to convert them to class labels
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)

        accuracy = np.mean(predictions == targets)
        new_stats = pd.DataFrame({
            self.name: [accuracy]
        })
        new_stats.index = [epoch]
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])

        return new_stats
   

class PrecisionRecallTestMetric(TestMetricInterface):
    def __init__(self):
        super().__init__("Precision and recall")
        self.columns = ["test_precision", "test_recall"]

    def test(self, test_input:dict,epoch:int):
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
        # Number of classes 
        n_classes = predictions.shape[1]
        #Predictions are probabilities, so we need to convert them to class labels
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)

        confusion_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                confusion_matrix[i, j] = np.sum((predictions == i) & (targets == j))

        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

        new_stats = pd.DataFrame({
            "precision": np.mean(precision),
            "recall": np.mean(recall)
        }, index=[epoch])
        new_stats.index.name = "epoch"
        new_stats.columns = self.columns
        if self.stats is None:
            self.stats = new_stats
        else:
            self.stats = pd.concat([self.stats, new_stats])

        return new_stats
    
    

class MetricManager:
    """
    A class for managing metrics. The MetricManager class is used to accumulate training loss and test metrics over time for plotting and reporting.
    """
    def __init__(self, model_name:str, test_interval:int = 1, tests:List[TestMetricInterface] = None):
        self.model_name = model_name
        # the standard we use is the first loss is the same loss used for training the model
        self.tests = tests
        self.training_loss_accumulators = MetricAccumulator("train_loss")
        self.lr_accumulator = MetricAccumulator("learning_rate")
        self.epoch = 0
        self.test_interval = test_interval


    def test(self, test_inp:dict)->pd.DataFrame:
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
        for test_metric in self.tests:
            result = pd.concat([result, test_metric.test(test_inp,self.epoch)], axis=1)
        self.epoch += self.test_interval
        return result
    
    def accumulate_training_loss(self, epoch:int, loss:float):
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

    def accumulate_learning_rate(self, epoch:int, lr:float):
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
    
    def get_stats(self)->pd.DataFrame:
        """
        Get the stats for the training loss and test metrics accumulated so far.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the training loss and test metrics accumulated so far.

        """

        result = self.training_loss_accumulators.stats if not self.training_loss_accumulators.stats.empty else pd.DataFrame()
        if not self.lr_accumulator.stats.empty:
            result = pd.concat([result, self.lr_accumulator.stats], axis=1)
        for test_metric in self.tests:
            stats = test_metric.stats
            result = pd.concat([result, stats], axis=1)
        # Make the epoch index normal column
        result.reset_index(inplace=True)

        return result
    

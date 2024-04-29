import torch
import pandas as pd
import numpy as np
import functools

from alphabase.spectral_library.flat import *

from alphadia.finetunemetrics import (
    MetricManager, L1LossTestMetric, LinearRegressionTestMetric, AbsErrorPercentileTestMetric,
    CELossTestMetric, AccuracyTestMetric, PrecisionRecallTestMetric, Ms2SimilarityTestMetric
)

from peptdeep.settings import global_settings
from peptdeep.pretrained_models import ModelManager
from peptdeep.model.model_interface import LR_SchedulerInterface, ModelInterface, CallbackHandler
from peptdeep.model.ms2 import  normalize_fragment_intensities
from peptdeep.model.charge import  ChargeModelForModAASeq
import logging

logger = logging.getLogger()


settings = {
    # --------- USer settings ------------
    "batch_size": 4000,
     "max_lr": 0.001,
     "train_ratio": 0.8,
    "test_interval": 1,
    "lr_patience": 3,
    # --------- Our settings ------------
    "minimum_psms": 1200,
    "start_lr": 0.001,
    "target_batch_size": 4000,
    "epochs": 51,
    "warmup_epochs": 5,

    #--------------------------
    "nce": 25 ,
    "instrument": "Lumos"

}

class CustomScheduler(LR_SchedulerInterface):
    """
    A Lr scheduler that includes a warmup phase and then a ReduceLROnPlateau scheduler.
    """

    def __init__(self,
        optimizer:torch.optim.Optimizer,
        start_lr:float=0.001,
        **kwargs
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = kwargs.get("num_warmup_steps", 5)
        self.num_training_steps = kwargs.get("num_training_steps", 50)
        self.lamda_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience= settings['lr_patience'],
            factor=0.5,
            verbose=True
        )
        self.start_lr = start_lr
        self.epoch = 0

    def step(self, epoch:int, loss:float)->float:
        """
        Get the learning rate for the next epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        loss : float
            The loss value of the current epoch.

        Returns
        -------
        float
            The learning rate for the next epoch.

        """
        self.epoch = epoch+1
        if self.epoch < self.num_warmup_steps:
            lr =  float((self.epoch)*self.start_lr) / float(max(1, self.num_warmup_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            return self.lamda_lr.step(loss)
                
    def get_last_lr(self):
        """
        Get the last learning rate.
        """
        return [self.optimizer.param_groups[0]["lr"]]

class EarlyStopping:
    def __init__(self, patience: int = 5 ,margin: float = 0.05):
        self.patience = patience
        self.best_loss = np.inf
        self.last_loss = np.inf
        self.margin = margin
        self.counter = 0

    def step(self, val_loss: float):
        if val_loss > self.best_loss*(1-self.margin) or abs(val_loss - self.last_loss)/self.last_loss < self.margin:
            self.counter += 1
            if self.counter >= self.patience:
                return False
        else:
            self.best_loss = val_loss
            self.counter = 0
        self.last_loss = val_loss
        return True
    
    def reset(self):
        self.best_loss = np.inf
        self.last_loss = np.inf
        self.counter = 0
    


class CustomCallbackHandler(CallbackHandler):
    def __init__(self, test_callback, **callback_args):
        super().__init__()
        self.test_callback = test_callback
        self.callback_args = callback_args
    
    def epoch_callback(self, epoch:int, epoch_loss:float):
        return self.test_callback(epoch, epoch_loss, **self.callback_args)

    

class FinetuneManager(ModelManager):
    def __init__(self, mask_modloss: bool = False, device: str = "gpu",settings: dict = {}):
        super().__init__(mask_modloss, device)
        self.settings = settings
        self.early_stopping = EarlyStopping(patience= (settings['lr_patience']/settings['test_interval'])*3)

    def test_ms2(self,
                epoch:int,
                epoch_loss:float,
                precursor_df: pd.DataFrame,
                target_fragment_intensity_df: pd.DataFrame,
                metricAccumulator: MetricManager,
                default_instrument:str = "Lumos",
                default_nce:float = 30.0,) -> bool:
        """
        Test the MS2 model using the PSM and matched fargment intensity dataframes and accumulate both the training loss and test metrics.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        epoch_loss : float
            The train loss value of the current epoch.
        precursor_df : pd.DataFrame
            The PSM dataframe.
        target_fragment_intensity_df : pd.DataFrame
            The matched fragment intensity dataframe.
        metricAccumulator : MetricManager
            The metric manager object.
        default_instrument : str
            The default instrument name.
        default_nce : float
            The default NCE value.

        Returns
        -------
        bool
            Whether to continue training or not based on the early stopping criteria.
    
        """
        
        if epoch % self.settings['test_interval'] == 0:
            metricAccumulator.accumulate_training_loss(epoch, epoch_loss)
            current_lr = self.ms2_model.optimizer.param_groups[0]["lr"]
            metricAccumulator.accumulate_learning_rate(epoch, current_lr)

            if "instrument" not in precursor_df.columns:
                precursor_df["instrument"] = default_instrument
            if "nce" not in precursor_df.columns:
                precursor_df["nce"] = default_nce
            columns = np.intersect1d(
                self.ms2_model.charged_frag_types,
                target_fragment_intensity_df.columns.values,
            )
    
            pred = self.ms2_model.predict(precursor_df)[columns]
            test_input = {
                "psm_df": precursor_df,
                "predicted": pred,
                "target": target_fragment_intensity_df[columns],
            }
            results = metricAccumulator.test(test_input)
            logger.progress(f" Epoch {epoch}: Lr:{round(current_lr,5)} Training loss: {round(epoch_loss,5)}, Test loss: {round(results['test_loss'].values[-1],5)}")
            continue_training = self.early_stopping.step(results["test_loss"].values[-1])
            return continue_training
    def finetune_ms2(self, psm_df: pd.DataFrame, matched_intensity_df: pd.DataFrame)->pd.DataFrame:
        """
        Fine tune the MS2 model using the PSM and matched fargment intensity dataframes.

        Parameters
        ----------
        psm_df : pd.DataFrame
            The PSM dataframe.
        matched_intensity_df : pd.DataFrame
            The matched fragment intensity dataframe.

        Returns
        -------
        pd.DataFrame
            Accumulated metrics during the fine tuning process.
        """

        
        # Shuffle the psm_df and split it into train and test
        train_psm_df = psm_df.sample(frac=self.settings['train_ratio'])
        test_psm_df = psm_df.drop(train_psm_df.index)

        tr_inten_df = pd.DataFrame()
        for frag_type in self.ms2_model.charged_frag_types:
            if frag_type in matched_intensity_df.columns:
                tr_inten_df[frag_type] = matched_intensity_df[frag_type]
            else:
                tr_inten_df[frag_type] = 0.0

        test_inten_df = tr_inten_df.copy()
        normalize_fragment_intensities(
            train_psm_df, tr_inten_df
        )
        normalize_fragment_intensities(
            test_psm_df, test_inten_df
        )

        self.set_default_nce_instrument(train_psm_df)
        self.set_default_nce_instrument(test_psm_df)


        # Create a metric manager
        test_metric_manager = MetricManager(model_name="ms2", 
                                            test_interval=self.settings['test_interval'],
                                            tests=[L1LossTestMetric(),Ms2SimilarityTestMetric()]
                                            )



        # create a callback handler
        callback_handler = CustomCallbackHandler(
            self.test_ms2,
            precursor_df=test_psm_df,
            target_fragment_intensity_df=test_inten_df,
            metricAccumulator=test_metric_manager,
        )
                                                 

        # set the callback handler
        self.ms2_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.ms2_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
        self.early_stopping.reset()

        # Train the model
        logger.progress(" Fine-tuning MS2 model")
        self.ms2_model.train(
            precursor_df= train_psm_df,
            fragment_intensity_df=tr_inten_df,
            epoch=self.settings['epochs'],
            batch_size=self.settings['batch_size'],
            warmup_epoch=self.settings['warmup_epochs'],
            lr=self.settings['start_lr'],
        )

        metrics = test_metric_manager.get_stats()
        # Print the last entry of all metrics
        msg =" Fine tuning finished at "
        for col in metrics.columns:
            msg += f" {col}: {round(metrics[col].values[-1],5)} \n"
        logger.progress(msg)

        return metrics
        

    def test_rt(self, 
                epoch:int,
                epoch_loss:float, 
                test_df: pd.DataFrame, 
                metricAccumulator: MetricManager)->bool:
        """
        Test the RT model using the PSM dataframe and accumulate both the training loss and test metrics.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        epoch_loss : float
            The train loss value of the current epoch.
        test_df : pd.DataFrame
            The PSM dataframe.
        metricAccumulator : MetricManager
            The metric manager object.

        Returns
        -------
        bool
            Whether to continue training or not based on the early stopping criteria.
        """
        
        if epoch % self.settings['test_interval'] == 0:
            metricAccumulator.accumulate_training_loss(epoch, epoch_loss)
            current_lr = self.rt_model.optimizer.param_groups[0]["lr"]
            metricAccumulator.accumulate_learning_rate(epoch, current_lr)
            pred = self.rt_model.predict(test_df)
            test_input = {
                "predicted": pred["rt_pred"].values,
                "target": test_df["rt_norm"].values,
            }
            results = metricAccumulator.test(test_input)
            logger.progress(f" Epoch {epoch}: Lr:{round(current_lr,5)} Training loss: {round(epoch_loss,5)}, Test loss: {round(results['test_loss'].values[-1],5)}")

            loss = results["test_loss"].values[-1]

            continue_training = self.early_stopping.step(loss)
            return continue_training

    def finetune_rt(self, psm_df: pd.DataFrame)->pd.DataFrame:
        """
        Fine tune the RT model using the PSM dataframe.

        Parameters
        ----------
        psm_df : pd.DataFrame
            The PSM dataframe.

        Returns
        -------
        pd.DataFrame
            Accumulated metrics during the fine tuning process.
        """
        
    
        # Shuffle the psm_df and split it into train and test
        train_df = psm_df.sample(frac=self.settings['train_ratio'])
        test_df = psm_df.drop(train_df.index)
        # Create a test metric manager 
        test_metric_manager = MetricManager(model_name="rt", 
                                            test_interval=self.settings['test_interval'],
                                            tests=[L1LossTestMetric(),LinearRegressionTestMetric(), AbsErrorPercentileTestMetric(95)])
   

        # Create a callback handler
        callback_handler = CustomCallbackHandler(
            self.test_rt,
            test_df=test_df,
            metricAccumulator=test_metric_manager
        )
        # Set the callback handler
        self.rt_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.rt_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
        self.early_stopping.reset()
        
        # Train the model
        logger.progress(" Fine-tuning RT model")
        self.rt_model.train(train_df,
                batch_size=self.settings['batch_size'],
                epoch=self.settings['epochs'],
                warmup_epoch=self.settings['warmup_epochs'],
                lr=self.settings['start_lr'],
            )
        
        metrics = test_metric_manager.get_stats()
        # Print the last entry of all metrics
        msg =" Fine tuning finished at "
        for col in metrics.columns:
            msg += f" {col}: {round(metrics[col].values[-1],5)} \n"
        logger.progress(msg)

        return metrics

    def test_charge(self, 
                    epoch:int,
                    epoch_loss:float,
                    test_df: pd.DataFrame,
                    metricAccumulator: MetricManager)->bool:
        """
        Test the charge model using the PSM dataframe and accumulate both the training loss and test metrics.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        epoch_loss : float
            The train loss value of the current epoch.
        test_df : pd.DataFrame
            The PSM dataframe.
        metricAccumulator : MetricManager
            The metric manager object.

        Returns
        -------
        bool
            Whether to continue training or not based on the early stopping criteria.
        """
        
        if epoch % self.settings['test_interval'] == 0:
            metricAccumulator.accumulate_training_loss(epoch, epoch_loss)
            current_lr = self.charge_model.optimizer.param_groups[0]["lr"]
            metricAccumulator.accumulate_learning_rate(epoch, current_lr)
            pred = self.charge_model.predict(test_df)
            test_inp   = {
                "target": np.array(test_df["charge_indicators"].values.tolist()),
                "predicted": np.array(pred["charge_probs"].values.tolist()),
            }
            results = metricAccumulator.test(test_inp)
            logger.progress(f" Epoch {epoch}: Lr:{round(current_lr,5)} Training loss: {round(epoch_loss,5)}, Test loss: {round(results['test_loss'].values[-1],5)}")

            loss = results["test_loss"].values[-1]

            continue_training = self.early_stopping.step(loss)
            return continue_training

    def finetune_charge(self,psm_df: pd.DataFrame)->pd.DataFrame:
        """
        Fine tune the charge model using the PSM dataframe.

        Parameters
        ----------
        psm_df : pd.DataFrame
            The PSM dataframe.

        Returns
        -------
        pd.DataFrame
            Accumulated metrics during the fine tuning process.
        """
        max_charge = np.max(psm_df["charge"])
        min_charge = np.min(psm_df["charge"])
        
        if self.charge_model is None:    
            self.charge_model = ChargeModelForModAASeq(max_charge=max_charge, min_charge=min_charge)
            self.charge_model.predict_batch_size = global_settings['model_mgr']['predict']['batch_size_charge']
            self.charge_prob_cutoff = global_settings['model_mgr']['charge_prob_cutoff']
            self.use_predicted_charge_in_speclib = global_settings['model_mgr']['use_predicted_charge_in_speclib']

        
        

        template_charge_indicators = np.zeros(max_charge-min_charge+1)
        all_possible_charge_indicators = {charge: template_charge_indicators.copy() for charge in range(min_charge, max_charge+1)}
        for charge in all_possible_charge_indicators:
            all_possible_charge_indicators[charge][charge-min_charge] = 1.0

        # map charge to a new column where the new column value is charge_indicators[charge-min_charge] = 1.0
        psm_df["charge_indicators"] = psm_df["charge"].map(all_possible_charge_indicators)

        
        # Shuffle the psm_df and split it into train and test
        train_df = psm_df.sample(frac=self.settings['train_ratio'])
        test_df = psm_df.drop(train_df.index)

        # Create a test metric manager 
        test_metric_manager = MetricManager(model_name="charge",
                                            test_interval=self.settings['test_interval'],
                                            tests=[CELossTestMetric(), AccuracyTestMetric(), PrecisionRecallTestMetric()])

        # Create a callback handler
        callback_handler = CustomCallbackHandler(
            self.test_charge,
            test_df=test_df,
            metricAccumulator=test_metric_manager
        )

        # Set the callback handler
        self.charge_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.charge_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
        self.early_stopping.reset()

        # Train the model
        logger.progress(" Fine-tuning Charge model")
        self.charge_model.train(psm_df,
                batch_size=self.settings['batch_size'],
                epoch=self.settings['epochs'],
                warmup_epoch=self.settings['warmup_epochs'],
                lr=self.settings['start_lr'],
            )

        metrics = test_metric_manager.get_stats()
        # Print the last entry of all metrics
        msg =" Fine tuning finished at "
        for col in metrics.columns:
            msg += f" {col}: {round(metrics[col].values[-1],5)} \n"
        logger.progress(msg)

        return metrics





            
            



    
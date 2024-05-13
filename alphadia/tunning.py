import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from alphabase.peptide.fragment import remove_unused_fragments
from alphabase.spectral_library.flat import *

from alphadia.finetunemetrics import (
    MetricManager,
    L1LossTestMetric,
    LinearRegressionTestMetric,
    AbsErrorPercentileTestMetric,
    CELossTestMetric,
    AccuracyTestMetric,
    PrecisionRecallTestMetric,
    Ms2SimilarityTestMetric,
)

from peptdeep.settings import global_settings
from peptdeep.pretrained_models import ModelManager
from peptdeep.model.model_interface import LR_SchedulerInterface, CallbackHandler
from peptdeep.model.ms2 import normalize_fragment_intensities
from peptdeep.model.charge import ChargeModelForModAASeq
import logging

logger = logging.getLogger()


settings = {
    # --------- USer settings ------------
    "batch_size": 1000,
    "max_lr": 0.0005,
    "train_ratio": 0.8,
    "test_interval": 2,
    "lr_patience": 3,
    # --------- Our settings ------------
    "minimum_psms": 1200,
    "epochs": 51,
    "warmup_epochs": 5,
    # --------------------------
    "nce": 25,
    "instrument": "Lumos",
}


class CustomScheduler(LR_SchedulerInterface):
    """
    A Lr scheduler that includes a warmup phase and then a ReduceLROnPlateau scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer object.
    start_lr : float
        The starting learning rate. Defaults to 0.001.
    **kwargs : dict
        Additional keyword arguments. It includes:
        - num_warmup_steps: int
            The number of warmup steps. Defaults to 5.
        - num_training_steps: int
            The number of training steps. Defaults to 50.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, **kwargs
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = kwargs.get("num_warmup_steps", 5)
        self.num_training_steps = kwargs.get("num_training_steps", 50)
        self.reduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=settings["lr_patience"],
            factor=0.5,
            verbose=True,
        )
        self.warmupLr = LambdaLR(optimizer, self._warmup)

    def _warmup(self, epoch: int):
        """
        Warmup the learning rate.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        return float(epoch+1) / float(max(1, self.num_warmup_steps))
    
    def step(self, epoch: int, loss: float) -> float:
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
        if epoch < self.num_warmup_steps:
            self.warmupLr.step(epoch)
        else:
            self.reduceLROnPlateau.step(loss)

    def get_last_lr(self):
        """
        Get the last learning rate.
        """
        return [self.optimizer.param_groups[0]["lr"]]


class EarlyStopping:
    """
    A class to implement early stopping based on the validation loss.
    Checks if the validation loss is not improving for a certain number of epochs (patience).
    """

    def __init__(self, patience: int = 5, margin: float = 0.05):
        self.patience = patience
        self.best_loss = np.inf
        self.last_loss = np.inf
        self.margin = margin
        self.counter = 0

    def step(self, val_loss: float):
        """
        Step the early stopping criteria and see if the training should continue.

        Parameters
        ----------
        val_loss : float
            The validation loss value.

        Returns
        -------
        continue_training : bool
            Whether to continue training or not based on the early stopping criteria.
        """
        if (
            val_loss > self.best_loss * (1 - self.margin)
            or abs(val_loss - self.last_loss) / self.last_loss < self.margin
        ):
            self.counter += 1
            if self.counter >= self.patience:
                return False
        else:
            self.best_loss = val_loss
            self.counter = 0
        self.last_loss = val_loss
        return True

    def reset(self):
        """
        Reset the early stopping criteria.
        """
        self.best_loss = np.inf
        self.last_loss = np.inf
        self.counter = 0


class CustomCallbackHandler(CallbackHandler):
    """
    A custom callback handler that allows for setting callback methods for the
    training loop implemented in the model Interface.

    parameters
    ----------
    test_callback : function
        The test callback function to be called after each epoch.
    **callback_args : dict
        The arguments to pass to the test callback function.
    """

    def __init__(self, test_callback, **callback_args):
        super().__init__()
        self.test_callback = test_callback
        self.callback_args = callback_args

    def epoch_callback(self, epoch: int, epoch_loss: float):
        """
        The epoch callback method that is called after each epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        epoch_loss : float
            The loss value of the current epoch.

        Returns
        -------
        bool: continue_training
            Whether to continue training or not based on the early stopping criteria.
        """
        return self.test_callback(epoch, epoch_loss, **self.callback_args)
    


class FinetuneManager(ModelManager):
    """
    FinetuneManager class that handles the fine tuning of the models. It inherits from the ModelManager class which is used to intialize and manage the loading and saving of the models.
    The finetune manager implements the fine tuning of the MS2, RT and Charge models.

    Parameters
    ----------
    mask_modloss : bool
        Whether to mask the modification loss or not. defaults to False.
    device : str
        The device to use for training the models. defaults to "gpu".
    settings : dict
        The settings for the fine tuning process.

    """

    def __init__(
        self, mask_modloss: bool = False, device: str = "gpu", settings: dict = {}
    ):
        super().__init__(mask_modloss, device)
        self.settings = settings
        self.early_stopping = EarlyStopping(
            patience=(settings["lr_patience"] // settings["test_interval"]) * 3
        )
    def _order_intensities(self, precursor_df_target: pd.DataFrame, precursor_df_pred: pd.DataFrame, target_intensity_df: pd.DataFrame,pred_intensity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rearrange the predicted fragment intensities to match the order used by the start and stop indices in the precursor_df_target.
        This is used because when the fragment intensities are predicted, peptdeep changes the frag_start_idx and frag_stop_idx, 
        so with this function we can directly compare the predicted intensities with the target intensities.

        Parameters
        ----------
        precursor_df_target : pd.DataFrame
            The PSM dataframe before prediction.
        precursor_df_pred : pd.DataFrame
            The PSM dataframe after prediction.
        target_intensity_df : pd.DataFrame
            The target fragment intensity dataframe.
        pred_intensity_df : pd.DataFrame
            The predicted fragment intensity dataframe.

        Returns
        -------
        pd.DataFrame
            The rearranged predicted fragment intensity dataframe.
        """
        new_pred = target_intensity_df.copy()
        for i in range(len(precursor_df_pred)):
            cur_mod_seq_hash = precursor_df_pred.iloc[i]["mod_seq_charge_hash"]
            cur_proba = precursor_df_pred.iloc[i]["proba"]

            # find the index of the the same mod_seq_hash and proba in the precursor_df
            target = precursor_df_target[(precursor_df_target["mod_seq_charge_hash"] == cur_mod_seq_hash) & (precursor_df_target["proba"] == cur_proba)]
            target_idx = target.index[0]

            
            pred_start_idx = precursor_df_pred.iloc[i]["frag_start_idx"]
            pred_end_idx = precursor_df_pred.iloc[i]["frag_stop_idx"]

            target_start_idx = precursor_df_target.loc[target_idx]["frag_start_idx"]
            target_end_idx = precursor_df_target.loc[target_idx]["frag_stop_idx"]

            new_pred.iloc[target_start_idx:target_end_idx, :] = pred_intensity_df.iloc[pred_start_idx:pred_end_idx, :]
        return new_pred
    
    def _test_ms2(
        self,
        epoch: int,
        epoch_loss: float,
        precursor_df: pd.DataFrame,
        target_fragment_intensity_df: pd.DataFrame,
        metricAccumulator: MetricManager,
        default_instrument: str = "Lumos",
        default_nce: float = 30.0,
    ) -> bool:
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
        continue_training = True
        if epoch % self.settings["test_interval"] == 0:
            self.ms2_model.model.eval()
            
            metricAccumulator.accumulate_training_loss(epoch, epoch_loss)
            current_lr = 0
            current_lr = self.ms2_model.optimizer.param_groups[0]["lr"]
            metricAccumulator.accumulate_learning_rate(epoch, current_lr)
            if "instrument" not in precursor_df.columns:
                precursor_df["instrument"] = default_instrument
            if "nce" not in precursor_df.columns:
                precursor_df["nce"] = default_nce
          
            precursor_copy = precursor_df.copy()
            pred_intensities = self.ms2_model.predict(precursor_copy)


             # Lets rearrange the prediction to have the same order as the target
            ordered_pred = self._order_intensities(precursor_df, precursor_copy, target_fragment_intensity_df, pred_intensities)
                   
            test_input = {
                "psm_df": precursor_df,
                "predicted": ordered_pred,
                "target": target_fragment_intensity_df,
            }
            results = metricAccumulator.test(test_input)
            # Using zero padded strings and 4 decimal places
            logger.progress(
                f" Epoch {epoch:<3} Lr: {current_lr:.5f}   Training loss: {epoch_loss:.4f}   Test loss: {results['test_loss'].values[-1]:.4f}"

            )
            continue_training = self.early_stopping.step(
                results["test_loss"].values[-1]
            )
            self.ms2_model.model.train()
        return continue_training
    def _normalize_intensity(self, precursor_df: pd.DataFrame, fragment_intensity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the fragment intensity dataframe inplace.

        Parameters
        ----------
        precursor_df : pd.DataFrame
            The PSM dataframe.
        fragment_intensity_df : pd.DataFrame
            The fragment intensity dataframe.

        Returns
        -------
        pd.DataFrame
            The normalized fragment intensity dataframe.
        """
        for start, stop in zip(precursor_df['frag_start_idx'], precursor_df['frag_stop_idx']):
            iloc_slice = fragment_intensity_df.iloc[start:stop, :]
            max_intensity = np.max(iloc_slice.values.flatten())
            if np.isnan(max_intensity):
                fragment_intensity_df.iloc[start:stop, :] = 0
                continue
            if max_intensity != 0:
                fragment_intensity_df.iloc[start:stop, :] = iloc_slice / max_intensity
                
    
    def finetune_ms2(
        self, psm_df: pd.DataFrame, matched_intensity_df: pd.DataFrame
    ) -> pd.DataFrame:
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
        self._normalize_intensity(psm_df, matched_intensity_df)

        # Shuffle the psm_df and split it into train and test
        train_psm_df = psm_df.sample(frac=self.settings["train_ratio"]).copy()
        test_psm_df = psm_df.drop(train_psm_df.index).copy()


        tr_inten_df = pd.DataFrame()
        for frag_type in self.ms2_model.charged_frag_types:
            if frag_type in matched_intensity_df.columns:
                tr_inten_df[frag_type] = matched_intensity_df[frag_type]
            else:
                tr_inten_df[frag_type] = 0.0

        test_inten_df = tr_inten_df.copy()

        self.set_default_nce_instrument(train_psm_df)
        self.set_default_nce_instrument(test_psm_df)

        train_psm_df, tr_inten_df = remove_unused_fragments(train_psm_df, [tr_inten_df])
        tr_inten_df = tr_inten_df[0]
        test_psm_df, test_inten_df = remove_unused_fragments(test_psm_df, [test_inten_df])
        test_inten_df = test_inten_df[0]

        # Create a metric manager
        test_metric_manager = MetricManager(
            model_name="ms2",
            test_interval=self.settings["test_interval"],
            tests=[L1LossTestMetric(), Ms2SimilarityTestMetric()],
        )

        # create a callback handler
        callback_handler = CustomCallbackHandler(
            self._test_ms2,
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
        self.ms2_model.model.train()
        self.ms2_model.train(
            precursor_df=train_psm_df,
            fragment_intensity_df=tr_inten_df,
            epoch=self.settings["epochs"],
            batch_size=self.settings["batch_size"],
            warmup_epoch=self.settings["warmup_epochs"],
            lr=settings["max_lr"],
            verbose=True,
        )

        metrics = test_metric_manager.get_stats()
        # Print the last entry of all metrics
        msg = " Fine tuning finished at "
        for col in metrics.columns:
            msg += f" {col}: {round(metrics[col].values[-1],5)} \n"
        logger.progress(msg)

        return metrics

    def _test_rt(
        self,
        epoch: int,
        epoch_loss: float,
        test_df: pd.DataFrame,
        metricAccumulator: MetricManager,
    ) -> bool:
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
        continue_training = True
        if epoch % self.settings["test_interval"] == 0:
            self.rt_model.model.eval()
            metricAccumulator.accumulate_training_loss(epoch, epoch_loss)
            current_lr = self.rt_model.optimizer.param_groups[0]["lr"]
            metricAccumulator.accumulate_learning_rate(epoch, current_lr)
            pred = self.rt_model.predict(test_df)
            test_input = {
                "predicted": pred["rt_pred"].values,
                "target": test_df["rt_norm"].values,
            }
            results = metricAccumulator.test(test_input)
            logger.progress(
                f" Epoch {epoch:<3} Lr: {current_lr:.5f}   Training loss: {epoch_loss:.4f}   Test loss: {results['test_loss'].values[-1]:.4f}"
            )

            loss = results["test_loss"].values[-1]

            continue_training = self.early_stopping.step(loss)
            self.rt_model.model.train()
        return continue_training

    def finetune_rt(self, psm_df: pd.DataFrame) -> pd.DataFrame:
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
        train_df = psm_df.sample(frac=self.settings["train_ratio"])
        test_df = psm_df.drop(train_df.index)
        # Create a test metric manager
        test_metric_manager = MetricManager(
            model_name="rt",
            test_interval=self.settings["test_interval"],
            tests=[
                L1LossTestMetric(),
                LinearRegressionTestMetric(),
                AbsErrorPercentileTestMetric(95),
            ],
        )

        # Create a callback handler
        callback_handler = CustomCallbackHandler(
            self._test_rt, test_df=test_df, metricAccumulator=test_metric_manager
        )
        # Set the callback handler
        self.rt_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.rt_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
        self.early_stopping.reset()

        # Train the model
        logger.progress(" Fine-tuning RT model")
        self.rt_model.model.train()
        self.rt_model.train(
            train_df,
            batch_size=self.settings["batch_size"],
            epoch=self.settings["epochs"],
            warmup_epoch=self.settings["warmup_epochs"],
            lr=settings["max_lr"],
        )

        metrics = test_metric_manager.get_stats()
        # Print the last entry of all metrics
        msg = " Fine tuning finished at "
        for col in metrics.columns:
            msg += f" {col}: {round(metrics[col].values[-1],5)} \n"
        logger.progress(msg)

        return metrics

    def _test_charge(
        self,
        epoch: int,
        epoch_loss: float,
        test_df: pd.DataFrame,
        metricAccumulator: MetricManager,
    ) -> bool:
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
        continue_training = True
        if epoch % self.settings["test_interval"] == 0:
            self.charge_model.model.eval()
            metricAccumulator.accumulate_training_loss(epoch, epoch_loss)
            current_lr = self.charge_model.optimizer.param_groups[0]["lr"]
            metricAccumulator.accumulate_learning_rate(epoch, current_lr)
            pred = self.charge_model.predict(test_df)
            test_inp = {
                "target": np.array(test_df["charge_indicators"].values.tolist()),
                "predicted": np.array(pred["charge_probs"].values.tolist()),
            }
            results = metricAccumulator.test(test_inp)
            logger.progress(
                f" Epoch {epoch:<3} Lr: {current_lr:.5f}   Training loss: {epoch_loss:.4f}   Test loss: {results['test_loss'].values[-1]:.4f}"
            )

            loss = results["test_loss"].values[-1]

            continue_training = self.early_stopping.step(loss)
            self.charge_model.model.train()
        return continue_training

    def finetune_charge(self, psm_df: pd.DataFrame) -> pd.DataFrame:
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
            self.charge_model = ChargeModelForModAASeq(
                max_charge=max_charge, min_charge=min_charge
            )
            self.charge_model.predict_batch_size = global_settings["model_mgr"][
                "predict"
            ]["batch_size_charge"]
            self.charge_prob_cutoff = global_settings["model_mgr"]["charge_prob_cutoff"]
            self.use_predicted_charge_in_speclib = global_settings["model_mgr"][
                "use_predicted_charge_in_speclib"
            ]

        template_charge_indicators = np.zeros(max_charge - min_charge + 1)
        all_possible_charge_indicators = {
            charge: template_charge_indicators.copy()
            for charge in range(min_charge, max_charge + 1)
        }
        for charge in all_possible_charge_indicators:
            all_possible_charge_indicators[charge][charge - min_charge] = 1.0

        # map charge to a new column where the new column value is charge_indicators[charge-min_charge] = 1.0
        psm_df["charge_indicators"] = psm_df["charge"].map(
            all_possible_charge_indicators
        )

        # Shuffle the psm_df and split it into train and test
        train_df = psm_df.sample(frac=self.settings["train_ratio"])
        test_df = psm_df.drop(train_df.index)

        # Create a test metric manager
        test_metric_manager = MetricManager(
            model_name="charge",
            test_interval=self.settings["test_interval"],
            tests=[
                CELossTestMetric(),
                AccuracyTestMetric(),
                PrecisionRecallTestMetric(),
            ],
        )

        # Create a callback handler
        callback_handler = CustomCallbackHandler(
            self._test_charge, test_df=test_df, metricAccumulator=test_metric_manager
        )

        # Set the callback handler
        self.charge_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.charge_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
        self.early_stopping.reset()

        # Train the model
        logger.progress(" Fine-tuning Charge model")
        self.charge_model.model.train()
        self.charge_model.train(
            psm_df,
            batch_size=self.settings["batch_size"],
            epoch=self.settings["epochs"],
            warmup_epoch=self.settings["warmup_epochs"],
            lr=settings["max_lr"],
        )

        metrics = test_metric_manager.get_stats()
        # Print the last entry of all metrics
        msg = " Fine tuning finished at "
        for col in metrics.columns:
            msg += f" {col}: {round(metrics[col].values[-1],5)} \n"
        logger.progress(msg)

        return metrics

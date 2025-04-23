import logging

import numpy as np
import pandas as pd
import torch
from alphabase.peptide.fragment import remove_unused_fragments
from alphabase.peptide.mobility import ccs_to_mobility_for_df, mobility_to_ccs_for_df
from alphabase.peptide.precursor import refine_precursor_df
from peptdeep.model.charge import ChargeModelForModAASeq
from peptdeep.model.model_interface import CallbackHandler, LR_SchedulerInterface
from peptdeep.pretrained_models import ModelManager
from peptdeep.settings import global_settings
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from alphadia.transferlearning.metrics import (
    AbsErrorPercentileTestMetric,
    AccuracyTestMetric,
    CELossTestMetric,
    L1LossTestMetric,
    LinearRegressionTestMetric,
    MetricManager,
    Ms2SimilarityTestMetric,
    PrecisionRecallTestMetric,
)

logger = logging.getLogger()


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
        - lr_patience: int
            The patience for the ReduceLROnPlateau scheduler. Defaults to 3.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs):
        self._optimizer = optimizer
        self._num_warmup_steps = kwargs.get("num_warmup_steps", 5)
        self._num_training_steps = kwargs.get("num_training_steps", 50)
        self._lr_patience = kwargs.get("lr_patience", 3)
        self._reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self._lr_patience,
            factor=0.5,
        )
        self._warmup_lr = LambdaLR(optimizer, self._warmup)

    def _warmup(self, epoch: int):
        """
        Warmup the learning rate.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        return float(epoch + 1) / float(max(1, self._num_warmup_steps))

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
        if epoch < self._num_warmup_steps:
            self._warmup_lr.step()
        else:
            self._reduce_lr_on_plateau.step(loss)

    def get_last_lr(self):
        """
        Get the last learning rate.
        """
        return [self._optimizer.param_groups[0]["lr"]]


class EarlyStopping:
    """
    A class to implement early stopping based on the validation loss.
    Checks if the validation loss is not improving for a certain number of epochs (patience).
    """

    def __init__(self, patience: int = 5, margin: float = 0.01):
        self._patience = patience
        self._best_loss = np.inf
        self._last_loss = np.inf
        self._margin = margin
        self._counter = 0

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
        if self._last_loss != np.inf:
            if (
                val_loss > self._best_loss * (1 + self._margin)
                or abs(val_loss - self._last_loss) / self._last_loss < self._margin
            ):
                self._counter += 1
                if self._counter >= self._patience:
                    return False
            else:
                self._best_loss = val_loss
                self._counter = 0
        self._last_loss = val_loss
        return True

    def reset(self):
        """
        Reset the early stopping criteria.
        """
        self._best_loss = np.inf
        self._last_loss = np.inf
        self._counter = 0


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
        self._test_callback = test_callback
        self._callback_args = callback_args

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
        return self._test_callback(epoch, epoch_loss, **self._callback_args)


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
        self,
        mask_modloss: bool = False,
        device: str = "gpu",
        lr_patience: int = 3,
        test_interval: int = 1,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.1,
        epochs: int = 51,
        warmup_epochs: int = 5,
        batch_size: int = 1000,
        max_lr: float = 0.0005,
        nce: float = 25,
        instrument: str = "Lumos",
    ):
        super().__init__(mask_modloss, device)
        self._test_interval = test_interval
        self._train_fraction = train_fraction
        self._validation_fraction = validation_fraction
        self._test_fraction = test_fraction
        self._epochs = epochs
        self._warmup_epochs = warmup_epochs
        self._batch_size = batch_size
        self._max_lr = max_lr
        self.nce = nce
        self.instrument = instrument

        self.device = device
        self.early_stopping = EarlyStopping(patience=(lr_patience // test_interval) * 4)

        assert (
            self._train_fraction + self._validation_fraction + self._test_fraction
            <= 1.0
        ), "The sum of the train, validation and test fractions should be less than or equal to 1.0"

    def _reset_frag_idx(self, df):
        """
        Reset the frag_start_idx and frag_stop_idx of the dataframe so both columns will be monotonically increasing.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to reset the indices.

        Returns
        -------
        pd.DataFrame
            The dataframe with the reset indices.
        """
        new_df = df.copy()
        number_of_fragments = new_df["frag_stop_idx"] - new_df["frag_start_idx"]
        accumlated_frags = number_of_fragments.cumsum()

        new_frag_start_idx = accumlated_frags - number_of_fragments
        new_frag_stop_idx = accumlated_frags

        new_df["frag_start_idx"] = new_frag_start_idx
        new_df["frag_stop_idx"] = new_frag_stop_idx
        return new_df

    def _order_intensities(
        self,
        reordered_precursor_df: pd.DataFrame,
        unordered_precursor_df: pd.DataFrame,
        unordered_frag_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rearrange the fragment intensities to match the order used by the start and stop indices in reordered_precursor_df.
        The goal of this is to reorder the fragment intensities using a newer precursor_df that only has different start and stop indices.

        Parameters
        ----------
        reordered_precursor_df : pd.DataFrame
            The dataframe with the new frag_start_idx and frag_stop_idx to respect.
        unordered_precursor_df : pd.DataFrame
            The dataframe with the old frag_start_idx and frag_stop_idx.

        unordered_frag_df : pd.DataFrame
            The fragment intensity dataframe to be reordered.
        Returns
        -------
        pd.DataFrame
            The reordered fragment intensity dataframe.
        """
        reordered = unordered_frag_df.copy()
        for i in tqdm(range(len(reordered_precursor_df))):
            new_start_idx = reordered_precursor_df.iloc[i]["frag_start_idx"]
            new_end_idx = reordered_precursor_df.iloc[i]["frag_stop_idx"]

            old_start_idx = unordered_precursor_df.iloc[i]["frag_start_idx"]
            old_end_idx = unordered_precursor_df.iloc[i]["frag_stop_idx"]

            reordered.iloc[new_start_idx:new_end_idx, :] = unordered_frag_df.iloc[
                old_start_idx:old_end_idx, :
            ]
        return reordered

    def _accumulate_training_metrics(
        self,
        metric_accumulator: MetricManager,
        epoch: int,
        epoch_loss: float,
        current_lr: float,
        property_name: str,
    ):
        """
        Accumulate the training metrics (training loss and learning rate) for the given property.

        Parameters
        ----------
        metric_accumulator : MetricManager
            The metric manager object.
        epoch : int
            The current epoch number.
        epoch_loss : float
            The training loss value of the current epoch.
        current_lr : float
            The current learning rate.
        property_name : str
            The property name to accumulate the metrics for.
        """
        loss_name = "ce_loss" if property_name == "charge" else "l1_loss"
        metric_accumulator.accumulate_metrics(
            epoch,
            metric=epoch_loss,
            metric_name=loss_name,
            data_split="train",
            property_name=property_name,
        )
        metric_accumulator.accumulate_metrics(
            epoch,
            metric=current_lr,
            metric_name="lr",
            data_split="train",
            property_name=property_name,
        )

    def _evaluate_metrics(
        self,
        test_input: dict,
        metric_accumulator: MetricManager,
        epoch: int,
        data_split: str,
        property_name: str,
        epoch_loss: float,
        current_lr: float,
    ) -> bool:
        """
        Evaluate the model using the test_input, accumulate the metrics and return the continue_training flag based on the early stopping criteria.

        Parameters
        ----------
        test_input : dict
            The input data for calculating the metrics.
        metric_accumulator : MetricManager
            The metric manager object.
        epoch : int
            The current epoch number.
        data_split : str
            The dataset label to test on e.g. "validation", "train"
        property_name : str
            The property name to accumulate the metrics for.
        epoch_loss : float
            The training loss value of the current epoch.
        current_lr : float
            The current learning rate.

        Returns
        -------
        bool
            Whether to continue training or not based on the early stopping criteria applied on the metrics.
        """
        continue_training = True
        val_metrics = metric_accumulator.calculate_test_metric(
            test_input, epoch, data_split=data_split, property_name=property_name
        )
        if epoch != -1 and data_split == "validation":
            loss_name = "ce_loss" if property_name == "charge" else "l1_loss"
            self._accumulate_training_metrics(
                metric_accumulator, epoch, epoch_loss, current_lr, property_name
            )
            val_loss = val_metrics[val_metrics["metric_name"] == loss_name][
                "value"
            ].values[0]
            continue_training = self.early_stopping.step(val_loss)
            logger.progress(
                f" Epoch {epoch:<3} Lr: {current_lr:.5f}   Training loss: {epoch_loss:.4f}   validation loss: {val_loss:.4f}"
            )
        else:
            logger.progress(
                f" Model tested on {data_split} dataset with the following metrics:"
            )
            for i in range(len(val_metrics)):
                logger.progress(
                    f" {val_metrics['metric_name'].values[i]:<30}: {val_metrics['value'].values[i]:.4f}"
                )

        return continue_training

    def _test_ms2(
        self,
        epoch: int,
        epoch_loss: float,
        precursor_df: pd.DataFrame,
        target_fragment_intensity_df: pd.DataFrame,
        metric_accumulator: MetricManager,
        data_split: str,
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
        metric_accumulator : MetricManager
            The metric manager object.
        data_split : str
            The dataset label to test on e.g. "validation", "train"
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
        if epoch % self._test_interval and epoch != -1:
            return continue_training

        self.ms2_model.model.eval()
        if "instrument" not in precursor_df.columns:
            precursor_df["instrument"] = default_instrument
        if "nce" not in precursor_df.columns:
            precursor_df["nce"] = default_nce

        precursor_copy = precursor_df.copy()
        pred_intensities = self.ms2_model.predict(precursor_copy)

        test_input = {
            "psm_df": precursor_df,
            "predicted": pred_intensities,
            "target": target_fragment_intensity_df,
        }

        current_lr = (
            self.ms2_model.optimizer.param_groups[0]["lr"] if epoch != -1 else 0
        )
        continue_training = self._evaluate_metrics(
            test_input,
            metric_accumulator,
            epoch,
            data_split,
            "ms2",
            epoch_loss,
            current_lr,
        )

        self.ms2_model.model.train()
        return continue_training

    def _normalize_intensity(
        self, precursor_df: pd.DataFrame, fragment_intensity_df: pd.DataFrame
    ) -> pd.DataFrame:
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
        for start, stop in zip(
            precursor_df["frag_start_idx"], precursor_df["frag_stop_idx"], strict=True
        ):
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
        train_psm_df = psm_df.sample(frac=self._train_fraction).copy()
        val_psm_df = (
            psm_df.drop(train_psm_df.index)
            .sample(frac=self._validation_fraction / (1 - self._train_fraction))
            .copy()
        )
        test_psm_df = psm_df.drop(train_psm_df.index).drop(val_psm_df.index).copy()

        train_intensity_df = pd.DataFrame()
        for frag_type in self.ms2_model.charged_frag_types:
            if frag_type in matched_intensity_df.columns:
                train_intensity_df[frag_type] = matched_intensity_df[frag_type]
            else:
                train_intensity_df[frag_type] = 0.0

        val_intensity_df = train_intensity_df.copy()
        test_intensity_df = train_intensity_df.copy()

        self.set_default_nce_instrument(train_psm_df)
        self.set_default_nce_instrument(val_psm_df)
        self.set_default_nce_instrument(test_psm_df)

        train_psm_df, train_intensity_df = remove_unused_fragments(
            train_psm_df, [train_intensity_df]
        )
        train_intensity_df = train_intensity_df[0]

        val_psm_df, val_intensity_df = remove_unused_fragments(
            val_psm_df, [val_intensity_df]
        )
        val_intensity_df = val_intensity_df[0]

        test_psm_df, test_intensity_df = remove_unused_fragments(
            test_psm_df, [test_intensity_df]
        )
        test_intensity_df = test_intensity_df[0]

        # Prepare order for peptdeep prediction
        val_psm_df = refine_precursor_df(val_psm_df)
        reordered_val_psm_df = self._reset_frag_idx(val_psm_df)
        reordered_val_intensity_df = self._order_intensities(
            reordered_precursor_df=reordered_val_psm_df,
            unordered_precursor_df=val_psm_df,
            unordered_frag_df=val_intensity_df,
        )
        test_psm_df = refine_precursor_df(test_psm_df)
        reordered_test_psm_df = self._reset_frag_idx(test_psm_df)
        reordered_test_intensity_df = self._order_intensities(
            reordered_precursor_df=reordered_test_psm_df,
            unordered_precursor_df=test_psm_df,
            unordered_frag_df=test_intensity_df,
        )

        test_metric_manager = MetricManager(
            test_metrics=[L1LossTestMetric(), Ms2SimilarityTestMetric()],
        )

        callback_handler = CustomCallbackHandler(
            self._test_ms2,
            precursor_df=reordered_val_psm_df,
            target_fragment_intensity_df=reordered_val_intensity_df,
            metric_accumulator=test_metric_manager,
            data_split="validation",
        )

        self.ms2_model.set_callback_handler(callback_handler)

        self.ms2_model.set_lr_scheduler_class(CustomScheduler)

        self.early_stopping.reset()

        # Test the model before training
        self._test_ms2(
            -1,
            0,
            reordered_val_psm_df,
            reordered_val_intensity_df,
            test_metric_manager,
            data_split="validation",
        )
        # Train the model
        logger.progress(" Fine-tuning MS2 model with the following settings:")
        logger.info(
            f" Train fraction:      {self._train_fraction:3.2f}     Train size:      {len(train_psm_df):<10}"
        )
        logger.info(
            f" Validation fraction: {self._validation_fraction:3.2f}     Validation size: {len(val_psm_df):<10}"
        )
        logger.info(
            f" Test fraction:       {self._test_fraction:3.2f}     Test size:       {len(test_psm_df):<10}"
        )
        self.ms2_model.model.train()
        self.ms2_model.train(
            precursor_df=train_psm_df,
            fragment_intensity_df=train_intensity_df,
            epoch=self._epochs,
            batch_size=self._batch_size,
            warmup_epoch=self._warmup_epochs,
            lr=self._max_lr,
        )

        self._test_ms2(
            self._epochs,
            0,
            reordered_test_psm_df,
            reordered_test_intensity_df,
            test_metric_manager,
            data_split="test",
        )

        metrics = test_metric_manager.get_stats()

        return metrics

    def _test_rt(
        self,
        epoch: int,
        epoch_loss: float,
        test_df: pd.DataFrame,
        metric_accumulator: MetricManager,
        data_split: str,
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
        metric_accumulator : MetricManager
            The metric manager object.
        data_split : str
            The dataset label to test on. e.g. "validation", "train"
        Returns
        -------
        bool
            Whether to continue training or not based on the early stopping criteria.
        """
        continue_training = True
        if epoch % self._test_interval != 0 and epoch != -1:
            return continue_training

        self.rt_model.model.eval()

        pred = self.rt_model.predict(test_df)
        test_input = {
            "predicted": pred["rt_pred"].values,
            "target": test_df["rt_norm"].values,
        }
        current_lr = self.rt_model.optimizer.param_groups[0]["lr"] if epoch != -1 else 0
        continue_training = self._evaluate_metrics(
            test_input,
            metric_accumulator,
            epoch,
            data_split,
            "rt",
            epoch_loss,
            current_lr,
        )

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
        train_df = psm_df.sample(frac=self._train_fraction)
        val_df = psm_df.drop(train_df.index).sample(
            frac=self._validation_fraction / (1 - self._train_fraction)
        )
        test_df = psm_df.drop(train_df.index).drop(val_df.index)

        test_metric_manager = MetricManager(
            test_metrics=[
                L1LossTestMetric(),
                LinearRegressionTestMetric(),
                AbsErrorPercentileTestMetric(95),
            ],
        )

        callback_handler = CustomCallbackHandler(
            self._test_rt,
            test_df=val_df,
            metric_accumulator=test_metric_manager,
            data_split="validation",
        )
        self.rt_model.set_callback_handler(callback_handler)

        self.rt_model.set_lr_scheduler_class(CustomScheduler)

        self.early_stopping.reset()

        # Test the model before training
        self._test_rt(-1, 0, psm_df, test_metric_manager, data_split="all")
        # Train the model
        logger.progress(" Fine-tuning RT model with the following settings:")
        logger.info(
            f" Train fraction:      {self._train_fraction:3.2f}     Train size:      {len(train_df):<10}"
        )
        logger.info(
            f" Validation fraction: {self._validation_fraction:3.2f}     Validation size: {len(val_df):<10}"
        )
        logger.info(
            f" Test fraction:       {self._test_fraction:3.2f}     Test size:       {len(test_df):<10}"
        )
        self.rt_model.model.train()
        self.rt_model.train(
            train_df,
            batch_size=self._batch_size,
            epoch=self._epochs,
            warmup_epoch=self._warmup_epochs,
            lr=self._max_lr,
        )

        self._test_rt(self._epochs, 0, test_df, test_metric_manager, data_split="test")

        metrics = test_metric_manager.get_stats()

        return metrics

    def _test_charge(
        self,
        epoch: int,
        epoch_loss: float,
        test_df: pd.DataFrame,
        metric_accumulator: MetricManager,
        data_split: str,
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
        metric_accumulator : MetricManager
            The metric manager object.
        data_split : str
            The dataset label to test on. e.g. "validation", "train"

        Returns
        -------
        bool
            Whether to continue training or not based on the early stopping criteria.
        """
        continue_training = True
        if epoch % self._test_interval != 0 and epoch != -1:
            return continue_training

        self.charge_model.model.eval()

        pred = self.charge_model.predict(test_df)
        test_input = {
            "target": np.array(test_df["charge_indicators"].values.tolist()),
            "predicted": np.array(pred["charge_probs"].values.tolist()),
        }
        current_lr = (
            self.charge_model.optimizer.param_groups[0]["lr"] if epoch != -1 else 0
        )
        continue_training = self._evaluate_metrics(
            test_input,
            metric_accumulator,
            epoch,
            data_split,
            "charge",
            epoch_loss,
            current_lr,
        )
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
                max_charge=max_charge, min_charge=min_charge, device=self.device
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
        train_df = psm_df.sample(frac=self._train_fraction)
        val_df = psm_df.drop(train_df.index).sample(
            frac=self._validation_fraction / (1 - self._train_fraction)
        )
        test_df = psm_df.drop(train_df.index).drop(val_df.index)

        test_metric_manager = MetricManager(
            test_metrics=[
                CELossTestMetric(),
                AccuracyTestMetric(),
                PrecisionRecallTestMetric(),
            ],
        )

        callback_handler = CustomCallbackHandler(
            self._test_charge,
            test_df=val_df,
            metric_accumulator=test_metric_manager,
            data_split="validation",
        )

        self.charge_model.set_callback_handler(callback_handler)

        self.charge_model.set_lr_scheduler_class(CustomScheduler)

        self.early_stopping.reset()

        # Test the model before training
        self._test_charge(-1, 0, psm_df, test_metric_manager, data_split="all")

        # Train the model
        logger.progress(" Fine-tuning Charge model with following settings:")
        logger.info(
            f" Train fraction:      {self._train_fraction:3.2f}     Train size:      {len(train_df):<10}"
        )
        logger.info(
            f" Validation fraction: {self._validation_fraction:3.2f}     Validation size: {len(val_df):<10}"
        )
        logger.info(
            f" Test fraction:       {self._test_fraction:3.2f}     Test size:       {len(test_df):<10}"
        )
        self.charge_model.model.train()
        self.charge_model.train(
            train_df,
            batch_size=self._batch_size,
            epoch=self._epochs,
            warmup_epoch=self._warmup_epochs,
            lr=self._max_lr,
        )

        self._test_charge(
            self._epochs, 0, test_df, test_metric_manager, data_split="test"
        )
        metrics = test_metric_manager.get_stats()

        return metrics

    def _test_ccs(
        self,
        epoch: int,
        epoch_loss: float,
        test_df: pd.DataFrame,
        metric_accumulator: MetricManager,
        data_split: str,
    ) -> bool:
        """
        Test the CCS model using the PSM dataframe and accumulate both the training loss and test metrics.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        epoch_loss : float
            The train loss value of the current epoch.
        test_df : pd.DataFrame
            The PSM dataframe.
        metric_accumulator : MetricManager
            The metric manager object.
        data_split : str
            The dataset label to test on. e.g. "validation", "train"
        Returns
        -------
        bool
            Whether to continue training or not based on the early stopping criteria.
        """
        continue_training = True
        if epoch % self._test_interval != 0 and epoch != -1:
            return continue_training

        self.ccs_model.model.eval()

        pred = self.ccs_model.predict(test_df)

        test_input = {
            "predicted": pred["ccs_pred"].values,
            "target": test_df["ccs"].values,
        }

        current_lr = (
            self.ccs_model.optimizer.param_groups[0]["lr"] if epoch != -1 else 0
        )
        continue_training = self._evaluate_metrics(
            test_input,
            metric_accumulator,
            epoch,
            data_split,
            "ccs",
            epoch_loss,
            current_lr,
        )

        self.ccs_model.model.train()

        return continue_training

    def finetune_ccs(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fine tune the CCS model using the PSM dataframe.

        Parameters
        ----------
        psm_df : pd.DataFrame
            The PSM dataframe.

        Returns
        -------
        pd.DataFrame
            Accumulated metrics during the fine tuning process.
        """
        if "mobility" not in psm_df.columns and "ccs" not in psm_df.columns:
            logger.error(
                "Failed to finetune CCS model. PSM dataframe does not contain mobility or ccs columns."
            )
            return None
        if "ccs" not in psm_df.columns:
            psm_df["ccs"] = mobility_to_ccs_for_df(psm_df, "mobility")
        elif "mobility" not in psm_df.columns:
            psm_df["mobility"] = ccs_to_mobility_for_df(psm_df, "ccs")

        # Shuffle the psm_df and split it into train and test
        train_df = psm_df.sample(frac=self._train_fraction)
        val_df = psm_df.drop(train_df.index).sample(
            frac=self._validation_fraction / (1 - self._train_fraction)
        )
        test_df = psm_df.drop(train_df.index).drop(val_df.index)

        test_metric_manager = MetricManager(
            test_metrics=[
                L1LossTestMetric(),
                LinearRegressionTestMetric(),
                AbsErrorPercentileTestMetric(95),
            ],
        )
        callback_handler = CustomCallbackHandler(
            self._test_ccs,
            test_df=val_df,
            metric_accumulator=test_metric_manager,
            data_split="validation",
        )
        self.ccs_model.set_callback_handler(callback_handler)

        self.ccs_model.set_lr_scheduler_class(CustomScheduler)

        self.early_stopping.reset()

        # Test the model before training
        self._test_ccs(-1, 0, psm_df, test_metric_manager, data_split="all")
        # Train the model
        logger.progress(" Fine-tuning CCS model with the following settings:")
        logger.info(
            f" Train fraction:      {self._train_fraction:3.2f}     Train size:      {len(train_df):<10}"
        )
        logger.info(
            f" Validation fraction: {self._validation_fraction:3.2f}     Validation size: {len(val_df):<10}"
        )
        logger.info(
            f" Test fraction:       {self._test_fraction:3.2f}     Test size:       {len(test_df):<10}"
        )
        self.ccs_model.model.train()
        self.ccs_model.train(
            train_df,
            batch_size=self._batch_size,
            epoch=self._epochs,
            warmup_epoch=self._warmup_epochs,
            lr=self._max_lr,
        )

        self._test_ccs(self._epochs, 0, test_df, test_metric_manager, data_split="test")

        metrics = test_metric_manager.get_stats()

        return metrics

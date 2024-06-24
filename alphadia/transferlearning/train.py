import logging

import numpy as np
import pandas as pd
import torch
from alphabase.peptide.fragment import remove_unused_fragments
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

settings = {
    # --------- USer settings ------------
    "batch_size": 1000,
    "max_lr": 0.0005,
    "train_fraction": 0.7,
    "validation_fraction": 0.2,
    "test_fraction": 0.1,
    "test_interval": 1,
    "lr_patience": 3,
    # --------- Our settings ------------
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

    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs):
        self.optimizer = optimizer
        self.num_warmup_steps = kwargs.get("num_warmup_steps", 5)
        self.num_training_steps = kwargs.get("num_training_steps", 50)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=settings["lr_patience"],
            factor=0.5,
        )
        self.warmup_lr = LambdaLR(optimizer, self._warmup)

    def _warmup(self, epoch: int):
        """
        Warmup the learning rate.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        return float(epoch + 1) / float(max(1, self.num_warmup_steps))

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
            self.warmup_lr.step()
        else:
            self.reduce_lr_on_plateau.step(loss)

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

    def __init__(self, patience: int = 5, margin: float = 0.01):
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
        if self.last_loss != np.inf:
            if (
                val_loss > self.best_loss * (1 + self.margin)
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
        self,
        mask_modloss: bool = False,
        device: str = "gpu",
        settings: dict | None = None,
    ):
        if settings is None:
            settings = {}
        super().__init__(mask_modloss, device)
        self.device = device
        self.settings = settings
        self.early_stopping = EarlyStopping(
            patience=(settings["lr_patience"] // settings["test_interval"]) * 4
        )

        assert (
            settings["train_fraction"]
            + settings["validation_fraction"]
            + settings["test_fraction"]
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
        if epoch % self.settings["test_interval"] == 0:
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
            val_metrics = metric_accumulator.calculate_test_metric(
                test_input, epoch, data_split=data_split, property_name="ms2"
            )
            if epoch != -1 and data_split == "validation":  # A training epoch
                metric_accumulator.accumulate_metrics(
                    epoch,
                    metric=epoch_loss,
                    metric_name="l1_loss",
                    data_split="train",
                    property_name="ms2",
                )
                current_lr = self.ms2_model.optimizer.param_groups[0]["lr"]
                metric_accumulator.accumulate_metrics(
                    epoch,
                    metric=current_lr,
                    metric_name="lr",
                    data_split="train",
                    property_name="ms2",
                )
                val_loss = val_metrics[val_metrics["metric_name"] == "l1_loss"][
                    "value"
                ].values[0]
                continue_training = self.early_stopping.step(val_loss)
                logger.progress(
                    f" Epoch {epoch:<3} Lr: {current_lr:.5f}   Training loss: {epoch_loss:.4f}   validation loss: {val_loss:.4f}"
                )
            else:
                logger.progress(
                    f" Ms2 model tested on {data_split} dataset with the following metrics:"
                )
                for i in range(len(val_metrics)):
                    logger.progress(
                        f" {val_metrics['metric_name'].values[i]:<30}: {val_metrics['value'].values[i]:.4f}"
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
        train_psm_df = psm_df.sample(frac=self.settings["train_fraction"]).copy()
        val_psm_df = (
            psm_df.drop(train_psm_df.index)
            .sample(
                frac=self.settings["validation_fraction"]
                / (1 - self.settings["train_fraction"])
            )
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

        reordered_val_psm_df = self._reset_frag_idx(val_psm_df)
        reordered_val_intensity_df = self._order_intensities(
            reordered_precursor_df=reordered_val_psm_df,
            unordered_precursor_df=val_psm_df,
            unordered_frag_df=val_intensity_df,
        )
        reordered_test_psm_df = self._reset_frag_idx(test_psm_df)
        reordered_test_intensity_df = self._order_intensities(
            reordered_precursor_df=reordered_test_psm_df,
            unordered_precursor_df=test_psm_df,
            unordered_frag_df=test_intensity_df,
        )

        # Create a metric manager
        test_metric_manager = MetricManager(
            test_metrics=[L1LossTestMetric(), Ms2SimilarityTestMetric()],
        )

        # create a callback handler
        callback_handler = CustomCallbackHandler(
            self._test_ms2,
            precursor_df=reordered_val_psm_df,
            target_fragment_intensity_df=reordered_val_intensity_df,
            metric_accumulator=test_metric_manager,
            data_split="validation",
        )

        # set the callback handler
        self.ms2_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.ms2_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
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
            f" Train fraction:      {self.settings['train_fraction']:3.2f}     Train size:      {len(train_psm_df):<10}"
        )
        logger.info(
            f" Validation fraction: {self.settings['validation_fraction']:3.2f}     Validation size: {len(val_psm_df):<10}"
        )
        logger.info(
            f" Test fraction:       {self.settings['test_fraction']:3.2f}     Test size:       {len(test_psm_df):<10}"
        )
        self.ms2_model.model.train()
        self.ms2_model.train(
            precursor_df=train_psm_df,
            fragment_intensity_df=train_intensity_df,
            epoch=self.settings["epochs"],
            batch_size=self.settings["batch_size"],
            warmup_epoch=self.settings["warmup_epochs"],
            lr=settings["max_lr"],
        )

        self._test_ms2(
            self.settings["epochs"],
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
        if epoch % self.settings["test_interval"] == 0:
            self.rt_model.model.eval()

            pred = self.rt_model.predict(test_df)
            test_input = {
                "predicted": pred["rt_pred"].values,
                "target": test_df["rt_norm"].values,
            }
            val_metrics = metric_accumulator.calculate_test_metric(
                test_input, epoch, data_split=data_split, property_name="rt"
            )
            if epoch != -1 and data_split == "validation":  # A training epoch
                metric_accumulator.accumulate_metrics(
                    epoch,
                    metric=epoch_loss,
                    metric_name="l1_loss",
                    data_split="train",
                    property_name="rt",
                )
                current_lr = self.rt_model.optimizer.param_groups[0]["lr"]
                metric_accumulator.accumulate_metrics(
                    epoch,
                    metric=current_lr,
                    metric_name="lr",
                    data_split="train",
                    property_name="rt",
                )
                val_loss = val_metrics[val_metrics["metric_name"] == "l1_loss"][
                    "value"
                ].values[0]
                continue_training = self.early_stopping.step(val_loss)
                logger.progress(
                    f" Epoch {epoch:<3} Lr: {current_lr:.5f}   Training loss: {epoch_loss:.4f}   validation loss: {val_loss:.4f}"
                )
            else:
                logger.progress(
                    f" RT model tested on {data_split} dataset with the following metrics:"
                )
                for i in range(len(val_metrics)):
                    logger.progress(
                        f" {val_metrics['metric_name'].values[i]:<30}: {val_metrics['value'].values[i]:.4f}"
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
        train_df = psm_df.sample(frac=self.settings["train_fraction"])
        val_df = psm_df.drop(train_df.index).sample(
            frac=self.settings["validation_fraction"]
            / (1 - self.settings["train_fraction"])
        )
        test_df = psm_df.drop(train_df.index).drop(val_df.index)

        # Create a test metric manager
        test_metric_manager = MetricManager(
            test_metrics=[
                L1LossTestMetric(),
                LinearRegressionTestMetric(),
                AbsErrorPercentileTestMetric(95),
            ],
        )

        # Create a callback handler
        callback_handler = CustomCallbackHandler(
            self._test_rt,
            test_df=val_df,
            metric_accumulator=test_metric_manager,
            data_split="validation",
        )
        # Set the callback handler
        self.rt_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.rt_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
        self.early_stopping.reset()

        # Test the model before training
        self._test_rt(-1, 0, psm_df, test_metric_manager, data_split="all")
        # Train the model
        logger.progress(" Fine-tuning RT model with the following settings:")
        logger.info(
            f" Train fraction:      {self.settings['train_fraction']:3.2f}     Train size:      {len(train_df):<10}"
        )
        logger.info(
            f" Validation fraction: {self.settings['validation_fraction']:3.2f}     Validation size: {len(val_df):<10}"
        )
        logger.info(
            f" Test fraction:       {self.settings['test_fraction']:3.2f}     Test size:       {len(test_df):<10}"
        )
        self.rt_model.model.train()
        self.rt_model.train(
            train_df,
            batch_size=self.settings["batch_size"],
            epoch=self.settings["epochs"],
            warmup_epoch=self.settings["warmup_epochs"],
            lr=settings["max_lr"],
        )

        self._test_rt(
            self.settings["epochs"], 0, test_df, test_metric_manager, data_split="test"
        )

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
        if epoch % self.settings["test_interval"] == 0:
            self.charge_model.model.eval()

            pred = self.charge_model.predict(test_df)
            test_input = {
                "target": np.array(test_df["charge_indicators"].values.tolist()),
                "predicted": np.array(pred["charge_probs"].values.tolist()),
            }
            val_metrics = metric_accumulator.calculate_test_metric(
                test_input, epoch, data_split=data_split, property_name="charge"
            )
            if epoch != -1 and data_split == "validation":  # A training epoch
                metric_accumulator.accumulate_metrics(
                    epoch,
                    metric=epoch_loss,
                    metric_name="ce_loss",
                    data_split="train",
                    property_name="charge",
                )
                current_lr = self.charge_model.optimizer.param_groups[0]["lr"]
                metric_accumulator.accumulate_metrics(
                    epoch,
                    metric=current_lr,
                    metric_name="lr",
                    data_split="train",
                    property_name="charge",
                )
                val_loss = val_metrics[val_metrics["metric_name"] == "ce_loss"][
                    "value"
                ].values[0]
                continue_training = self.early_stopping.step(val_loss)
                logger.progress(
                    f" Epoch {epoch:<3} Lr: {current_lr:.5f}   Training loss: {epoch_loss:.4f}   validation loss: {val_loss:.4f}"
                )
            else:
                logger.progress(
                    f" Charge model tested on {data_split} dataset with the following metrics: "
                )
                for i in range(len(val_metrics)):
                    logger.progress(
                        f" {val_metrics['metric_name'].values[i]:<30}: {val_metrics['value'].values[i]:.4f}"
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
        train_df = psm_df.sample(frac=self.settings["train_fraction"])
        val_df = psm_df.drop(train_df.index).sample(
            frac=self.settings["validation_fraction"]
            / (1 - self.settings["train_fraction"])
        )
        test_df = psm_df.drop(train_df.index).drop(val_df.index)

        # Create a test metric manager
        test_metric_manager = MetricManager(
            test_metrics=[
                CELossTestMetric(),
                AccuracyTestMetric(),
                PrecisionRecallTestMetric(),
            ],
        )

        # Create a callback handler
        callback_handler = CustomCallbackHandler(
            self._test_charge,
            test_df=val_df,
            metric_accumulator=test_metric_manager,
            data_split="validation",
        )

        # Set the callback handler
        self.charge_model.set_callback_handler(callback_handler)

        # Change the learning rate scheduler
        self.charge_model.set_lr_scheduler_class(CustomScheduler)

        # Reset the early stopping
        self.early_stopping.reset()

        # Test the model before training
        self._test_charge(-1, 0, psm_df, test_metric_manager, data_split="all")

        # Train the model
        logger.progress(" Fine-tuning Charge model with following settings:")
        logger.info(
            f" Train fraction:      {self.settings['train_fraction']:3.2f}     Train size:      {len(train_df):<10}"
        )
        logger.info(
            f" Validation fraction: {self.settings['validation_fraction']:3.2f}     Validation size: {len(val_df):<10}"
        )
        logger.info(
            f" Test fraction:       {self.settings['test_fraction']:3.2f}     Test size:       {len(test_df):<10}"
        )
        self.charge_model.model.train()
        self.charge_model.train(
            train_df,
            batch_size=self.settings["batch_size"],
            epoch=self.settings["epochs"],
            warmup_epoch=self.settings["warmup_epochs"],
            lr=settings["max_lr"],
        )

        self._test_charge(
            self.settings["epochs"], 0, test_df, test_metric_manager, data_split="test"
        )
        metrics = test_metric_manager.get_stats()

        return metrics

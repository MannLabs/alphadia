import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from alphadia.fdr_.utils import manage_torch_threads, train_test_split_

logger = logging.getLogger()


class Classifier(ABC):
    """Abstract base class for classifiers.

    Attributes
    ----------

    fitted : bool
        Whether the classifier has been fitted.

    """

    @property
    @abstractmethod
    def fitted(self):
        """Return whether the classifier has been fitted."""

    @abstractmethod
    def fit(self, x: np.array, y: np.array):
        """Fit the classifier to the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.array, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """

    @abstractmethod
    def predict(self, x: np.array):
        """Predict the class of the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------

        y : np.array, dtype=int
            Predicted class of shape (n_samples,).

        """

    @abstractmethod
    def predict_proba(self, x: np.array):
        """Predict the class probabilities of the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------

        y : np.array, dtype=float
            Predicted class probabilities of shape (n_samples, n_classes).

        """

    @abstractmethod
    def to_state_dict(self):
        """
        Return a state dict of the classifier.

        Returns
        -------

        state_dict : dict
            State dict of the classifier.
        """

    @abstractmethod
    def from_state_dict(self, state_dict: dict):
        """
        Load a state dict of the classifier.

        Parameters
        ----------

        state_dict : dict
            State dict of the classifier.

        """


class BinaryClassifierLegacy(Classifier):  # TODO: remove
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 2,
        test_size: float = 0.2,
        batch_size: int = 1000,
        epochs: int = 10,
        learning_rate: float = 0.0002,
        weight_decay: float = 0.00001,
        layers: list[int] | None = None,
        dropout: float = 0.001,
        metric_interval: int = 1000,
        **kwargs,
    ):
        """Binary Classifier using a feed forward neural network.

        Parameters
        ----------

        input_dim : int, default=10
            Number of input features.

        output_dim : int, default=2
            Number of output classes.

        test_size : float, default=0.2
            Fraction of the data to be used for testing.

        batch_size : int, default=1000
            Batch size for training.

        epochs : int, default=10
            Number of epochs for training.

        learning_rate : float, default=0.0002
            Learning rate for training.

        weight_decay : float, default=0.00001
            Weight decay for training.

        layers : typing.List[int], default=[100, 50, 20, 5]
            typing.List of hidden layer sizes.

        dropout : float, default=0.001
            Dropout probability for training.

        metric_interval : int, default=1000
            Interval for logging metrics during training.

        """
        if layers is None:
            layers = [100, 50, 20, 5]
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.layers = layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.metric_interval = metric_interval

        self.network = None
        self.optimizer = None
        self._fitted = False

        self.metrics = {
            "epoch": [],
            "batch_count": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        if kwargs:
            warnings.warn(f"Unknown arguments: {kwargs}")

    @property
    def fitted(self):
        return self._fitted

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    def to_state_dict(self):
        """Save the state of the classifier as a dictionary.

        Returns
        -------

        dict : dict
            Dictionary containing the state of the classifier.

        """
        dict = {
            "_fitted": self._fitted,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "test_size": self.test_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "layers": self.layers,
            "dropout": self.dropout,
            "metric_interval": self.metric_interval,
            "metrics": self.metrics,
        }

        if self._fitted:
            dict["network_state_dict"] = self.network.state_dict()

        return dict

    def from_state_dict(self, state_dict: dict, load_hyperparameters: bool = False):
        """Load the state of the classifier from a dictionary.

        Parameters
        ----------

        dict : dict
            Dictionary containing the state of the classifier.

        """

        _state_dict = deepcopy(state_dict)

        if "network_state_dict" in _state_dict:
            self.network = FeedForwardNN(
                input_dim=_state_dict.pop("input_dim"),
                output_dim=_state_dict.pop("output_dim"),
                layers=_state_dict.pop("layers"),
                dropout=_state_dict.pop("dropout"),
            )
            self.network.load_state_dict(state_dict.pop("network_state_dict"))
            self._fitted = True

        if load_hyperparameters:
            self.__dict__.update(_state_dict)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit the classifier to the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.array, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """

        force_reinit = False

        if self.input_dim != x.shape[1] and self.network is not None:
            warnings.warn(
                "Input dimension of network has changed. Network has been reinitialized."
            )
            force_reinit = True

        # check if network has to be initialized
        if self.network is None or force_reinit:
            self.input_dim = x.shape[1]
            self.network = FeedForwardNN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                layers=self.layers,
                dropout=self.dropout,
            )

        # normalize input
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

        if y.ndim == 1:
            y = np.stack([1 - y, y], axis=1)

        x_train, x_test, y_train, y_test = train_test_split_(
            x, y, test_size=self.test_size
        )
        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss = nn.BCELoss()

        batch_count = 0

        for j in range(self.epochs):
            order = np.random.permutation(len(x_train))
            x_train = torch.Tensor(x_train[order])
            y_train = torch.Tensor(y_train[order])

            for batch_x, batch_y in zip(
                x_train.split(self.batch_size),
                y_train.split(self.batch_size),
                strict=True,
            ):
                y_pred = self.network(batch_x)
                loss_value = loss(y_pred, batch_y)

                self.network.zero_grad()
                loss_value.backward()
                optimizer.step()

                if batch_count % self.metric_interval == 0:
                    self.network.eval()
                    with torch.no_grad():
                        self.metrics["epoch"].append(j)
                        self.metrics["batch_count"].append(batch_count)
                        self.metrics["train_loss"].append(loss_value.item())

                        y_pred_test = self.network(x_test)
                        loss_value = loss(y_pred_test, y_test)
                        self.metrics["test_loss"].append(loss_value.item())

                        y_pred_train = self.network(x_train).detach().numpy()
                        y_pred_test = self.network(x_test).detach().numpy()

                        self.metrics["train_accuracy"].append(
                            np.sum(
                                y_train[:, 1].detach().numpy()
                                == np.argmax(y_pred_train, axis=1)
                            )
                            / len(y_train)
                        )

                        self.metrics["test_accuracy"].append(
                            np.sum(
                                y_test[:, 1].detach().numpy()
                                == np.argmax(y_pred_test, axis=1)
                            )
                            / len(y_test)
                        )
                    self.network.train()

                batch_count += 1

        self._fitted = True

    def predict(self, x):
        """Predict the class of the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------

        y : np.array, dtype=int
            Predicted class of shape (n_samples,).
        """

        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        assert (
            x.ndim == 2
        ), "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert (
            x.shape[1] == self.input_dim
        ), "Input data must have the same number of features as the fitted classifier."

        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        self.network.eval()
        return np.argmax(self.network(torch.Tensor(x)).detach().numpy(), axis=1)

    def predict_proba(self, x: np.ndarray):
        """Predict the class probabilities of the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------

        y : np.array, dtype=float
            Predicted class probabilities of shape (n_samples, n_classes).

        """

        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        assert (
            x.ndim == 2
        ), "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert (
            x.shape[1] == self.input_dim
        ), "Input data must have the same number of features as the fitted classifier."

        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        self.network.eval()
        return self.network(torch.Tensor(x)).detach().numpy()


def get_scaled_training_params(df, base_lr=0.001, max_batch=4096, min_batch=128):
    """
    Scale batch size and learning rate based on dataframe size using square root relationship.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    base_lr : float, optional
        Base learning rate for 1024 batch size, defaults to 0.01
    max_batch : int, optional
        Maximum batch size (1024 for >= 1M samples), defaults to 1024
    min_batch : int, optional
        Minimum batch size, defaults to 128

    Returns
    -------
    tuple(int, float)
        (batch_size, learning_rate)
    """
    n_samples = len(df)

    # For >= 1M samples, use max batch size
    if n_samples >= 1_000_000:
        return max_batch, base_lr

    # Calculate scaled batch size (linear scaling between min and max)
    batch_size = int(np.clip((n_samples / 1_000_000) * max_batch, min_batch, max_batch))

    # Scale learning rate using square root relationship
    # sqrt(batch_size) / sqrt(max_batch) = scaled_lr / base_lr
    learning_rate = base_lr * np.sqrt(batch_size / max_batch)

    return batch_size, learning_rate


class BinaryClassifierLegacyNewBatching(Classifier):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 2,
        test_size: float = 0.2,
        batch_size: int = 1000,
        epochs: int = 10,
        learning_rate: float = 0.0002,
        weight_decay: float = 0.00001,
        layers: list[int] | None = None,
        dropout: float = 0.001,
        metric_interval: int = 1000,
        experimental_hyperparameter_tuning: bool = False,
        **kwargs,
    ):
        """Binary Classifier using a feed forward neural network.

        Parameters
        ----------

        input_dim : int, default=10
            Number of input features.

        output_dim : int, default=2
            Number of output classes.

        test_size : float, default=0.2
            Fraction of the data to be used for testing.

        batch_size : int, default=1000
            Batch size for training.

        epochs : int, default=10
            Number of epochs for training.

        learning_rate : float, default=0.0002
            Learning rate for training.

        weight_decay : float, default=0.00001
            Weight decay for training.

        layers : typing.List[int], default=[100, 50, 20, 5]
            typing.List of hidden layer sizes.

        dropout : float, default=0.001
            Dropout probability for training.

        metric_interval : int, default=1000
            Interval for logging metrics during training.

        experimental_hyperparameter_tuning: bool, default=False
            Whether to use experimental hyperparameter tuning.

        """
        if layers is None:
            layers = [100, 50, 20, 5]
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.layers = layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.metric_interval = metric_interval
        self.experimental_hyperparameter_tuning = experimental_hyperparameter_tuning

        self.network = None
        self.optimizer = None
        self._fitted = False

        self.metrics = {
            "epoch": [],
            "batch_count": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        if kwargs:
            warnings.warn(f"Unknown arguments: {kwargs}")

    @property
    def fitted(self):
        return self._fitted

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    def to_state_dict(self):
        """Save the state of the classifier as a dictionary.

        Returns
        -------

        dict : dict
            Dictionary containing the state of the classifier.

        """
        dict = {
            "_fitted": self._fitted,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "test_size": self.test_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "layers": self.layers,
            "dropout": self.dropout,
            "metric_interval": self.metric_interval,
            "metrics": self.metrics,
        }

        if self._fitted:
            dict["network_state_dict"] = self.network.state_dict()

        return dict

    def from_state_dict(self, state_dict: dict, load_hyperparameters: bool = False):
        """Load the state of the classifier from a dictionary.

        Parameters
        ----------

        dict : dict
            Dictionary containing the state of the classifier.

        """

        _state_dict = deepcopy(state_dict)

        if "network_state_dict" in _state_dict:
            self.network = FeedForwardNN(
                input_dim=_state_dict.pop("input_dim"),
                output_dim=_state_dict.pop("output_dim"),
                layers=_state_dict.pop("layers"),
                dropout=_state_dict.pop("dropout"),
            )
            self.network.load_state_dict(state_dict.pop("network_state_dict"))
            self._fitted = True

        if load_hyperparameters:
            self.__dict__.update(_state_dict)

    @manage_torch_threads(max_threads=2)
    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit the classifier to the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.array, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """
        if self.experimental_hyperparameter_tuning:
            self.batch_size, self.learning_rate = get_scaled_training_params(x)
            logger.info(
                f"Estimating optimal hyperparameters - "
                f"samples: {len(x):,}, "
                f"batch_size: {self.batch_size:,}, "
                f"learning_rate: {self.learning_rate:.2e}"
            )

        force_reinit = False

        if self.input_dim != x.shape[1] and self.network is not None:
            warnings.warn(
                "Input dimension of network has changed. Network has been reinitialized."
            )
            force_reinit = True

        # check if network has to be initialized
        if self.network is None or force_reinit:
            self.input_dim = x.shape[1]
            self.network = FeedForwardNN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                layers=self.layers,
                dropout=self.dropout,
            )

        # normalize input
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

        if y.ndim == 1:
            y = np.stack([1 - y, y], axis=1)

        x_train, x_test, y_train, y_test = train_test_split_(
            x, y, test_size=self.test_size
        )
        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss = nn.BCELoss()

        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)

        num_batches = (x_train.shape[0] // self.batch_size) - 1
        batch_start_list = np.arange(num_batches) * self.batch_size
        batch_stop_list = np.arange(num_batches) * self.batch_size + self.batch_size

        batch_count = 0

        for epoch in tqdm(range(self.epochs)):
            # shuffle batches
            order = np.random.permutation(num_batches)
            batch_start_list = batch_start_list[order]
            batch_stop_list = batch_stop_list[order]

            for batch_start, batch_stop in zip(
                batch_start_list, batch_stop_list, strict=True
            ):
                x_train_batch = x_train[batch_start:batch_stop]
                y_train_batch = y_train[batch_start:batch_stop]
                y_pred = self.network(x_train_batch)
                loss_value = loss(y_pred, y_train_batch)

                self.network.zero_grad()
                loss_value.backward()
                optimizer.step()

                if batch_count % self.metric_interval == 0:
                    self.network.eval()
                    with torch.no_grad():
                        self.metrics["epoch"].append(epoch)
                        self.metrics["batch_count"].append(batch_count)
                        self.metrics["train_loss"].append(loss_value.item())

                        y_pred_test = self.network(x_test)
                        loss_value = loss(y_pred_test, y_test)
                        self.metrics["test_loss"].append(loss_value.item())

                        y_pred_train = self.network(x_train_batch).detach().numpy()
                        y_pred_test = self.network(x_test).detach().numpy()

                        self.metrics["train_accuracy"].append(
                            np.sum(
                                y_train_batch[:, 1].detach().numpy()
                                == np.argmax(y_pred_train, axis=1)
                            )
                            / len(y_train_batch)
                        )

                        self.metrics["test_accuracy"].append(
                            np.sum(
                                y_test[:, 1].detach().numpy()
                                == np.argmax(y_pred_test, axis=1)
                            )
                            / len(y_test)
                        )
                    self.network.train()

                batch_count += 1

        self._fitted = True

    @manage_torch_threads(max_threads=2)
    def predict(self, x):
        """Predict the class of the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------

        y : np.array, dtype=int
            Predicted class of shape (n_samples,).
        """

        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        assert (
            x.ndim == 2
        ), "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert (
            x.shape[1] == self.input_dim
        ), "Input data must have the same number of features as the fitted classifier."

        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        self.network.eval()
        return np.argmax(self.network(torch.Tensor(x)).detach().numpy(), axis=1)

    @manage_torch_threads(max_threads=2)
    def predict_proba(self, x: np.ndarray):
        """Predict the class probabilities of the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------

        y : np.array, dtype=float
            Predicted class probabilities of shape (n_samples, n_classes).

        """

        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        assert (
            x.ndim == 2
        ), "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert (
            x.shape[1] == self.input_dim
        ), "Input data must have the same number of features as the fitted classifier."

        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        self.network.eval()
        return self.network(torch.Tensor(x)).detach().numpy()


class FeedForwardNN(nn.Module):  # TODO: remove
    def __init__(
        self,
        input_dim,
        output_dim=2,
        layers: list[int] | None = None,
        dropout=0.5,
    ):
        """
        built a simple feed forward network for FDR estimation

        """
        if layers is None:
            layers = [20, 10, 5]
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = [input_dim] + layers
        self.dropout = dropout

        self._build_model()

    def _build_model(self):
        layers = []
        # add batch norm layer
        layers.append(nn.BatchNorm1d(self.input_dim))
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.layers[-1], self.output_dim))
        # add softmax layer
        layers.append(nn.Softmax(dim=1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)


class SupervisedLoss:  # TODO: remove
    def __init__(self) -> None:
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        # output_selection = get_group_mask(y_pred.detach().detach().cpu().numpy(), y_groups.detach().cpu().numpy(), n_groups)

        # y_pred = y_pred[output_selection]
        # y_true = y_true[output_selection]

        return self.loss(y_pred, y_true)

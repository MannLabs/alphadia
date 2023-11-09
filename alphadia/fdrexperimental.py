# native imports
import typing
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

# alphadia imports

# alpha family imports

# third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
from tqdm import tqdm


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
        pass

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
        pass

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
        pass

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
        pass

    @abstractmethod
    def to_state_dict(self):
        """
        Return a state dict of the classifier.

        Returns
        -------

        state_dict : dict
            State dict of the classifier.
        """
        pass

    @abstractmethod
    def from_state_dict(self, state_dict: dict):
        """
        Load a state dict of the classifier.

        Parameters
        ----------

        state_dict : dict
            State dict of the classifier.

        """
        pass


class BinaryClassifier(Classifier):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 2,
        test_size: float = 0.2,
        batch_size: int = 1000,
        epochs: int = 10,
        learning_rate: float = 0.0002,
        weight_decay: float = 0.00001,
        layers: typing.List[int] = [100, 50, 20, 5],
        dropout: float = 0.001,
        metric_interval: int = 1000,
        num_workers: int = 0,
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

        num_workers : int, default=0
            Number of parallel workers for data loading. Should not be larger than the number of avaliable CPU cores.

        """

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
        self.num_workers = num_workers

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

    def from_state_dict(self, state_dict: dict):
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

        best_test_accuracy = 0.0
        patience = 5
        counter_patience = 0

        # normalize input
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

        if y.ndim == 1:
            y = np.stack([1 - y, y], axis=1)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=self.test_size
        )

        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)

        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss = nn.BCELoss()

        batch_count = 0

        for j in tqdm(range(self.epochs)):
            train_loss_sum = 0.0
            train_accuracy_sum = 0.0
            test_loss_sum = 0.0
            test_accuracy_sum = 0.0

            num_batches_train = 0
            num_batches_test = 0

            for batch_x, batch_y in train_loader:
                y_pred = self.network(batch_x)
                loss_value = loss(y_pred, batch_y)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                train_loss_sum += loss_value.detach()
                train_accuracy_sum += (
                    (batch_y[:, 1] == y_pred.argmax(dim=1)).float().mean()
                )
                num_batches_train += 1

            self.network.eval()
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    y_pred_test = self.network(batch_x)
                    loss_value = loss(y_pred_test, batch_y)
                    test_loss_sum += loss_value
                    test_accuracy_sum += (
                        (batch_y[:, 1] == y_pred_test.argmax(dim=1)).float().mean()
                    )
                    num_batches_test += 1

            self.network.train()

            # log metrics
            average_train_loss = train_loss_sum / num_batches_train
            average_train_accuracy = train_accuracy_sum / num_batches_train

            average_test_loss = test_loss_sum / num_batches_test
            average_test_accuracy = test_accuracy_sum / num_batches_test

            self.metrics["train_loss"].append(average_train_loss.item())
            self.metrics["train_accuracy"].append(average_train_accuracy.item())

            self.metrics["train_loss"].append(average_train_loss.item())
            self.metrics["train_accuracy"].append(average_train_accuracy.item())
            self.metrics["test_loss"].append(average_test_loss.item())
            self.metrics["test_accuracy"].append(average_test_accuracy.item())
            self.metrics["epoch"].append(j)

            batch_count += num_batches_train
            self.metrics["batch_count"].append(batch_count)

            # check for early stopping
            if average_test_accuracy > best_test_accuracy:
                best_test_accuracy = average_test_accuracy
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience >= patience:
                break

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


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=2,
        layers=[20, 10, 5],
        dropout=0.5,
    ):
        """
        built a simple feed forward network for FDR estimation

        """
        super(FeedForwardNN, self).__init__()
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


class SupervisedLoss:
    def __init__(self) -> None:
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        # output_selection = get_group_mask(y_pred.detach().detach().cpu().numpy(), y_groups.detach().cpu().numpy(), n_groups)

        # y_pred = y_pred[output_selection]
        # y_true = y_true[output_selection]

        return self.loss(y_pred, y_true)

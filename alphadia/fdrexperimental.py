# native imports
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

# alphadia imports
# alpha family imports
# third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import model_selection
from torchmetrics.classification import BinaryAUROC
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
        max_batch_size: int = 10000,
        min_batch_number: int = 100,
        epochs: int = 10,
        learning_rate: float = 0.0002,
        weight_decay: float = 0.00001,
        layers: list[int] | None = None,
        dropout: float = 0.001,
        calculate_metrics: bool = True,
        metric_interval: int = 1,
        patience: int = 15,
        use_gpu: bool = True,
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

        max_batch_size : int, default=10000
            Maximum batch size for training.
            The actual batch will be scaled to make sure at least min_batch_number batches are used.

        min_batch_number : int, default=100
            Minimum number of batches for training.
            The actual batch number will be scaled if more than min_batchnumber * max_batch_size samples are available.

        epochs : int, default=10
            Number of epochs for training.

        learning_rate : float, default=0.0002
            Base learning rate for a batch size of max_batch_size.
            If smaller batches are used, the learning rate will be scaled linearly.

        weight_decay : float, default=0.00001
            Weight decay for training.

        layers : typing.List[int], default=[100, 50, 20, 5]
            typing.List of hidden layer sizes.

        dropout : float, default=0.001
            Dropout probability for training.

        calculate_metrics : bool, default=True
            Whether to calculate metrics during training.

        metric_interval : int, default=1
            Interval for logging metrics during training, once per metric_interval epochs.

        patience : int, default=15
            Number of epochs to wait for improvement before early stopping.

        use_gpu : bool, default=True
            Whether to use GPU acceleration if available.
        """

        if layers is None:
            layers = [100, 50, 20, 5]
        self.test_size = test_size
        self.max_batch_size = max_batch_size
        self.min_batch_number = min_batch_number
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.layers = layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.metric_interval = metric_interval
        self.calculate_metrics = calculate_metrics
        self.patience = patience
        self.use_gpu = use_gpu

        self.network = None
        self.optimizer = None
        self._fitted = False
        self.device = self.determine_device()

        self.metrics = {
            "epoch": [],
            "batch_count": [],
            "train_loss": [],
            "test_loss": [],
            "train_auc": [],
            "train_fdr01": [],
            "train_fdr1": [],
            "test_auc": [],
            "test_fdr01": [],
            "test_fdr1": [],
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
            self._fitted = True

        self.__dict__.update(_state_dict)

    def determine_device(self):
        if self.use_gpu:
            if torch.cuda.is_available():
                return torch.device("cuda")
            # elif torch.backends.mps.is_available():  # slows things down as of 13.12.2023
            #     return torch.device("mps")
            else:
                print(
                    "GPU requested, but no compatible GPU found. Falling back to CPU."
                )
        return torch.device("cpu")

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit the classifier to the data.

        Parameters
        ----------

        x : np.array, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.array, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """

        batch_number = max(self.min_batch_number, x.shape[0] // self.max_batch_size)
        batch_size = x.shape[0] // batch_number
        lr_scaled = self.learning_rate * batch_size / self.max_batch_size

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
            ).to(self.device)

        optimizer = optim.AdamW(
            self.network.parameters(),
            lr=lr_scaled,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, steps_per_epoch=batch_number, epochs=self.epochs
        )

        loss = nn.BCELoss()

        binary_auroc = BinaryAUROC()

        best_fdr1 = 0.0
        patience = self.patience

        x -= x.mean(axis=0)
        x /= x.std(axis=0) + 1e-6

        if y.ndim == 1:
            y = np.stack([1 - y, y], axis=1)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=self.test_size
        )
        x_train = torch.from_numpy(x_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        x_test = torch.from_numpy(x_test).float().to(self.device)
        y_test = torch.from_numpy(y_test).float().to(self.device)

        batch_count = 0
        for epoch in tqdm(range(self.epochs)):
            train_loss_sum = 0.0
            test_loss_sum = 0.0

            num_batches_train = 0
            num_batches_test = 0

            # shuffle training data

            permuted_indices = torch.randperm(x_train.shape[0])

            train_predictions_list = []
            train_labels_list = []

            for batch_start in range(0, x_train.shape[0], batch_size):
                batch_indices = permuted_indices[batch_start : batch_start + batch_size]

                x_train_batch = x_train[batch_indices]
                y_train_batch = y_train[batch_indices]

                y_pred = self.network(x_train_batch)
                loss_value = loss(y_pred, y_train_batch)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                scheduler.step()

                train_loss_sum += loss_value.detach()
                train_predictions_list.append(y_pred.detach())
                train_labels_list.append(y_train_batch.detach()[:, 1])
                num_batches_train += 1

            train_predictions = torch.cat(train_predictions_list, dim=0)
            train_labels = torch.cat(train_labels_list, dim=0)

            auc, fdr01, fdr1 = self.get_auc_fdr(
                train_predictions,
                train_labels,
                roc_object=binary_auroc,
            )

            if not self.calculate_metrics:
                # check for early stopping
                if fdr1 > best_fdr1:
                    best_fdr1 = fdr1
                    patience = self.patience
                else:
                    patience -= 1

                if patience <= 0:
                    break
                continue

            if epoch % self.metric_interval != 0:  # skip metrics if wrong epoch
                continue

            self.network.eval()
            with torch.no_grad():
                test_predictions_list = []
                test_labels_list = []

                test_batch_size = min(batch_size, x_test.shape[0])
                test_num_batches = x_test.shape[0] // test_batch_size
                test_batch_start_list = np.arange(test_num_batches) * test_batch_size
                test_batch_stop_list = (
                    np.arange(test_num_batches) * test_batch_size + test_batch_size
                )

                for batch_start, batch_stop in zip(
                    test_batch_start_list, test_batch_stop_list, strict=True
                ):
                    batch_x_test = x_test[batch_start:batch_stop]
                    batch_y_test = y_test[batch_start:batch_stop]

                    y_pred_test = self.network(batch_x_test)
                    test_loss = loss(y_pred_test, batch_y_test)
                    test_predictions_list.append(y_pred_test.detach())
                    test_labels_list.append(batch_y_test.detach()[:, 1])
                    num_batches_test += 1
                    test_loss_sum += test_loss

                # log metrics for train and test
                average_train_loss = train_loss_sum / num_batches_train
                average_test_loss = test_loss_sum / num_batches_test

                self.metrics["train_loss"].append(average_train_loss.item())
                self.metrics["test_loss"].append(average_test_loss.item())

                self.metrics["train_auc"].append(auc.item())
                self.metrics["train_fdr01"].append(fdr01.item())
                self.metrics["train_fdr1"].append(fdr1.item())

                test_predictions = torch.cat(test_predictions_list, dim=0)
                test_labels = torch.cat(test_labels_list, dim=0)

                auc, fdr01, fdr1 = self.get_auc_fdr(
                    test_predictions, test_labels, roc_object=binary_auroc
                )
                self.metrics["test_auc"].append(auc.item())
                self.metrics["test_fdr01"].append(fdr01.item())
                self.metrics["test_fdr1"].append(fdr1.item())

                self.metrics["epoch"].append(epoch)

                batch_count += num_batches_train
                self.metrics["batch_count"].append(batch_count)

            self.network.train()

            # check for early stopping
            if fdr1 > best_fdr1:
                best_fdr1 = fdr1
                patience = self.patience
            else:
                patience -= 1

            if patience <= 0:
                break

        self._fitted = True

    @torch.jit.export
    def get_auc_fdr(self, predicted_probas: torch.Tensor, y: torch.Tensor, roc_object):
        """Calculates the AUC and FDR for a given set of predicted probabilities and labels.

        Parameters
        ----------
        predicted_probas : torch.Tensor
            The predicted probabilities.

        y : torch.Tensor
            True labels. Decoys are expected to be 1 and targets 0.

        roc_object : torchmetrics.classification.BinaryAUROC
            The ROC object to use for calculating the AUC.

        Returns
        -------
        torch.Tensor
        """
        scores = predicted_probas[:, 1]
        sorted_indices = torch.argsort(scores, stable=True)
        decoys_sorted = y[sorted_indices]

        # getting q values
        decoy_cumsum = torch.cumsum(decoys_sorted, dim=0)
        target_cumsum = torch.cumsum(1 - decoys_sorted, dim=0)
        fdr_values = decoy_cumsum.float() / target_cumsum.float()

        reversed_fdr = torch.flip(fdr_values, dims=[0])
        cumulative_mins, _ = torch.cummin(reversed_fdr, dim=0)
        qval = torch.flip(cumulative_mins, dims=[0])

        decoys_zero_mask = decoys_sorted == 0
        qval = qval[decoys_zero_mask]

        # calculating auc & fdr
        y_pred = torch.round(scores)
        auc = roc_object(y_pred, y)
        fdr01 = torch.sum(qval < 0.001)
        fdr1 = torch.sum(qval < 0.01)
        return auc, fdr01, fdr1

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
        return np.argmax(self.network(torch.Tensor(x)).detach().cpu().numpy(), axis=1)

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
        inp = torch.Tensor(x).to(self.device)
        return self.network(inp).detach().cpu().numpy()


class BinaryClassifierLegacy(Classifier):
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

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
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

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
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


class SupervisedLoss:
    def __init__(self) -> None:
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        # output_selection = get_group_mask(y_pred.detach().detach().cpu().numpy(), y_groups.detach().cpu().numpy(), n_groups)

        # y_pred = y_pred[output_selection]
        # y_true = y_true[output_selection]

        return self.loss(y_pred, y_true)

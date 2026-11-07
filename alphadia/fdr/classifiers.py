"""Classifier module for FDR estimation."""

import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from alphadia_search_rs import (
    fdr_mlp_fit,
    fdr_mlp_predict,
    fdr_quantile_fit,
    fdr_quantile_transform,
)
from sklearn.preprocessing import QuantileTransformer
from torch import nn, optim
from tqdm import tqdm

from alphadia.constants.keys import FeatureTransform
from alphadia.fdr.utils import manage_torch_threads, train_test_split_indices

logger = logging.getLogger()

_N_QUANTILES = 1001

# Below this number of training rows the empirical quantiles are too coarse to be
# a better input than the raw features.
_MIN_ROWS_FOR_QUANTILE_TRANSFORM = 2000

# Rows predicted at once. A forward pass over all rows at once holds the activations of every
# layer for every row, gigabytes for the 10 M+ rows of a final search.
# TODO: find optimal value for this
_PREDICT_ROWS_CHUNK_SIZE = 250_000


class LRSchedule:
    """Learning rate schedules of `BinaryClassifierLegacyNewBatching`."""

    CONSTANT = "constant"
    # decays the learning rate along a half cosine from its start value to zero over all minibatches of the fit
    COSINE = "cosine"

    @classmethod
    def get_values(cls) -> list[str]:
        """All valid schedule names."""
        return [cls.CONSTANT, cls.COSINE]


class Classifier(ABC):
    """Abstract base class for classifiers.

    Attributes
    ----------
    fitted : bool
        Whether the classifier has been fitted.

    """

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Return whether the classifier has been fitted."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier to the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.ndarray, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """

    @abstractmethod
    def reset(self) -> None:
        """Set the classifier back to an unfitted state."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        y : np.ndarray, dtype=int
            Predicted class of shape (n_samples,).

        """

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict the class probabilities of the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        y : np.ndarray, dtype=float
            Predicted class probabilities of shape (n_samples, n_classes).

        """

    @abstractmethod
    def to_state_dict(self) -> dict:
        """Return a state dict of the classifier.

        Returns
        -------
        state_dict : dict
            State dict of the classifier.

        """

    @abstractmethod
    def from_state_dict(self, state_dict: dict) -> None:
        """Load a state dict of the classifier.

        Parameters
        ----------
        state_dict : dict
            State dict of the classifier.

        """


def _get_scaled_training_params(
    df: pd.DataFrame | np.ndarray,
    base_lr: float = 0.001,
    max_batch: int = 4096,
    min_batch: int = 128,
) -> tuple[int, float]:
    """Scale batch size and learning rate based on dataframe size using square root relationship.

    Parameters
    ----------
    df : pd.DataFrame | np.ndarray
        Input dataframe or array
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
    if n_samples >= 1_000_000:  # noqa: PLR2004
        return max_batch, base_lr

    # Calculate scaled batch size (linear scaling between min and max)
    batch_size = int(np.clip((n_samples / 1_000_000) * max_batch, min_batch, max_batch))

    # Scale learning rate using square root relationship
    # sqrt(batch_size) / sqrt(max_batch) = scaled_lr / base_lr
    learning_rate = base_lr * np.sqrt(batch_size / max_batch)

    return batch_size, learning_rate


class BinaryClassifierLegacyNewBatching(Classifier):
    """Binary Classifier using a feed forward neural network."""

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        input_dim: int = 10,
        test_size: float = 0.2,
        batch_size: int = 1000,
        epochs: int = 10,
        learning_rate: float = 0.0002,
        weight_decay: float = 0.00001,
        layers: list[int] | None = None,
        dropout: float = 0.001,
        metric_interval: int = 1000,
        feature_transform: str = FeatureTransform.NONE,
        *,
        experimental_hyperparameter_tuning: bool = False,
        random_state: int | None = None,
        **kwargs,  # TODO: needed?
    ):
        """Binary Classifier using a feed forward neural network.

        Parameters
        ----------
        input_dim : int, default=10
            Number of input features.

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

        layers : typing.List[int], default=[128, 64, 32]
            typing.List of hidden layer sizes.

        dropout : float, default=0.001
            Dropout probability for training.

        metric_interval : int, default=1000
            Interval for logging metrics during training.

        feature_transform : str, default="none"
            Transform applied to the features before they enter the network,
            either "none" or "quantile". See `_transform_features`.

        experimental_hyperparameter_tuning: bool, default=False
            Whether to use experimental hyperparameter tuning.

        random_state : int, optional
            Random seed for reproducibility of torch, numpy, and train-test-split.

        **kwargs : dict
            Keyword arguments. Will raise a warning if used.

        """
        if layers is None:
            # no narrow bottleneck: when all units of a 5-unit last layer are zero for the
            # target-like PSMs, they all get the output bias, i.e. one tied probability
            layers = [128, 64, 32]
        if feature_transform not in FeatureTransform.get_values():
            raise ValueError(
                f"Unknown feature transform: {feature_transform}. "
                f"Valid options are {FeatureTransform.get_values()}"
            )
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.layers = layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.metric_interval = metric_interval
        self.feature_transform = feature_transform
        self.experimental_hyperparameter_tuning = experimental_hyperparameter_tuning

        self.network = None
        self.optimizer = None
        self._fitted = False
        self._quantile_transformer: QuantileTransformer | None = None

        self.metrics = {
            "epoch": [],
            "batch_count": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        self._np_rng = np.random.default_rng(seed=random_state)
        if random_state is not None:
            random_state_torch = self._np_rng.integers(0, 1_000_000)
            torch.manual_seed(random_state_torch)
            logger.info(
                f"Classifier: using random state {random_state} for numpy, {random_state_torch} for pytorch"
            )

        if kwargs:
            warnings.warn(f"Unknown arguments: {kwargs}")

    @property
    def fitted(self) -> bool:
        """Return whether the classifier has been fitted."""
        return self._fitted

    def to_state_dict(self) -> dict:
        """Save the state of the classifier as a dictionary.

        Returns
        -------
        dict : dict
            Dictionary containing the state of the classifier.

        """
        state_dict = {
            "_fitted": self._fitted,
            "input_dim": self.input_dim,
            "test_size": self.test_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "layers": self.layers,
            "dropout": self.dropout,
            "metric_interval": self.metric_interval,
            "metrics": self.metrics,
            "feature_transform": self.feature_transform,
            "_quantile_transformer": self._quantile_transformer,
        }

        if self._fitted:
            state_dict["network_state_dict"] = self.network.state_dict()

        return state_dict

    def from_state_dict(
        self, state_dict: dict, *, load_hyperparameters: bool = False
    ) -> None:
        """Load the state of the classifier from a dictionary.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing the state of the classifier.

        load_hyperparameters : bool, default=False
            Whether to load hyperparameters from the state dict.

        """
        _state_dict = deepcopy(state_dict)

        self._quantile_transformer = _state_dict.pop("_quantile_transformer")

        if "network_state_dict" in _state_dict:
            self.network = FeedForwardNN(
                input_dim=_state_dict.pop("input_dim"),
                layers=_state_dict.pop("layers"),
                dropout=_state_dict.pop("dropout"),
            )
            self.network.load_state_dict(state_dict.pop("network_state_dict"))
            self._fitted = True

        if load_hyperparameters:
            self.__dict__.update(_state_dict)

    def reset(self) -> None:
        """Set the classifier back to an unfitted state.

        The next fit starts from new random weights, not from the first ones. A retry after
        a collapsed fit needs different start weights.
        """
        self.network = None
        self.optimizer = None
        self._fitted = False
        for values in self.metrics.values():
            values.clear()

    def _fit_feature_transform(self, x: np.ndarray) -> QuantileTransformer | None:
        """Fit the configured feature transform on the training data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Training data of shape (n_samples, n_features).

        Returns
        -------
        QuantileTransformer | None
            The fitted transformer, or None if the raw features are used.

        """
        if self.feature_transform == FeatureTransform.NONE:
            return None

        if len(x) < _MIN_ROWS_FOR_QUANTILE_TRANSFORM:
            logger.warning(
                f"Only {len(x):,} rows available for the quantile feature transform "
                f"(minimum {_MIN_ROWS_FOR_QUANTILE_TRANSFORM:,}), using the raw features instead."
            )
            return None

        return QuantileTransformer(
            n_quantiles=_N_QUANTILES, output_distribution="normal", subsample=None
        ).fit(x)

    def _transform_features(self, x: np.ndarray) -> np.ndarray:
        """Map every feature to a standard normal score through its empirical CDF.

        Extreme candidates would otherwise lie outside the training range, where the
        network extrapolates arbitrarily.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        x_transformed : np.ndarray, dtype=float
            Transformed data of shape (n_samples, n_features), or the input if no
            transform was fitted.

        """
        if self._quantile_transformer is None:
            return x

        return self._quantile_transformer.transform(x)

    @manage_torch_threads(max_threads=2)
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:  # noqa: PLR0915 # Too many statements
        """Fit the classifier to the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.ndarray, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """
        self._quantile_transformer = self._fit_feature_transform(x)
        x = self._transform_features(x)

        if self.experimental_hyperparameter_tuning:
            self.batch_size, self.learning_rate = _get_scaled_training_params(x)
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
                layers=self.layers,
                dropout=self.dropout,
            )

        if y.ndim == 2:  # noqa: PLR2004
            y = y[:, 1]

        random_state = self._np_rng.integers(0, 1_000_000)
        logger.info(f"Using random state {random_state} for train-test-split")
        indices_train, indices_test = train_test_split_indices(
            len(x), test_size=self.test_size, random_state=random_state
        )
        x_test = torch.as_tensor(x[indices_test], dtype=torch.float32)
        y_test = torch.Tensor(y[indices_test])

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # BCE on the logit: BCELoss on a softmax output loses the gradient wherever the float32
        # probability rounds to 0 or 1.
        loss = nn.BCEWithLogitsLoss()

        # Set model to training mode for BatchNorm
        self.network.train()

        x_train = torch.as_tensor(x[indices_train], dtype=torch.float32)
        y_train = torch.Tensor(y[indices_train])
        del x

        num_batches = (x_train.shape[0] // self.batch_size) - 1
        batch_start_list = np.arange(num_batches) * self.batch_size
        batch_stop_list = np.arange(num_batches) * self.batch_size + self.batch_size

        batch_count = 0

        for epoch in tqdm(range(self.epochs)):
            # shuffle batches
            order = self._np_rng.permutation(num_batches)
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
                            np.mean(
                                y_train_batch.detach().numpy() == (y_pred_train > 0)
                            )
                        )

                        self.metrics["test_accuracy"].append(
                            np.mean(y_test.detach().numpy() == (y_pred_test > 0))
                        )
                    self.network.train()

                batch_count += 1

        self._fitted = True

    @manage_torch_threads(max_threads=2)
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        y : np.ndarray, dtype=int
            Predicted class of shape (n_samples,).

        """
        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        assert (
            x.ndim == 2  # noqa: PLR2004
        ), "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert (
            x.shape[1] == self.input_dim
        ), "Input data must have the same number of features as the fitted classifier."

        assert self.network is not None, "Network must be initialized after fitting"
        return (self._predict_logit(x) > 0).astype(int)

    @manage_torch_threads(max_threads=2)
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict the class probabilities of the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        y : np.ndarray, dtype=float
            Predicted class probabilities of shape (n_samples, n_classes).

        """
        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        assert (
            x.ndim == 2  # noqa: PLR2004
        ), "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert (
            x.shape[1] == self.input_dim
        ), "Input data must have the same number of features as the fitted classifier."

        assert self.network is not None, "Network must be initialized after fitting"
        logit = self._predict_logit(x).astype(np.float64)
        # in float64 the sigmoid keeps the ranking of logits far beyond where float32 rounds to 1
        proba = 1 / (1 + np.exp(-logit))
        return np.stack([1 - proba, proba], axis=1)

    def _predict_logit(self, x: np.ndarray) -> np.ndarray:
        """Return the network's logit for every row, transforming and predicting in chunks."""
        self.network.eval()
        output = np.empty(len(x), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(x), _PREDICT_ROWS_CHUNK_SIZE):
                stop = start + _PREDICT_ROWS_CHUNK_SIZE
                x_chunk = torch.as_tensor(
                    self._transform_features(x[start:stop]), dtype=torch.float32
                )
                output[start:stop] = self.network(x_chunk).numpy()
        return output


class FeedForwardNN(nn.Module):
    """A simple feed forward neural network for FDR estimation."""

    def __init__(
        self,
        input_dim: int,
        layers: list[int] | None = None,
        dropout: float = 0.5,
    ):
        """Built a simple feed forward network for FDR estimation that outputs one logit."""
        if layers is None:
            layers = [20, 10, 5]
        super().__init__()
        self.input_dim = input_dim  # type: ignore[assignment]

        self.layers = [input_dim, *layers]  # type: ignore[assignment]
        self.dropout = dropout  # type: ignore[assignment]

        self._build_model()

    def _build_model(self) -> None:
        """Build the feed forward network model."""
        layers = []
        # add batch norm layer
        layers.append(nn.BatchNorm1d(self.input_dim))
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.layers[-1], 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:  # noqa: ANN401
        """Forward pass through the network, returning the logit of each sample."""
        return self.fc_layers(x).squeeze(1)


class RustMLPClassifier(Classifier):
    """Binary classifier for FDR estimation, computed in alphadia-search-rs.

    The same task and training as `BinaryClassifierLegacyNewBatching`: fully connected ReLU
    layers with one output logit, BCE-with-logits loss and Adam, on quantile-transformed
    features. It has no batch norm (the quantile transform already standardises the input)
    and no dropout. The feature transform and the training run in Rust on the
    alphadia-search-rs thread pool.

    All state is held as numpy arrays (quantiles and the flat network parameters), so the
    classifier pickles and deep-copies like a plain Python object. A second `fit` continues
    from the current parameters, as the torch classifier does.
    """

    def __init__(  # noqa: PLR0913 # Too many arguments
        self,
        layers: list[int] | None = None,
        epochs: int = 10,
        batch_size: int = 5000,
        learning_rate: float = 0.001,
        weight_decay: float = 0.00001,
        feature_transform: str = FeatureTransform.QUANTILE,
        quantile_subsample: int | None = None,
        *,
        experimental_hyperparameter_tuning: bool = False,
        random_state: int | None = None,
        matmul_backend: str | None = None,
        lr_schedule: str = LRSchedule.CONSTANT,
        weight_averaging: bool = False,
    ):
        """Binary classifier for FDR estimation, computed in alphadia-search-rs.

        Parameters
        ----------
        layers : list[int], default=[128, 64, 32]
            Hidden layer sizes.

        epochs : int, default=10
            Number of passes over the training data.

        batch_size : int, default=5000
            Rows per optimizer step.

        learning_rate : float, default=0.001
            Adam learning rate.

        weight_decay : float, default=0.00001
            L2 penalty added to the gradient.

        feature_transform : str, default="quantile"
            Transform applied to the features before they enter the network,
            either "none" or "quantile".

        quantile_subsample : int, optional
            Fit the quantile transform on this many random rows instead of all rows.

        experimental_hyperparameter_tuning : bool, default=False
            Scale batch size and learning rate with the number of rows, as the torch
            classifier does.

        random_state : int, optional
            Seed for the weight initialisation, row order and quantile subsample.

        matmul_backend : str, optional
            Override the matrix multiplication backend of alphadia-search-rs
            ("accelerate", "simd" or "gemm"). Defaults to the fastest one for the platform.

        lr_schedule : str, default="constant"
            Learning rate schedule, "constant" or "cosine" (half cosine from `learning_rate` to zero over the fit).

        weight_averaging : bool, default=False
            Predict with the mean of the parameters over the minibatches of the last epoch instead of the
            parameters after the last minibatch.

        """
        if feature_transform not in FeatureTransform.get_values():
            raise ValueError(
                f"Unknown feature transform: {feature_transform}. "
                f"Valid options are {FeatureTransform.get_values()}"
            )
        self.layers = [128, 64, 32] if layers is None else list(layers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feature_transform = feature_transform
        self.quantile_subsample = quantile_subsample
        self.experimental_hyperparameter_tuning = experimental_hyperparameter_tuning
        self.matmul_backend = matmul_backend
        self.lr_schedule = LRSchedule.CONSTANT
        self.weight_averaging = False
        self.configure_training(
            lr_schedule=lr_schedule, weight_averaging=weight_averaging
        )

        self.input_dim: int | None = None
        self._params: np.ndarray | None = None
        self._quantiles: np.ndarray | None = None
        self._np_rng = np.random.default_rng(seed=random_state)

    @property
    def fitted(self) -> bool:
        """Return whether the classifier has been fitted."""
        return self._params is not None

    def configure_training(
        self, *, lr_schedule: str | None = None, weight_averaging: bool | None = None
    ) -> None:
        """Set the learning rate schedule and the weight averaging of the next fits; None leaves a setting as is."""
        if lr_schedule is not None:
            if lr_schedule not in LRSchedule.get_values():
                raise ValueError(
                    f"Unknown learning rate schedule: {lr_schedule}. "
                    f"Valid options are {LRSchedule.get_values()}"
                )
            self.lr_schedule = lr_schedule
        if weight_averaging is not None:
            self.weight_averaging = weight_averaging

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier to the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.ndarray, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if y.ndim == 2:  # noqa: PLR2004
            y = y[:, 1]
        y = np.ascontiguousarray(y, dtype=np.float32)
        seed = int(self._np_rng.integers(0, 2**31))

        self._quantiles = self._fit_feature_transform(x, seed)
        x = self._transform_features(x)

        batch_size, learning_rate = self.batch_size, self.learning_rate
        if self.experimental_hyperparameter_tuning:
            batch_size, learning_rate = _get_scaled_training_params(x)

        if self._params is not None and self.input_dim != x.shape[1]:
            warnings.warn(
                "Input dimension of network has changed. Network has been reinitialized."
            )
            self._params = None
        self.input_dim = x.shape[1]

        self._params = fdr_mlp_fit(
            x,
            y,
            hidden_layers=self.layers,
            epochs=self.epochs,
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=self.weight_decay,
            seed=seed,
            params=self._params,
            backend=self.matmul_backend,
            cosine_schedule=self.lr_schedule == LRSchedule.COSINE,
            average_last_epoch=self.weight_averaging,
        )

    def reset(self) -> None:
        """Set the classifier back to an unfitted state; the next fit starts from new random weights."""
        self._params = None
        self._quantiles = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        y : np.ndarray, dtype=int
            Predicted class of shape (n_samples,).

        """
        return (self._predict_logit(x) > 0).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict the class probabilities of the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        y : np.ndarray, dtype=float
            Predicted class probabilities of shape (n_samples, n_classes).

        """
        logit = self._predict_logit(x).astype(np.float64)
        # in float64 the sigmoid keeps the ranking of logits far beyond where float32 rounds to 1
        proba = 1 / (1 + np.exp(-logit))
        return np.stack([1 - proba, proba], axis=1)

    def to_state_dict(self) -> dict:
        """Return the state of the classifier as a dictionary of plain values and numpy arrays."""
        return {
            "_fitted": self.fitted,
            "input_dim": self.input_dim,
            "layers": self.layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "feature_transform": self.feature_transform,
            "quantile_subsample": self.quantile_subsample,
            "lr_schedule": self.lr_schedule,
            "weight_averaging": self.weight_averaging,
            "params": None if self._params is None else self._params.copy(),
            "quantiles": None if self._quantiles is None else self._quantiles.copy(),
        }

    def from_state_dict(
        self, state_dict: dict, *, load_hyperparameters: bool = False
    ) -> None:
        """Load the state of the classifier from a dictionary written by `to_state_dict`.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing the state of the classifier.

        load_hyperparameters : bool, default=False
            Whether to load hyperparameters from the state dict.

        """
        if "params" not in state_dict:
            raise ValueError(
                "State dict was not written by RustMLPClassifier, it has no 'params'."
            )
        self.input_dim = state_dict["input_dim"]
        self.layers = list(state_dict["layers"])
        self._params = state_dict["params"]
        self._quantiles = state_dict["quantiles"]
        if load_hyperparameters:
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "feature_transform",
                "quantile_subsample",
                "lr_schedule",
                "weight_averaging",
            ):
                if key in state_dict:
                    setattr(self, key, state_dict[key])

    def _fit_feature_transform(self, x: np.ndarray, seed: int) -> np.ndarray | None:
        """Fit the configured feature transform; returns the quantiles or None for raw features."""
        if self.feature_transform == FeatureTransform.NONE:
            return None

        if len(x) < _MIN_ROWS_FOR_QUANTILE_TRANSFORM:
            logger.warning(
                f"Only {len(x):,} rows available for the quantile feature transform "
                f"(minimum {_MIN_ROWS_FOR_QUANTILE_TRANSFORM:,}), using the raw features instead."
            )
            return None

        return fdr_quantile_fit(
            x, n_quantiles=_N_QUANTILES, subsample=self.quantile_subsample, seed=seed
        )

    def _transform_features(self, x: np.ndarray) -> np.ndarray:
        """Map every feature to a standard normal score, or return x if no transform was fitted."""
        if self._quantiles is None:
            return x
        return fdr_quantile_transform(x, self._quantiles)

    def _predict_logit(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")
        if x.ndim != 2 or x.shape[1] != self.input_dim:  # noqa: PLR2004
            raise ValueError(
                f"Input data must have shape (n_samples, {self.input_dim}), got {x.shape}."
            )
        x = self._transform_features(np.ascontiguousarray(x, dtype=np.float32))
        return fdr_mlp_predict(
            x,
            hidden_layers=self.layers,
            params=self._params,
            backend=self.matmul_backend,
        )

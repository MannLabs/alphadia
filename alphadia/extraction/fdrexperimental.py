import numpy as np
import numba as nb
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import model_selection
import warnings 
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

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
    def fit(self, x : np.array, y : np.array):
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
    def predict(self, x : np.array):
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
    def predict_proba(self, x : np.array):
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
    def from_state_dict(self, state_dict : dict):
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
            input_dim : int = 10,
            output_dim : int = 2,
            test_size : float = 0.2,
            batch_size : int = 1000,
            epochs : int = 10,
            learning_rate : float = 0.0002,
            weight_decay : float = 0.00001,
            layers : List[int] = [100, 50, 20, 5],
            dropout : float = 0.001,
            metric_interval : int = 1000,
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

        layers : List[int], default=[100, 50, 20, 5]
            List of hidden layer sizes.

        dropout : float, default=0.001
            Dropout probability for training.

        metric_interval : int, default=1000
            Interval for logging metrics during training.

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
    
    def from_state_dict(self, state_dict : dict):
        """Load the state of the classifier from a dictionary.

        Parameters
        ----------

        dict : dict
            Dictionary containing the state of the classifier.
        
        """

        _state_dict = deepcopy(state_dict)

        if "network_state_dict" in _state_dict:
            self.network = FeedForwardNN(
                input_dim = _state_dict.pop("input_dim"),
                output_dim = _state_dict.pop("output_dim"),
                layers = _state_dict.pop("layers"),
                dropout = _state_dict.pop("dropout"),
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
            warnings.warn("Input dimension of network has changed. Network has been reinitialized.")
            force_reinit = True

        # check if network has to be initialized
        if self.network is None or force_reinit:
            self.input_dim = x.shape[1]
            self.network = FeedForwardNN(
                input_dim = self.input_dim,
                output_dim = self.output_dim,
                layers = self.layers,
                dropout = self.dropout,
            )
 
        # normalize input
        x = (x - x.mean(axis=0)) /(x.std(axis=0) + 1e-6)

        if y.ndim == 1:
            y = np.stack([1-y, y], axis=1)

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=self.test_size)

        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)

        optimizer = optim.Adam(
            self.network.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        loss = nn.BCELoss()

        batch_count = 0

        for j in range(self.epochs):
            order = np.random.permutation(len(x_train))
            x_train = torch.Tensor(x_train[order])
            y_train = torch.Tensor(y_train[order])
            
            for i, (batch_x, batch_y) in enumerate(zip(x_train.split(self.batch_size), y_train.split(self.batch_size))):
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
                            np.sum(y_train[:,1].detach().numpy() == np.argmax(y_pred_train, axis=1))/len(y_train)
                        )

                        self.metrics["test_accuracy"].append(
                            np.sum(y_test[:,1].detach().numpy() == np.argmax(y_pred_test, axis=1))/len(y_test)
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
        
        assert x.ndim == 2, "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert x.shape[1] == self.input_dim, "Input data must have the same number of features as the fitted classifier."
        
        x = (x - x.mean(axis=0)) /(x.std(axis=0) + 1e-6)
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
        
        assert x.ndim == 2, "Input data must have batch and feature dimension. (n_samples, n_features)"
        assert x.shape[1] == self.input_dim, "Input data must have the same number of features as the fitted classifier."
        
        x = (x - x.mean(axis=0)) /(x.std(axis=0) + 1e-6)
        self.network.eval()
        return self.network(torch.Tensor(x)).detach().numpy()

class FDRDataset(torch.utils.data.Dataset):

    def __init__(self, 
                df_target, 
                df_decoy, 
                available_columns,
                competetive=True, 
                group_channels=True):
        
        self.available_columns = available_columns
        self.competetive = competetive
        self.group_channels = group_channels

        if competetive:
            group_columns = ['elution_group_idx', 'channel'] if group_channels else ['elution_group_idx']
        else:
            group_columns = ['precursor_idx']

        self.df = pd.concat([df_target.copy(), df_decoy.copy()]).sort_values(group_columns)
        self.df['fdr_group'] = self.df.groupby(group_columns).ngroup()
        
        self.n_items = self.df['fdr_group'].nunique()

    def __len__(self):
        return self.n_items
                 
    def __getitem__(self, idx):
        df = self.df[self.df['fdr_group'] == idx]

        decoy_np = df['decoy'].values.astype(np.float32)
        decoy_np = np.stack([decoy_np, 1-decoy_np], axis=1)

        y_true = torch.tensor(decoy_np)

        return (
            torch.tensor(df[self.available_columns].values.astype(np.float32)), 
            y_true,
            torch.tensor(df['fdr_group'].values.astype(np.int64)),
        )

def batching_collate_fn(batch_list):

    # get first elements form list of tuples
    features, labels, groups = zip(*batch_list)
    features = torch.concat(features)
    labels = torch.concat(labels)
    groups = torch.concat(groups)

    return features, labels, groups

class FeedForwardNN(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim = 2,
        layers = [20, 10, 5],
        dropout = 0.5,
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
        for i in range(len(self.layers)-1):
            layers.append(nn.Linear(self.layers[i], self.layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.layers[-1], self.output_dim))
        # add softmax layer
        layers.append(nn.Softmax(dim=1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)
    
class SupervisedLoss():

    def __init__(self) -> None:
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):

        #output_selection = get_group_mask(y_pred.detach().detach().cpu().numpy(), y_groups.detach().cpu().numpy(), n_groups)

        #y_pred = y_pred[output_selection]
        #y_true = y_true[output_selection]

        return self.loss(y_pred, y_true)
    

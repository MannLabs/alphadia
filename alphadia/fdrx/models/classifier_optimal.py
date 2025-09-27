"""Optimal binary classifier implementation with PU learning."""

import os
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from alphadia.fdr.classifiers import Classifier
from alphadia.fdr.utils import manage_torch_threads

# Constants
NDIM_THRESHOLD = 2
FDR_THRESHOLD_01 = 0.01
FDR_THRESHOLD_001 = 0.001


class BinaryClassifierClaude(Classifier):
    """Minimal binary classifier using a simple neural network."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        test_size: float = 0.2,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 1000,
        pu_prior: float = 0.20,
        focus_gamma: float = 2.0,
        relabel_threshold: float = 0.5,
        **kwargs,
    ):
        """Minimal binary classifier.

        Parameters
        ----------
        test_size : float, default=0.2
            Fraction of the data to be used for testing.
        epochs : int, default=10
            Number of epochs for training.
        learning_rate : float, default=0.001
            Learning rate for training.
        batch_size : int, default=1000
            Batch size for training.
        pu_prior : float, default=0.05
            Prior probability of true positives in targets.
        focus_gamma : float, default=2.0
            Focal loss gamma parameter.
        relabel_threshold : float, default=0.8
            Confidence threshold for relabeling targets.
        **kwargs : dict
            Ignored keyword arguments.

        """
        self.test_size = test_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pu_prior = pu_prior
        self.focus_gamma = focus_gamma
        self.relabel_threshold = relabel_threshold
        self.network = None
        self._fitted = False

        if kwargs:
            warnings.warn(f"Unknown arguments: {kwargs}")

    def focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Compute focal loss for imbalanced classification."""
        bce_loss = functional.binary_cross_entropy(
            predictions, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def pu_relabel_targets(
        self, x_targets: torch.Tensor, current_labels: torch.Tensor
    ) -> torch.Tensor:
        """Relabel target samples based on current model predictions."""
        self.network.eval()
        with torch.no_grad():
            # Get predictions for target samples
            probs = self.network(x_targets)
            # Probability of being positive (class 1)
            pos_probs = probs[:, 1]

            # Strategy 1: Always select top pu_prior fraction as positive
            n_to_label = max(1, int(len(x_targets) * self.pu_prior))
            _, top_indices = torch.topk(pos_probs, n_to_label)

            new_labels = torch.zeros_like(current_labels)
            new_labels[top_indices] = 1.0

            # Strategy 2: Additionally, include high confidence samples above threshold
            high_conf_mask = pos_probs > self.relabel_threshold
            new_labels[high_conf_mask] = 1.0

            # But don't exceed reasonable limits (3x the prior)
            if new_labels.sum() > len(x_targets) * self.pu_prior * 3:
                # If too many, just keep the top ones
                n_keep = int(len(x_targets) * self.pu_prior * 3)
                _, keep_indices = torch.topk(pos_probs, n_keep)
                new_labels = torch.zeros_like(current_labels)
                new_labels[keep_indices] = 1.0

        return new_labels

    @property
    def fitted(self) -> bool:
        """Return whether the classifier has been fitted."""
        return self._fitted

    def to_state_dict(self) -> dict:
        """Save the state of the classifier as a dictionary."""
        state_dict = {
            "_fitted": self._fitted,
            "test_size": self.test_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "pu_prior": self.pu_prior,
            "focus_gamma": self.focus_gamma,
            "relabel_threshold": self.relabel_threshold,
        }
        if self._fitted:
            state_dict["network_state_dict"] = self.network.state_dict()
        return state_dict

    def from_state_dict(
        self, state_dict: dict, *, load_hyperparameters: bool = False
    ) -> None:
        """Load the state of the classifier from a dictionary."""
        if "network_state_dict" in state_dict:
            # Would need input_dim to rebuild network - simplified for this minimal version
            self._fitted = True
        if load_hyperparameters:
            self.test_size = state_dict.get("test_size", self.test_size)
            self.epochs = state_dict.get("epochs", self.epochs)
            self.learning_rate = state_dict.get("learning_rate", self.learning_rate)
            self.batch_size = state_dict.get("batch_size", self.batch_size)
            self.pu_prior = state_dict.get("pu_prior", self.pu_prior)
            self.focus_gamma = state_dict.get("focus_gamma", self.focus_gamma)
            self.relabel_threshold = state_dict.get(
                "relabel_threshold", self.relabel_threshold
            )

    @manage_torch_threads(max_threads=2)
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier using PU learning approach."""
        # Set environment variables for optimal CPU performance
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        input_dim = x.shape[1]

        # Create network matching working classifier architecture
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(50, 2),
            nn.Softmax(dim=1),
        )

        # Convert y to proper format - original targets (1) and decoys (0)
        if y.ndim == NDIM_THRESHOLD:
            y = np.argmax(y, axis=1)  # Convert from [1-y, y] format to class indices

        # Split into targets (originally labeled as 1) and decoys (labeled as 0)
        target_mask = y == 1
        decoy_mask = y == 0

        x_targets = x[target_mask]
        x_decoys = x[decoy_mask]

        # Convert to tensors
        x_targets_tensor = torch.Tensor(x_targets)
        x_decoys_tensor = torch.Tensor(x_decoys)
        target_labels_tensor = torch.zeros(len(x_targets))

        optimizer = optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=0.00001
        )

        # PU Learning: Iterative training with relabeling
        for epoch in tqdm(range(self.epochs)):
            # Step 1: Relabel targets based on current model (after epoch 0)
            if epoch > 0:
                target_labels_tensor = self.pu_relabel_targets(
                    x_targets_tensor, target_labels_tensor
                )
                # Print relabeling progress (removed for production)

            # Step 2: Create training data with decoys (all negative) + relabeled targets
            # Only train on confident examples
            confident_target_mask = (
                target_labels_tensor == 1
            )  # High confidence positives

            if confident_target_mask.sum() > 0:
                # Use confident targets as positives and all decoys as negatives
                x_train = torch.cat(
                    [x_targets_tensor[confident_target_mask], x_decoys_tensor]
                )
                y_train = torch.cat(
                    [
                        torch.ones(
                            confident_target_mask.sum()
                        ),  # Confident targets as positive
                        torch.zeros(len(x_decoys)),  # All decoys as negative
                    ]
                )
            else:
                # If no confident targets yet, train only on decoys vs random targets
                n_pseudo_pos = max(1, int(len(x_targets) * self.pu_prior))
                random_indices = torch.randperm(len(x_targets))[:n_pseudo_pos]

                x_train = torch.cat([x_targets_tensor[random_indices], x_decoys_tensor])
                y_train = torch.cat(
                    [torch.ones(n_pseudo_pos), torch.zeros(len(x_decoys))]
                )

            # Convert to 2D format for BCE loss
            y_train_2d = torch.stack([1 - y_train, y_train], dim=1)

            # Create DataLoader for this epoch
            train_dataset = TensorDataset(x_train, y_train_2d)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

            # Step 3: Train on current labels with focal loss
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                self.network.train()
                y_pred = self.network(batch_x)

                # Use focal loss to focus on hard examples
                loss = self.focal_loss(
                    y_pred, batch_y, alpha=1.0, gamma=self.focus_gamma
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self._fitted = True

    @manage_torch_threads(max_threads=2)
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the data."""
        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        self.network.eval()
        with torch.no_grad():
            return np.argmax(self.network(torch.Tensor(x)).numpy(), axis=1)

    @manage_torch_threads(max_threads=2)
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict the class probabilities of the data."""
        if not self.fitted:
            raise ValueError("Classifier has not been fitted yet.")

        self.network.eval()
        with torch.no_grad():
            return self.network(torch.Tensor(x)).numpy()

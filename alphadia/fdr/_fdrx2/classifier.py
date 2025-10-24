"""XGBoost classifier for FDR estimation."""

import pickle

import numpy as np
import xgboost as xgb
from alphadia.fdr.classifiers import Classifier

# ruff: noqa


class BinaryClassifierXGBoost(Classifier):
    """Binary Classifier using XGBoost."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        random_state: int | None = None,
        **kwargs,
    ):
        """Binary Classifier using XGBoost.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting rounds.

        max_depth : int, default=6
            Maximum tree depth.

        learning_rate : float, default=0.3
            Boosting learning rate.

        random_state : int, optional
            Random seed for reproducibility.

        **kwargs : dict
            Additional XGBoost parameters.

        """
        if xgb is None:
            raise ImportError("XGBoost not found. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = None
        self._fitted = False
        self.input_dim = None

    @property
    def fitted(self) -> bool:
        """Return whether the classifier has been fitted."""
        return self._fitted

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier to the data.

        Parameters
        ----------
        x : np.ndarray, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.ndarray, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """
        self.input_dim = x.shape[1]

        if y.ndim == 2:  # noqa: PLR2004
            y = y[:, 1]

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs,
        )

        self.model.fit(x, y)
        self._fitted = True

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

        return self.model.predict(x)

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

        return self.model.predict_proba(x)

    def to_state_dict(self) -> dict:
        """Return a state dict of the classifier.

        Returns
        -------
        state_dict : dict
            State dict of the classifier.

        """
        state_dict = {
            "_fitted": self._fitted,
            "input_dim": self.input_dim,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "kwargs": self.kwargs,
        }

        if self._fitted:
            state_dict["model"] = pickle.dumps(self.model)

        return state_dict

    def from_state_dict(self, state_dict: dict) -> None:
        """Load a state dict of the classifier.

        Parameters
        ----------
        state_dict : dict
            State dict of the classifier.

        """
        self._fitted = state_dict["_fitted"]
        self.input_dim = state_dict["input_dim"]
        self.n_estimators = state_dict["n_estimators"]
        self.max_depth = state_dict["max_depth"]
        self.learning_rate = state_dict["learning_rate"]
        self.random_state = state_dict["random_state"]
        self.kwargs = state_dict["kwargs"]

        if self._fitted:
            self.model = pickle.loads(state_dict["model"])

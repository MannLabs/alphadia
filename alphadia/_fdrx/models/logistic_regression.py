"""Implements the Logistic Regression classifier for use within the Alphadia framework."""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alphadia.fdr.classifiers import Classifier

logger = logging.getLogger()


class LogisticRegressionClassifier(Classifier):
    """Binary classifier using a logistic regression model."""

    def __init__(self) -> None:
        """Initializing a binary classifier using a logistic regression model."""
        self.scaler = StandardScaler()
        self.model = LogisticRegression()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """Return whether the classifier has been fitted."""
        return self._fitted

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier to the data.

        Parameters
        ----------
        x : np.array, dtype=float
            Training data of shape (n_samples, n_features).

        y : np.array, dtype=int
            Target values of shape (n_samples,) or (n_samples, n_classes).

        """
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self._fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the data.

        Parameters
        ----------
        x : np.array, dtype=float
            Data of shape (n_samples, n_features).

        Returns
        -------
        y : np.array, dtype=float
            Predicted class probabilities of shape (n_samples, n_classes).

        """
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
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
        x_scaled = self.scaler.transform(x)
        return self.model.predict_proba(x_scaled)

    def to_state_dict(self) -> dict:
        """Return the state of the classifier as a dictionary.

        Returns
        -------
        dict : dict
            Dictionary containing the state of the classifier.

        """
        state_dict = {"_fitted": self._fitted}

        if self._fitted:
            state_dict.update(
                {
                    "scaler_mean": self.scaler.mean_,
                    "scaler_var": self.scaler.var_,
                    "scaler_scale": self.scaler.scale_,
                    "scaler_n_samples_seen": self.scaler.n_samples_seen_,
                    "model_coef": self.model.coef_,
                    "model_intercept": self.model.intercept_,
                    "model_classes": self.model.classes_,
                    "is_fitted": self._fitted,
                }
            )

        return state_dict

    def from_state_dict(self, state_dict: dict) -> None:
        """Load the state of the classifier from a dictionary.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing the state of the classifier.

        """
        self._fitted = state_dict["_fitted"]

        if self._fitted:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(state_dict["scaler_mean"])
            self.scaler.var_ = np.array(state_dict["scaler_var"])
            self.scaler.scale_ = np.array(state_dict["scaler_scale"])
            self.scaler.n_samples_seen_ = np.array(state_dict["scaler_n_samples_seen"])

            self.model = LogisticRegression()
            self.model.coef_ = np.array(state_dict["model_coef"])
            self.model.intercept_ = np.array(state_dict["model_intercept"])
            self.model.classes_ = np.array(state_dict["model_classes"])

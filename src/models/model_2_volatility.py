"""
Model 2: Volatility Squeeze Detector
Uses SVM to detect periods of low volatility that precede explosive moves.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Model2Volatility(BaseModel):
    """Volatility squeeze detection model using SVM."""

    def __init__(self, config: dict):
        """
        Initialize Model 2.

        Args:
            config: Model configuration from config.yaml
        """
        super().__init__(config)

        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})

        self.model = SVC(
            kernel=hyperparams.get('kernel', 'rbf'),
            C=hyperparams.get('C', 1.0),
            gamma=hyperparams.get('gamma', 'scale'),
            probability=True,  # Enable predict_proba
            random_state=42
        )

        # SVM requires feature scaling
        self.scaler = StandardScaler()

        # Get feature list from config
        self.feature_names = config.get('features', [])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the SVM classifier.

        Args:
            X_train: Training features
            y_train: Training labels (0/1)
        """
        try:
            logger.info(f"Training {self.model_name}...")

            # Validate features
            if not self.validate_features(X_train):
                raise ValueError("Feature validation failed")

            # Select configured features
            X_train_selected = X_train[self.feature_names]

            # Scale features (SVM requires scaling)
            X_train_scaled = self.scaler.fit_transform(X_train_selected)

            # Handle class imbalance
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()

            if n_positive > 0:
                # Use class_weight for SVM
                class_weight = {0: 1.0, 1: n_negative / n_positive}
                self.model.set_params(class_weight=class_weight)
                logger.info(f"Class imbalance: {n_negative} negative, {n_positive} positive")

            # Fit model
            self.model.fit(X_train_scaled, y_train)

            self.is_trained = True
            logger.info(f"{self.model_name} training complete")

        except Exception as e:
            logger.error(f"Error training {self.model_name}: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Features

        Returns:
            Array of predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_selected = X[self.feature_names]
        X_scaled = self.scaler.transform(X_selected)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities of positive class.

        Args:
            X: Features

        Returns:
            Array of probabilities (0 to 1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_selected = X[self.feature_names]
        X_scaled = self.scaler.transform(X_selected)
        probas = self.model.predict_proba(X_scaled)

        # Return probability of positive class
        return probas[:, 1]

"""
Model 1: Technical Breakout Classifier
Uses XGBoost to predict explosive breakouts based on technical indicators.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Model1Breakout(BaseModel):
    """Technical breakout prediction model using XGBoost."""

    def __init__(self, config: dict):
        """
        Initialize Model 1.

        Args:
            config: Model configuration from config.yaml
        """
        super().__init__(config)

        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})

        self.model = xgb.XGBClassifier(
            n_estimators=hyperparams.get('n_estimators', 300),
            max_depth=hyperparams.get('max_depth', 6),
            learning_rate=hyperparams.get('learning_rate', 0.05),
            subsample=hyperparams.get('subsample', 0.8),
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )

        # Get feature list from config
        self.feature_names = config.get('features', [])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the XGBoost classifier.

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

            # Handle class imbalance with scale_pos_weight
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()

            if n_positive > 0:
                scale_pos_weight = n_negative / n_positive
                self.model.set_params(scale_pos_weight=scale_pos_weight)
                logger.info(f"Class imbalance: {n_negative} negative, {n_positive} positive (scale_pos_weight={scale_pos_weight:.2f})")

            # Fit model
            self.model.fit(X_train_selected, y_train)

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
        return self.model.predict(X_selected)

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
        probas = self.model.predict_proba(X_selected)

        # Return probability of positive class
        return probas[:, 1]

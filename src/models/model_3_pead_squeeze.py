"""
Model 3: PEAD + Short Squeeze Hybrid
Combines Post-Earnings-Announcement Drift with short squeeze detection.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Model3PEADSqueeze(BaseModel):
    """PEAD + Short Squeeze hybrid model using XGBoost."""

    def __init__(self, config: dict):
        """
        Initialize Model 3.

        Args:
            config: Model configuration from config.yaml
        """
        super().__init__(config)

        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})

        self.model = xgb.XGBClassifier(
            n_estimators=hyperparams.get('n_estimators', 400),
            max_depth=hyperparams.get('max_depth', 8),
            learning_rate=hyperparams.get('learning_rate', 0.03),
            subsample=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )

        # Build combined feature list from PEAD and Squeeze components
        self.feature_names = self._build_feature_list(config)

    def _build_feature_list(self, config: dict) -> list:
        """
        Build combined feature list from PEAD and Squeeze components.

        Args:
            config: Model configuration

        Returns:
            List of feature names
        """
        features = []

        components = config.get('components', {})

        # Add PEAD features if enabled
        pead_config = components.get('pead', {})
        if pead_config.get('enabled', True):
            pead_features = pead_config.get('features', [])
            features.extend(pead_features)
            logger.debug(f"Added {len(pead_features)} PEAD features")

        # Add Squeeze features if enabled
        squeeze_config = components.get('squeeze', {})
        if squeeze_config.get('enabled', True):
            squeeze_features = squeeze_config.get('features', [])
            features.extend(squeeze_features)
            logger.debug(f"Added {len(squeeze_features)} Squeeze features")

        return features

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the hybrid XGBoost classifier.

        Args:
            X_train: Training features (must include both PEAD and Squeeze features)
            y_train: Training labels (0/1)
        """
        try:
            logger.info(f"Training {self.model_name}...")

            # Check that we have features from both components
            available_features = [f for f in self.feature_names if f in X_train.columns]

            if len(available_features) == 0:
                logger.warning("No configured features found in training data")
                # Fall back to all numeric features
                available_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

            logger.info(f"Using {len(available_features)} features for training")

            # Select available features
            X_train_selected = X_train[available_features]
            self.feature_names = available_features

            # Handle class imbalance
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

    def get_component_importance(self) -> dict:
        """
        Analyze which component (PEAD vs Squeeze) is more important.

        Returns:
            Dictionary with component importance breakdown
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return {}

        feature_importance = self.get_feature_importance()

        if feature_importance is None:
            return {}

        # Categorize features by component
        pead_keywords = ['earnings', 'eps', 'revenue', 'analyst']
        squeeze_keywords = ['short', 'reddit', 'mentions', 'sentiment']

        pead_importance = 0
        squeeze_importance = 0
        other_importance = 0

        for _, row in feature_importance.iterrows():
            feature = row['feature'].lower()
            importance = row['importance']

            if any(kw in feature for kw in pead_keywords):
                pead_importance += importance
            elif any(kw in feature for kw in squeeze_keywords):
                squeeze_importance += importance
            else:
                other_importance += importance

        total = pead_importance + squeeze_importance + other_importance

        return {
            'pead_importance_pct': (pead_importance / total * 100) if total > 0 else 0,
            'squeeze_importance_pct': (squeeze_importance / total * 100) if total > 0 else 0,
            'other_importance_pct': (other_importance / total * 100) if total > 0 else 0
        }

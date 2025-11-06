"""
Ensemble voting system - combines predictions from all models.
Implements weighted voting with dynamic weight updates based on performance.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from src.models.model_1_breakout import Model1Breakout
from src.models.model_2_volatility import Model2Volatility
from src.models.model_3_pead_squeeze import Model3PEADSqueeze
from src.models.model_4_garp import Model4GARP
from src.models.model_6_anomaly import Model6Anomaly
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EnsembleVoter:
    """Ensemble voting system for combining model predictions."""

    def __init__(self, config: Dict):
        """
        Initialize ensemble voter.

        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.models = {}
        self.model_weights = {}
        self.ensemble_config = config.get('ensemble', {})

    def load_all_models(self) -> None:
        """Load all trained models from disk."""
        try:
            models_config = self.config.get('models', {})
            models_dir = Path('models/trained')

            # Model 1: Breakout
            if models_config.get('model_1_breakout', {}).get('enabled', False):
                self.models[1] = Model1Breakout(models_config['model_1_breakout'])
                model_file = models_dir / 'model_1_breakout.pkl'
                if model_file.exists():
                    self.models[1].load(str(model_file))
                    logger.info("Loaded Model 1 (Breakout)")

            # Model 2: Volatility
            if models_config.get('model_2_volatility', {}).get('enabled', False):
                self.models[2] = Model2Volatility(models_config['model_2_volatility'])
                model_file = models_dir / 'model_2_volatility.pkl'
                if model_file.exists():
                    self.models[2].load(str(model_file))
                    logger.info("Loaded Model 2 (Volatility)")

            # Model 3: PEAD + Squeeze
            if models_config.get('model_3_pead_squeeze', {}).get('enabled', False):
                self.models[3] = Model3PEADSqueeze(models_config['model_3_pead_squeeze'])
                model_file = models_dir / 'model_3_pead_squeeze.pkl'
                if model_file.exists():
                    self.models[3].load(str(model_file))
                    logger.info("Loaded Model 3 (PEAD + Squeeze)")

            # Model 4: GARP (filter only)
            if models_config.get('model_4_garp', {}).get('enabled', False):
                self.models[4] = Model4GARP(models_config['model_4_garp'])
                model_file = models_dir / 'model_4_garp.pkl'
                if model_file.exists():
                    self.models[4].load(str(model_file))
                    logger.info("Loaded Model 4 (GARP Filter)")

            # Model 6: Anomaly
            if models_config.get('model_6_anomaly', {}).get('enabled', False):
                self.models[6] = Model6Anomaly(models_config['model_6_anomaly'])
                model_file = models_dir / 'model_6_anomaly.pkl'
                if model_file.exists():
                    self.models[6].load(str(model_file))
                    logger.info("Loaded Model 6 (Anomaly)")

            # Load model weights
            self.model_weights = self.load_model_weights()

            logger.info(f"Loaded {len(self.models)} models")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def load_model_weights(self) -> Dict[int, float]:
        """
        Load model weights from performance tracking.

        Returns:
            Dictionary mapping model_id -> weight
        """
        weights_file = Path('models/performance/model_weights.json')

        try:
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    weights = json.load(f)

                # Convert string keys to int
                weights = {int(k): v for k, v in weights.items()}

                logger.info(f"Loaded model weights: {weights}")
                return weights

        except Exception as e:
            logger.warning(f"Error loading weights: {str(e)}")

        # Default equal weights for voting models
        voting_models = [m for m in self.models.keys() if m != 4]  # Exclude filter
        default_weight = 1.0 / len(voting_models) if voting_models else 0.25

        weights = {model_id: default_weight for model_id in voting_models}

        logger.info(f"Using default equal weights: {weights}")

        return weights

    def generate_signals(
        self,
        asset_type: str,
        tickers: List[str],
        features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate buy signals for a list of tickers.

        Args:
            asset_type: 'stocks' or 'crypto'
            tickers: List of ticker symbols
            features: Dictionary mapping ticker -> feature DataFrame

        Returns:
            DataFrame with columns [ticker, signal, confidence, contributing_models, ...]
        """
        try:
            logger.info(f"Generating signals for {len(tickers)} {asset_type}...")

            # Step 1: Filter universe with Model 4 (stocks only)
            if asset_type == 'stocks' and 4 in self.models:
                logger.info("Applying GARP filter (Model 4)...")
                tickers = self.models[4].filter_stocks(tickers, features)
                logger.info(f"After GARP filter: {len(tickers)} tickers")

            if len(tickers) == 0:
                logger.warning("No tickers passed GARP filter")
                return pd.DataFrame()

            # Step 2: Get voting models for this asset type
            ensemble_config = self.ensemble_config.get(asset_type, {})
            voting_models = ensemble_config.get('models_participating', [])

            # Filter to only loaded models
            voting_models = [m for m in voting_models if m in self.models and m != 4]

            if not voting_models:
                logger.error("No voting models available")
                return pd.DataFrame()

            logger.info(f"Voting models: {voting_models}")

            # Step 3: Generate predictions for each ticker
            signals = []

            for ticker in tickers:
                if ticker not in features:
                    logger.warning(f"Features not found for {ticker}")
                    continue

                X = features[ticker]

                if len(X) == 0:
                    continue

                # Get latest data point for prediction
                X_latest = X.tail(1)

                # Get probabilities from each model
                model_probas = {}

                for model_id in voting_models:
                    try:
                        proba = self.models[model_id].predict_proba(X_latest)
                        model_probas[model_id] = proba[0] if len(proba) > 0 else 0
                    except Exception as e:
                        logger.warning(f"Error getting prediction from Model {model_id} for {ticker}: {str(e)}")
                        model_probas[model_id] = 0

                # Step 4: Weighted voting
                weighted_confidence = sum(
                    model_probas[mid] * self.model_weights.get(mid, 0)
                    for mid in voting_models
                )

                # Step 5: Determine if signal triggers
                min_models = ensemble_config.get('min_models_required', 2)

                # Count how many models are above 0.5 threshold
                models_agreeing = sum(1 for p in model_probas.values() if p > 0.5)

                if models_agreeing >= min_models:
                    # Determine confidence tier
                    tiers = ensemble_config.get('confidence_tiers', {})
                    if weighted_confidence >= tiers.get('high', 0.85):
                        confidence = 'high'
                    elif weighted_confidence >= tiers.get('medium', 0.70):
                        confidence = 'medium'
                    else:
                        confidence = 'low'

                    signal = {
                        'ticker': ticker,
                        'signal': 'BUY',
                        'confidence': confidence,
                        'weighted_score': weighted_confidence,
                        'models_agreeing': models_agreeing
                    }

                    # Add individual model probabilities
                    for model_id in voting_models:
                        signal[f'model_{model_id}_proba'] = model_probas.get(model_id, 0)

                    signals.append(signal)

            if not signals:
                logger.info("No signals generated")
                return pd.DataFrame()

            df_signals = pd.DataFrame(signals)

            # Sort by confidence and weighted score
            df_signals = df_signals.sort_values(
                ['confidence', 'weighted_score'],
                ascending=[False, False]
            )

            logger.info(f"Generated {len(df_signals)} signals")
            logger.info(f"  High confidence: {(df_signals['confidence'] == 'high').sum()}")
            logger.info(f"  Medium confidence: {(df_signals['confidence'] == 'medium').sum()}")
            logger.info(f"  Low confidence: {(df_signals['confidence'] == 'low').sum()}")

            return df_signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()

    def update_weights(self, recent_performance: Dict[int, float]) -> None:
        """
        Update model weights based on recent performance.

        Args:
            recent_performance: Dictionary mapping model_id -> performance metric (e.g., precision)
        """
        try:
            logger.info("Updating model weights based on recent performance...")

            # Normalize performance to weights
            total = sum(recent_performance.values())

            if total > 0:
                self.model_weights = {
                    model_id: perf / total
                    for model_id, perf in recent_performance.items()
                }
            else:
                logger.warning("Total performance is zero, using equal weights")
                self.model_weights = {
                    model_id: 1.0 / len(recent_performance)
                    for model_id in recent_performance.keys()
                }

            # Save updated weights
            weights_file = Path('models/performance/model_weights.json')
            weights_file.parent.mkdir(parents=True, exist_ok=True)

            with open(weights_file, 'w') as f:
                json.dump(self.model_weights, f, indent=2)

            logger.info(f"Updated model weights: {self.model_weights}")

        except Exception as e:
            logger.error(f"Error updating weights: {str(e)}")

    def backtest_ensemble(
        self,
        features: Dict[str, pd.DataFrame],
        labels: Dict[str, pd.DataFrame],
        asset_type: str = 'stocks'
    ) -> pd.DataFrame:
        """
        Backtest ensemble on historical data.

        Args:
            features: Dictionary mapping ticker -> feature DataFrame
            labels: Dictionary mapping ticker -> label DataFrame
            asset_type: 'stocks' or 'crypto'

        Returns:
            DataFrame with backtest results
        """
        try:
            logger.info("Running ensemble backtest...")

            all_predictions = []

            for ticker in features.keys():
                if ticker not in labels:
                    continue

                X = features[ticker]
                y = labels[ticker]

                # Merge features and labels on date
                df = pd.merge(X, y, on='date', how='inner')

                if len(df) == 0:
                    continue

                # Generate predictions for each row
                for idx, row in df.iterrows():
                    X_row = pd.DataFrame([row])

                    # Get predictions from voting models
                    voting_models = self.ensemble_config.get(asset_type, {}).get('models_participating', [])
                    voting_models = [m for m in voting_models if m in self.models and m != 4]

                    model_probas = {}
                    for model_id in voting_models:
                        try:
                            proba = self.models[model_id].predict_proba(X_row)
                            model_probas[model_id] = proba[0] if len(proba) > 0 else 0
                        except:
                            model_probas[model_id] = 0

                    # Weighted confidence
                    weighted_confidence = sum(
                        model_probas[mid] * self.model_weights.get(mid, 0)
                        for mid in voting_models
                    )

                    # Binary prediction
                    prediction = 1 if weighted_confidence >= 0.5 else 0

                    all_predictions.append({
                        'ticker': ticker,
                        'date': row['date'],
                        'prediction': prediction,
                        'confidence': weighted_confidence,
                        'actual': row.get('label', 0)
                    })

            if not all_predictions:
                logger.warning("No backtest predictions generated")
                return pd.DataFrame()

            df_results = pd.DataFrame(all_predictions)

            # Calculate performance metrics
            accuracy = (df_results['prediction'] == df_results['actual']).mean()
            logger.info(f"Ensemble backtest accuracy: {accuracy:.2%}")

            return df_results

        except Exception as e:
            logger.error(f"Error in ensemble backtest: {str(e)}")
            return pd.DataFrame()

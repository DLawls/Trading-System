"""
Ensemble Manager for combining multiple ML models with various ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from loguru import logger
from dataclasses import dataclass
from enum import Enum

try:
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Limited ensemble capabilities.")

from .model_store import ModelStore
from .model_predictor import ModelPredictor, PredictionResult


class EnsembleMethod(Enum):
    """Types of ensemble methods"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTING = "majority_voting"
    STACKING = "stacking"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble creation"""
    method: EnsembleMethod
    model_ids: List[str]
    weights: Optional[List[float]] = None
    meta_learner_type: str = "logistic_regression"  # For stacking
    confidence_threshold: float = 0.5
    disagreement_threshold: float = 0.3
    performance_window: int = 100  # For dynamic weighting
    use_uncertainty: bool = True


class EnsemblePerformance:
    """Tracks ensemble performance over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = []
        self.actuals = []
        self.individual_performances = {}
        
    def add_prediction(
        self,
        ensemble_prediction: float,
        individual_predictions: Dict[str, float],
        actual: float = None
    ):
        """Add a prediction to performance tracking"""
        
        self.predictions.append({
            'ensemble': ensemble_prediction,
            'individual': individual_predictions,
            'actual': actual,
            'timestamp': datetime.utcnow()
        })
        
        if actual is not None:
            self.actuals.append(actual)
        
        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
            self.actuals = self.actuals[-self.window_size:]
    
    def get_model_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        
        if len(self.actuals) < 10:  # Need minimum data
            return {}
        
        model_errors = {}
        
        # Calculate errors for each model
        for i, pred_data in enumerate(self.predictions[-len(self.actuals):]):
            actual = self.actuals[i]
            
            for model_id, pred_value in pred_data['individual'].items():
                if model_id not in model_errors:
                    model_errors[model_id] = []
                
                error = abs(actual - pred_value)
                model_errors[model_id].append(error)
        
        # Calculate weights (inverse of average error)
        weights = {}
        for model_id, errors in model_errors.items():
            avg_error = np.mean(errors)
            weights[model_id] = 1.0 / (avg_error + 1e-8)  # Add small epsilon
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights


class EnsembleManager:
    """
    Manages ensemble methods for combining multiple ML models
    """
    
    def __init__(
        self,
        model_store: Optional[ModelStore] = None,
        model_predictor: Optional[ModelPredictor] = None
    ):
        """
        Initialize the ensemble manager
        
        Args:
            model_store: ModelStore instance
            model_predictor: ModelPredictor instance
        """
        self.model_store = model_store or ModelStore()
        self.model_predictor = model_predictor or ModelPredictor(self.model_store)
        
        self.ensembles = {}  # Store ensemble configurations
        self.performance_trackers = {}  # Track performance for each ensemble
        self.meta_learners = {}  # Store trained meta-learners for stacking
        
    def create_ensemble(
        self,
        ensemble_id: str,
        config: EnsembleConfig,
        description: str = ""
    ) -> bool:
        """
        Create a new ensemble
        
        Args:
            ensemble_id: Unique identifier for the ensemble
            config: Ensemble configuration
            description: Description of the ensemble
            
        Returns:
            True if successful
        """
        
        try:
            # Validate model IDs
            available_models = [m.model_id for m in self.model_store.list_models()]
            invalid_models = [m for m in config.model_ids if m not in available_models]
            
            if invalid_models:
                logger.error(f"Invalid model IDs: {invalid_models}")
                return False
            
            # Validate weights if provided
            if config.weights:
                if len(config.weights) != len(config.model_ids):
                    logger.error("Number of weights must match number of models")
                    return False
                
                if abs(sum(config.weights) - 1.0) > 1e-6:
                    logger.warning("Weights don't sum to 1.0, normalizing...")
                    total = sum(config.weights)
                    config.weights = [w / total for w in config.weights]
            
            # Store ensemble configuration
            self.ensembles[ensemble_id] = {
                'config': config,
                'description': description,
                'created_at': datetime.utcnow(),
                'prediction_count': 0
            }
            
            # Initialize performance tracker
            self.performance_trackers[ensemble_id] = EnsemblePerformance(
                window_size=config.performance_window
            )
            
            logger.info(f"Created ensemble {ensemble_id} with {len(config.model_ids)} models")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ensemble {ensemble_id}: {e}")
            return False
    
    def predict_ensemble(
        self,
        ensemble_id: str,
        features: Union[pd.DataFrame, pd.Series, Dict[str, float]],
        asset_id: str = None,
        timestamp: datetime = None,
        include_individual: bool = False
    ) -> Union[PredictionResult, Tuple[PredictionResult, Dict[str, PredictionResult]]]:
        """
        Make prediction using an ensemble
        
        Args:
            ensemble_id: Ensemble identifier
            features: Input features
            asset_id: Asset identifier
            timestamp: Prediction timestamp
            include_individual: Whether to return individual model predictions
            
        Returns:
            Ensemble prediction result, optionally with individual results
        """
        
        if ensemble_id not in self.ensembles:
            raise ValueError(f"Ensemble {ensemble_id} not found")
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        if asset_id is None:
            asset_id = "unknown"
        
        try:
            config = self.ensembles[ensemble_id]['config']
            
            # Get individual predictions
            individual_results = {}
            for model_id in config.model_ids:
                try:
                    result = self.model_predictor.predict(
                        features, model_id, asset_id, timestamp
                    )
                    individual_results[model_id] = result
                except Exception as e:
                    logger.warning(f"Error with model {model_id}: {e}")
                    continue
            
            if not individual_results:
                raise ValueError("No successful individual predictions")
            
            # Combine predictions based on method
            ensemble_result = self._combine_predictions(
                individual_results, config, ensemble_id
            )
            
            # Update ensemble metadata
            self.ensembles[ensemble_id]['prediction_count'] += 1
            
            # Track performance
            individual_predictions = {
                k: v.prediction_value for k, v in individual_results.items()
            }
            self.performance_trackers[ensemble_id].add_prediction(
                ensemble_result.prediction_value, individual_predictions
            )
            
            logger.debug(f"Ensemble {ensemble_id} prediction: {ensemble_result.prediction_value}")
            
            if include_individual:
                return ensemble_result, individual_results
            else:
                return ensemble_result
                
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise
    
    def train_stacking_ensemble(
        self,
        ensemble_id: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> bool:
        """
        Train a stacking ensemble meta-learner
        
        Args:
            ensemble_id: Ensemble identifier
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            True if successful
        """
        
        if ensemble_id not in self.ensembles:
            raise ValueError(f"Ensemble {ensemble_id} not found")
        
        config = self.ensembles[ensemble_id]['config']
        
        if config.method != EnsembleMethod.STACKING:
            raise ValueError("Ensemble must use stacking method")
        
        try:
            logger.info(f"Training stacking meta-learner for ensemble {ensemble_id}")
            
            # Generate base model predictions for training data
            base_predictions = []
            
            for model_id in config.model_ids:
                try:
                    # Get predictions for each sample
                    model_preds = []
                    for i, (_, row) in enumerate(X_train.iterrows()):
                        result = self.model_predictor.predict(row.to_dict(), model_id)
                        model_preds.append(result.prediction_value)
                    
                    base_predictions.append(model_preds)
                    
                except Exception as e:
                    logger.warning(f"Error getting predictions from {model_id}: {e}")
                    continue
            
            if not base_predictions:
                raise ValueError("No base model predictions available")
            
            # Create meta-features (base model predictions as features)
            meta_X = np.column_stack(base_predictions)
            meta_y = y_train.values
            
            # Train meta-learner
            if config.meta_learner_type == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression(random_state=42)
            elif config.meta_learner_type == "linear_regression":
                from sklearn.linear_model import LinearRegression
                meta_learner = LinearRegression()
            else:
                raise ValueError(f"Unsupported meta-learner: {config.meta_learner_type}")
            
            meta_learner.fit(meta_X, meta_y)
            
            # Store trained meta-learner
            self.meta_learners[ensemble_id] = meta_learner
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_score = self._evaluate_stacking_ensemble(
                    ensemble_id, X_val, y_val
                )
                logger.info(f"Validation score for ensemble {ensemble_id}: {val_score}")
            
            logger.info(f"Successfully trained meta-learner for ensemble {ensemble_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error training stacking ensemble: {e}")
            return False
    
    def update_ensemble_weights(
        self,
        ensemble_id: str,
        actual_values: List[float],
        predicted_values: List[Dict[str, float]]
    ) -> bool:
        """
        Update ensemble weights based on recent performance
        
        Args:
            ensemble_id: Ensemble identifier
            actual_values: List of actual target values
            predicted_values: List of prediction dictionaries {model_id: prediction}
            
        Returns:
            True if successful
        """
        
        if ensemble_id not in self.ensembles:
            raise ValueError(f"Ensemble {ensemble_id} not found")
        
        try:
            performance_tracker = self.performance_trackers[ensemble_id]
            
            # Add performance data
            for actual, pred_dict in zip(actual_values, predicted_values):
                # Calculate ensemble prediction (current method)
                config = self.ensembles[ensemble_id]['config']
                individual_results = {
                    model_id: type('Result', (), {'prediction_value': pred})()
                    for model_id, pred in pred_dict.items()
                }
                
                ensemble_pred = self._combine_predictions(
                    individual_results, config, ensemble_id
                ).prediction_value
                
                performance_tracker.add_prediction(ensemble_pred, pred_dict, actual)
            
            # Update weights for dynamic weighting methods
            if self.ensembles[ensemble_id]['config'].method == EnsembleMethod.DYNAMIC_WEIGHTING:
                new_weights = performance_tracker.get_model_weights()
                if new_weights:
                    # Update model order and weights
                    model_ids = list(new_weights.keys())
                    weights = list(new_weights.values())
                    
                    self.ensembles[ensemble_id]['config'].model_ids = model_ids
                    self.ensembles[ensemble_id]['config'].weights = weights
                    
                    logger.info(f"Updated weights for ensemble {ensemble_id}: {new_weights}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")
            return False
    
    def _combine_predictions(
        self,
        individual_results: Dict[str, PredictionResult],
        config: EnsembleConfig,
        ensemble_id: str
    ) -> PredictionResult:
        """Combine individual predictions based on ensemble method"""
        
        if not individual_results:
            raise ValueError("No individual results to combine")
        
        # Extract values and metadata
        predictions = [result.prediction_value for result in individual_results.values()]
        confidences = [result.confidence_score for result in individual_results.values() if result.confidence_score is not None]
        probabilities_list = [result.probabilities for result in individual_results.values() if result.probabilities is not None]
        
        # Use first result as template
        template = list(individual_results.values())[0]
        
        try:
            if config.method == EnsembleMethod.SIMPLE_AVERAGE:
                if all(isinstance(p, (int, float)) for p in predictions):
                    combined_value = np.mean(predictions)
                else:
                    # Majority voting for categorical
                    from collections import Counter
                    combined_value = Counter(predictions).most_common(1)[0][0]
            
            elif config.method == EnsembleMethod.WEIGHTED_AVERAGE:
                weights = config.weights or [1.0 / len(predictions)] * len(predictions)
                
                if all(isinstance(p, (int, float)) for p in predictions):
                    combined_value = np.average(predictions, weights=weights)
                else:
                    # Weighted voting for categorical
                    vote_counts = {}
                    for pred, weight in zip(predictions, weights):
                        vote_counts[pred] = vote_counts.get(pred, 0) + weight
                    combined_value = max(vote_counts, key=vote_counts.get)
            
            elif config.method == EnsembleMethod.MAJORITY_VOTING:
                from collections import Counter
                combined_value = Counter(predictions).most_common(1)[0][0]
            
            elif config.method == EnsembleMethod.CONFIDENCE_WEIGHTED:
                if confidences and len(confidences) == len(predictions):
                    # Weight by confidence scores
                    if all(isinstance(p, (int, float)) for p in predictions):
                        combined_value = np.average(predictions, weights=confidences)
                    else:
                        # Confidence-weighted voting
                        vote_counts = {}
                        for pred, conf in zip(predictions, confidences):
                            vote_counts[pred] = vote_counts.get(pred, 0) + conf
                        combined_value = max(vote_counts, key=vote_counts.get)
                else:
                    # Fall back to simple average
                    combined_value = np.mean(predictions) if all(isinstance(p, (int, float)) for p in predictions) else predictions[0]
            
            elif config.method == EnsembleMethod.DYNAMIC_WEIGHTING:
                # Use performance-based weights
                performance_tracker = self.performance_trackers.get(ensemble_id)
                if performance_tracker:
                    dynamic_weights = performance_tracker.get_model_weights()
                    if dynamic_weights:
                        model_ids = list(individual_results.keys())
                        weights = [dynamic_weights.get(mid, 1.0/len(model_ids)) for mid in model_ids]
                        
                        if all(isinstance(p, (int, float)) for p in predictions):
                            combined_value = np.average(predictions, weights=weights)
                        else:
                            # Dynamic weighted voting
                            vote_counts = {}
                            for pred, weight in zip(predictions, weights):
                                vote_counts[pred] = vote_counts.get(pred, 0) + weight
                            combined_value = max(vote_counts, key=vote_counts.get)
                    else:
                        combined_value = np.mean(predictions) if all(isinstance(p, (int, float)) for p in predictions) else predictions[0]
                else:
                    combined_value = np.mean(predictions) if all(isinstance(p, (int, float)) for p in predictions) else predictions[0]
            
            elif config.method == EnsembleMethod.STACKING:
                # Use trained meta-learner
                if ensemble_id in self.meta_learners:
                    meta_learner = self.meta_learners[ensemble_id]
                    meta_features = np.array(predictions).reshape(1, -1)
                    combined_value = meta_learner.predict(meta_features)[0]
                else:
                    logger.warning("No trained meta-learner found, using simple average")
                    combined_value = np.mean(predictions) if all(isinstance(p, (int, float)) for p in predictions) else predictions[0]
            
            else:
                # Default to simple average
                combined_value = np.mean(predictions) if all(isinstance(p, (int, float)) for p in predictions) else predictions[0]
            
            # Combine confidence scores
            combined_confidence = np.mean(confidences) if confidences else None
            
            # Combine probabilities
            combined_probabilities = None
            if probabilities_list and all(len(p) == len(probabilities_list[0]) for p in probabilities_list):
                combined_probabilities = np.mean(probabilities_list, axis=0).tolist()
            
            # Create ensemble result
            ensemble_result = PredictionResult(
                asset_id=template.asset_id,
                timestamp=template.timestamp,
                prediction_value=combined_value,
                model_id=f"ensemble_{ensemble_id}",
                confidence_score=combined_confidence,
                probabilities=combined_probabilities
            )
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            # Return first prediction as fallback
            return template
    
    def _evaluate_stacking_ensemble(
        self,
        ensemble_id: str,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Evaluate stacking ensemble on validation data"""
        
        try:
            # Get base model predictions on validation data
            base_predictions = []
            config = self.ensembles[ensemble_id]['config']
            
            for model_id in config.model_ids:
                model_preds = []
                for _, row in X_val.iterrows():
                    result = self.model_predictor.predict(row.to_dict(), model_id)
                    model_preds.append(result.prediction_value)
                base_predictions.append(model_preds)
            
            # Create meta-features
            meta_X = np.column_stack(base_predictions)
            
            # Get meta-learner predictions
            meta_learner = self.meta_learners[ensemble_id]
            meta_predictions = meta_learner.predict(meta_X)
            
            # Calculate score based on problem type
            if hasattr(meta_learner, 'predict_proba'):
                # Classification
                score = accuracy_score(y_val, meta_predictions)
            else:
                # Regression
                score = -mean_squared_error(y_val, meta_predictions)  # Negative MSE for maximization
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating stacking ensemble: {e}")
            return 0.0
    
    def get_ensemble_summary(self, ensemble_id: str) -> Dict[str, Any]:
        """Get summary information about an ensemble"""
        
        if ensemble_id not in self.ensembles:
            return {}
        
        ensemble_data = self.ensembles[ensemble_id]
        config = ensemble_data['config']
        
        # Get model information
        model_info = []
        for model_id in config.model_ids:
            try:
                performance = self.model_predictor.get_model_performance(model_id)
                model_info.append({
                    'model_id': model_id,
                    'model_type': performance.get('model_type', 'unknown'),
                    'metrics': performance.get('metrics', {})
                })
            except Exception:
                model_info.append({'model_id': model_id, 'model_type': 'unknown'})
        
        # Get performance tracker stats
        performance_tracker = self.performance_trackers.get(ensemble_id)
        performance_stats = {}
        if performance_tracker:
            if performance_tracker.actuals:
                ensemble_preds = [p['ensemble'] for p in performance_tracker.predictions[-len(performance_tracker.actuals):]]
                if all(isinstance(p, (int, float)) for p in ensemble_preds):
                    mse = np.mean([(p - a) ** 2 for p, a in zip(ensemble_preds, performance_tracker.actuals)])
                    performance_stats = {
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'predictions_tracked': len(performance_tracker.actuals)
                    }
        
        summary = {
            'ensemble_id': ensemble_id,
            'method': config.method.value,
            'model_count': len(config.model_ids),
            'models': model_info,
            'weights': config.weights,
            'created_at': ensemble_data['created_at'].isoformat(),
            'prediction_count': ensemble_data['prediction_count'],
            'description': ensemble_data['description'],
            'performance': performance_stats,
            'has_meta_learner': ensemble_id in self.meta_learners
        }
        
        return summary
    
    def list_ensembles(self) -> List[Dict[str, Any]]:
        """List all created ensembles"""
        
        return [
            self.get_ensemble_summary(ensemble_id)
            for ensemble_id in self.ensembles.keys()
        ]
    
    def delete_ensemble(self, ensemble_id: str) -> bool:
        """Delete an ensemble"""
        
        try:
            if ensemble_id in self.ensembles:
                del self.ensembles[ensemble_id]
            
            if ensemble_id in self.performance_trackers:
                del self.performance_trackers[ensemble_id]
            
            if ensemble_id in self.meta_learners:
                del self.meta_learners[ensemble_id]
            
            logger.info(f"Deleted ensemble {ensemble_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting ensemble {ensemble_id}: {e}")
            return False 
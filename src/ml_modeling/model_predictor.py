"""
Model Predictor for real-time ML inference and prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from loguru import logger
import warnings

from .model_store import ModelStore, ModelMetadata


class PredictionResult:
    """Container for prediction results"""
    
    def __init__(
        self,
        asset_id: str,
        timestamp: datetime,
        prediction_value: Union[float, int, str],
        model_id: str,
        confidence_score: float = None,
        probabilities: List[float] = None,
        features_used: Dict[str, float] = None
    ):
        self.asset_id = asset_id
        self.timestamp = timestamp
        self.prediction_value = prediction_value
        self.model_id = model_id
        self.confidence_score = confidence_score
        self.probabilities = probabilities
        self.features_used = features_used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'asset_id': self.asset_id,
            'timestamp': self.timestamp.isoformat(),
            'prediction_value': self.prediction_value,
            'model_id': self.model_id,
            'confidence_score': self.confidence_score,
            'probabilities': self.probabilities,
            'features_used': self.features_used
        }


class ModelPredictor:
    """
    Performs real-time ML inference using trained models
    """
    
    def __init__(self, model_store: Optional[ModelStore] = None):
        """
        Initialize the model predictor
        
        Args:
            model_store: ModelStore instance for loading models
        """
        self.model_store = model_store or ModelStore()
        self.loaded_models = {}  # Cache for loaded models
        self.prediction_history = []
        
    def predict(
        self,
        features: Union[pd.DataFrame, pd.Series, Dict[str, float]],
        model_id: str,
        asset_id: str = None,
        timestamp: datetime = None,
        include_probabilities: bool = True,
        include_confidence: bool = True
    ) -> PredictionResult:
        """
        Make a prediction using a specific model
        
        Args:
            features: Input features for prediction
            model_id: ID of the model to use
            asset_id: Asset identifier
            timestamp: Prediction timestamp
            include_probabilities: Whether to include class probabilities
            include_confidence: Whether to calculate confidence score
            
        Returns:
            PredictionResult object
        """
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        if asset_id is None:
            asset_id = "unknown"
        
        try:
            # Load model if not cached
            if model_id not in self.loaded_models:
                self._load_model(model_id)
            
            model, metadata = self.loaded_models[model_id]
            
            # Prepare features
            X = self._prepare_features(features, metadata)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Get probabilities if available and requested
            probabilities = None
            if include_probabilities and hasattr(model, 'predict_proba'):
                try:
                    prob_array = model.predict_proba(X)[0]
                    probabilities = prob_array.tolist()
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")
            
            # Calculate confidence score
            confidence_score = None
            if include_confidence:
                confidence_score = self._calculate_confidence(
                    model, X, prediction, probabilities
                )
            
            # Create result
            result = PredictionResult(
                asset_id=asset_id,
                timestamp=timestamp,
                prediction_value=prediction,
                model_id=model_id,
                confidence_score=confidence_score,
                probabilities=probabilities,
                features_used=self._extract_feature_values(features, metadata)
            )
            
            # Store in history
            self.prediction_history.append(result)
            
            logger.debug(f"Prediction for {asset_id}: {prediction} (confidence: {confidence_score})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_id}: {e}")
            raise
    
    def predict_batch(
        self,
        features_batch: List[Union[pd.DataFrame, pd.Series, Dict[str, float]]],
        model_id: str,
        asset_ids: List[str] = None,
        timestamps: List[datetime] = None,
        include_probabilities: bool = True,
        include_confidence: bool = True
    ) -> List[PredictionResult]:
        """
        Make predictions for a batch of feature sets
        
        Args:
            features_batch: List of feature sets
            model_id: ID of the model to use
            asset_ids: List of asset identifiers
            timestamps: List of prediction timestamps
            include_probabilities: Whether to include class probabilities
            include_confidence: Whether to calculate confidence scores
            
        Returns:
            List of PredictionResult objects
        """
        
        batch_size = len(features_batch)
        
        # Prepare defaults
        if asset_ids is None:
            asset_ids = [f"asset_{i}" for i in range(batch_size)]
        
        if timestamps is None:
            base_time = datetime.utcnow()
            timestamps = [base_time for _ in range(batch_size)]
        
        if len(asset_ids) != batch_size:
            raise ValueError("Number of asset_ids must match batch size")
        
        if len(timestamps) != batch_size:
            raise ValueError("Number of timestamps must match batch size")
        
        try:
            # Load model if not cached
            if model_id not in self.loaded_models:
                self._load_model(model_id)
            
            model, metadata = self.loaded_models[model_id]
            
            # Prepare all features
            X_batch = []
            for features in features_batch:
                X = self._prepare_features(features, metadata)
                X_batch.append(X[0])  # Extract single row
            
            X_batch = np.array(X_batch)
            
            # Make batch predictions
            predictions = model.predict(X_batch)
            
            # Get probabilities if available
            probabilities_batch = None
            if include_probabilities and hasattr(model, 'predict_proba'):
                try:
                    probabilities_batch = model.predict_proba(X_batch)
                except Exception as e:
                    logger.warning(f"Could not get batch probabilities: {e}")
            
            # Create results
            results = []
            for i in range(batch_size):
                # Extract probabilities for this prediction
                probabilities = None
                if probabilities_batch is not None:
                    probabilities = probabilities_batch[i].tolist()
                
                # Calculate confidence
                confidence_score = None
                if include_confidence:
                    X_single = X_batch[i:i+1]
                    confidence_score = self._calculate_confidence(
                        model, X_single, predictions[i], probabilities
                    )
                
                # Create result
                result = PredictionResult(
                    asset_id=asset_ids[i],
                    timestamp=timestamps[i],
                    prediction_value=predictions[i],
                    model_id=model_id,
                    confidence_score=confidence_score,
                    probabilities=probabilities,
                    features_used=self._extract_feature_values(features_batch[i], metadata)
                )
                
                results.append(result)
            
            # Store in history
            self.prediction_history.extend(results)
            
            logger.info(f"Made {batch_size} batch predictions with model {model_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions with model {model_id}: {e}")
            raise
    
    def predict_multiple_models(
        self,
        features: Union[pd.DataFrame, pd.Series, Dict[str, float]],
        model_ids: List[str],
        asset_id: str = None,
        timestamp: datetime = None,
        combine_predictions: bool = False,
        combination_method: str = "average"
    ) -> Union[List[PredictionResult], PredictionResult]:
        """
        Make predictions using multiple models
        
        Args:
            features: Input features
            model_ids: List of model IDs to use
            asset_id: Asset identifier
            timestamp: Prediction timestamp
            combine_predictions: Whether to combine predictions into single result
            combination_method: Method for combining ("average", "median", "voting")
            
        Returns:
            List of PredictionResult objects or single combined result
        """
        
        results = []
        
        for model_id in model_ids:
            try:
                result = self.predict(
                    features, model_id, asset_id, timestamp,
                    include_probabilities=True, include_confidence=True
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error with model {model_id}: {e}")
                continue
        
        if not results:
            raise ValueError("No successful predictions from any model")
        
        if not combine_predictions:
            return results
        
        # Combine predictions
        return self._combine_predictions(results, combination_method)
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a loaded model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with performance information
        """
        
        if model_id not in self.loaded_models:
            try:
                self._load_model(model_id)
            except Exception as e:
                logger.error(f"Could not load model {model_id}: {e}")
                return {}
        
        _, metadata = self.loaded_models[model_id]
        
        # Extract metrics from metadata
        performance = {
            'model_id': model_id,
            'model_type': metadata.model_type,
            'target_type': metadata.target_type,
            'created_at': metadata.created_at.isoformat(),
            'metrics': getattr(metadata, 'metrics', {}),
            'feature_count': len(getattr(metadata, 'feature_names', [])),
            'training_data_info': getattr(metadata, 'training_data_info', {})
        }
        
        return performance
    
    def _load_model(self, model_id: str) -> None:
        """Load model into cache"""
        
        try:
            model, metadata = self.model_store.load_model(model_id)
            self.loaded_models[model_id] = (model, metadata)
            logger.info(f"Loaded model {model_id} into cache")
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def _prepare_features(
        self,
        features: Union[pd.DataFrame, pd.Series, Dict[str, float]],
        metadata: ModelMetadata
    ) -> np.ndarray:
        """Prepare features for prediction"""
        
        expected_features = getattr(metadata, 'feature_names', [])
        
        # Convert input to DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            df = features.to_frame().T
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")
        
        # Ensure we have expected features
        if expected_features:
            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with zeros
                for feature in missing_features:
                    df[feature] = 0.0
            
            # Select and order features
            df = df[expected_features]
        
        # Handle missing values
        df = df.fillna(0)
        
        return df.values
    
    def _calculate_confidence(
        self,
        model: Any,
        X: np.ndarray,
        prediction: Union[float, int],
        probabilities: List[float] = None
    ) -> float:
        """Calculate confidence score for prediction"""
        
        try:
            if probabilities:
                # For classification: use max probability as confidence
                confidence = max(probabilities)
            elif hasattr(model, 'predict_proba'):
                # Try to get probabilities
                try:
                    probs = model.predict_proba(X)[0]
                    confidence = max(probs)
                except Exception:
                    confidence = 0.5  # Default for classification
            else:
                # For regression: use a simple confidence metric
                # This is a placeholder - you might want to implement
                # more sophisticated confidence estimation
                confidence = 0.7  # Default confidence
            
            return float(confidence)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence
    
    def _extract_feature_values(
        self,
        features: Union[pd.DataFrame, pd.Series, Dict[str, float]],
        metadata: ModelMetadata
    ) -> Dict[str, float]:
        """Extract feature values for logging/analysis"""
        
        try:
            if isinstance(features, dict):
                return features
            elif isinstance(features, pd.Series):
                return features.to_dict()
            elif isinstance(features, pd.DataFrame):
                return features.iloc[0].to_dict()
            else:
                return {}
        except Exception:
            return {}
    
    def _combine_predictions(
        self,
        results: List[PredictionResult],
        method: str = "average"
    ) -> PredictionResult:
        """Combine multiple prediction results"""
        
        if not results:
            raise ValueError("No results to combine")
        
        if len(results) == 1:
            return results[0]
        
        # Use first result as template
        template = results[0]
        
        try:
            if method == "average":
                # Average numeric predictions
                values = [r.prediction_value for r in results]
                if all(isinstance(v, (int, float)) for v in values):
                    combined_value = np.mean(values)
                else:
                    # For categorical predictions, use majority vote
                    from collections import Counter
                    combined_value = Counter(values).most_common(1)[0][0]
            
            elif method == "median":
                values = [r.prediction_value for r in results if isinstance(r.prediction_value, (int, float))]
                combined_value = np.median(values) if values else template.prediction_value
            
            elif method == "voting":
                values = [r.prediction_value for r in results]
                from collections import Counter
                combined_value = Counter(values).most_common(1)[0][0]
            
            else:
                logger.warning(f"Unknown combination method: {method}, using average")
                combined_value = template.prediction_value
            
            # Combine confidence scores
            confidences = [r.confidence_score for r in results if r.confidence_score is not None]
            combined_confidence = np.mean(confidences) if confidences else None
            
            # Combine probabilities if available
            combined_probabilities = None
            prob_lists = [r.probabilities for r in results if r.probabilities is not None]
            if prob_lists and all(len(p) == len(prob_lists[0]) for p in prob_lists):
                combined_probabilities = np.mean(prob_lists, axis=0).tolist()
            
            # Create combined result
            combined_result = PredictionResult(
                asset_id=template.asset_id,
                timestamp=template.timestamp,
                prediction_value=combined_value,
                model_id=f"ensemble_{len(results)}_models",
                confidence_score=combined_confidence,
                probabilities=combined_probabilities
            )
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return template  # Return first result as fallback
    
    def clear_cache(self) -> None:
        """Clear model cache"""
        self.loaded_models.clear()
        logger.info("Model cache cleared")
    
    def get_prediction_history(
        self,
        asset_id: str = None,
        model_id: str = None,
        limit: int = 100
    ) -> List[PredictionResult]:
        """Get prediction history with optional filtering"""
        
        history = self.prediction_history
        
        # Apply filters
        if asset_id:
            history = [r for r in history if r.asset_id == asset_id]
        
        if model_id:
            history = [r for r in history if r.model_id == model_id]
        
        # Return most recent predictions
        return history[-limit:]
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary statistics about predictions made"""
        
        if not self.prediction_history:
            return {'total_predictions': 0}
        
        # Count by model
        model_counts = {}
        asset_counts = {}
        
        for result in self.prediction_history:
            model_id = result.model_id
            asset_id = result.asset_id
            
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
            asset_counts[asset_id] = asset_counts.get(asset_id, 0) + 1
        
        # Calculate average confidence
        confidences = [r.confidence_score for r in self.prediction_history if r.confidence_score is not None]
        avg_confidence = np.mean(confidences) if confidences else None
        
        return {
            'total_predictions': len(self.prediction_history),
            'unique_models_used': len(model_counts),
            'unique_assets': len(asset_counts),
            'model_usage': model_counts,
            'asset_usage': asset_counts,
            'average_confidence': avg_confidence,
            'loaded_models': list(self.loaded_models.keys())
        } 
"""
Model Trainer for training ML models with cross-validation and hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from loguru import logger
import warnings
from dataclasses import dataclass

# ML imports
try:
    from sklearn.model_selection import (
        train_test_split, cross_val_score, GridSearchCV, 
        RandomizedSearchCV, TimeSeriesSplit
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Limited model training capabilities.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available.")

from .model_store import ModelStore
from .target_builder import TargetType


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: str
    target_type: str
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    use_time_series_split: bool = True
    scale_features: bool = True
    feature_selection: bool = False
    hyperparameter_tuning: bool = True
    hyperparameter_search: str = "grid"  # "grid", "random"
    max_iterations: int = 100
    early_stopping: bool = True
    class_weight: Optional[str] = None  # "balanced" for imbalanced datasets


class ModelTrainer:
    """
    Trains ML models with cross-validation and hyperparameter tuning
    """
    
    def __init__(self, model_store: Optional[ModelStore] = None):
        """
        Initialize the model trainer
        
        Args:
            model_store: ModelStore instance for saving trained models
        """
        self.model_store = model_store or ModelStore()
        self.trained_models = {}
        self.training_history = []
        
        # Ensure required libraries are available
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for model training")
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: TrainingConfig,
        model_id: str = None,
        feature_names: List[str] = None,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train a model with the given configuration
        
        Args:
            X: Feature matrix
            y: Target variable
            config: Training configuration
            model_id: Unique identifier for the model
            feature_names: List of feature names
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        
        if X.empty or y.empty:
            raise ValueError("Training data cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("Feature matrix and target must have same length")
        
        logger.info(f"Training {config.model_type} model for {config.target_type} target")
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Generate model ID if not provided
        if model_id is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_id = f"{config.model_type}_{config.target_type}_{timestamp}"
        
        # Store feature names
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        
        training_start_time = datetime.utcnow()
        
        try:
            # Preprocess data
            X_processed, y_processed, preprocessors = self._preprocess_data(X, y, config)
            
            # Split data
            train_test_data = self._split_data(X_processed, y_processed, config)
            X_train, X_val, X_test, y_train, y_val, y_test = train_test_data
            
            # Get model and hyperparameters
            model_class, param_grid = self._get_model_and_params(config)
            
            # Train model
            best_model, cv_results = self._train_with_cross_validation(
                X_train, y_train, model_class, param_grid, config
            )
            
            # Evaluate model
            train_metrics = self._evaluate_model(best_model, X_train, y_train, config.target_type)
            val_metrics = self._evaluate_model(best_model, X_val, y_val, config.target_type)
            test_metrics = self._evaluate_model(best_model, X_test, y_test, config.target_type)
            
            # Create final pipeline with preprocessing
            final_pipeline = self._create_pipeline(best_model, preprocessors, config)
            
            # Compile results
            training_results = {
                'model_id': model_id,
                'model_type': config.model_type,
                'target_type': config.target_type,
                'model': final_pipeline,
                'best_params': getattr(best_model, 'best_params_', {}),
                'feature_names': feature_names,
                'metrics': {
                    'train': train_metrics,
                    'validation': val_metrics,
                    'test': test_metrics,
                    'cross_validation': cv_results
                },
                'data_info': {
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'n_train': len(X_train),
                    'n_val': len(X_val),
                    'n_test': len(X_test),
                    'target_distribution': y.value_counts().to_dict() if y.dtype == 'object' else None
                },
                'config': config,
                'training_time': (datetime.utcnow() - training_start_time).total_seconds(),
                'created_at': training_start_time
            }
            
            # Save model if requested
            if save_model:
                self._save_trained_model(training_results)
            
            # Store in memory
            self.trained_models[model_id] = training_results
            self.training_history.append(training_results)
            
            logger.info(f"Model {model_id} trained successfully")
            logger.info(f"Test metrics: {test_metrics}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            raise
    
    def train_multiple_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_configs: List[TrainingConfig],
        experiment_name: str = None,
        feature_names: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models with different configurations
        
        Args:
            X: Feature matrix
            y: Target variable
            model_configs: List of training configurations
            experiment_name: Name for grouping related models
            feature_names: List of feature names
            
        Returns:
            Dictionary of training results by model_id
        """
        
        logger.info(f"Training {len(model_configs)} models")
        
        # Create experiment if name provided
        experiment_id = None
        if experiment_name and self.model_store:
            experiment_config = {
                'data_shape': X.shape,
                'target_type': model_configs[0].target_type if model_configs else None,
                'model_types': [config.model_type for config in model_configs]
            }
            experiment_id = self.model_store.create_model_experiment(
                experiment_name, experiment_config
            )
        
        results = {}
        
        for i, config in enumerate(model_configs):
            try:
                # Generate unique model ID
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                model_id = f"{config.model_type}_{config.target_type}_{timestamp}_{i:02d}"
                
                # Train model
                training_result = self.train_model(
                    X, y, config, model_id, feature_names, save_model=True
                )
                
                results[model_id] = training_result
                
                # Add to experiment
                if experiment_id and self.model_store:
                    test_metrics = training_result['metrics']['test']
                    self.model_store.add_model_to_experiment(
                        experiment_id, model_id, test_metrics
                    )
                
                logger.info(f"Completed model {i+1}/{len(model_configs)}: {model_id}")
                
            except Exception as e:
                logger.error(f"Error training model {i+1}: {e}")
                continue
        
        logger.info(f"Trained {len(results)} models successfully")
        return results
    
    def _preprocess_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: TrainingConfig
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess features and target"""
        
        preprocessors = {}
        
        # Handle missing values
        X_clean = X.fillna(X.median(numeric_only=True))
        
        # Remove constant features
        constant_features = X_clean.columns[X_clean.nunique() <= 1]
        if len(constant_features) > 0:
            logger.warning(f"Removing {len(constant_features)} constant features")
            X_clean = X_clean.drop(columns=constant_features)
        
        # Scale features if requested
        if config.scale_features:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_clean)
            preprocessors['scaler'] = scaler
        else:
            X_scaled = X_clean.values
        
        # Handle target variable
        y_clean = y.dropna()
        
        # Align X and y after cleaning
        common_index = X_clean.index.intersection(y_clean.index)
        X_final = X_scaled[X_clean.index.isin(common_index)]
        y_final = y_clean[common_index].values
        
        return X_final, y_final, preprocessors
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets"""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y if len(np.unique(y)) > 1 and len(np.unique(y)) < 50 else None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = config.validation_size / (1 - config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=config.random_state,
            stratify=y_temp if len(np.unique(y_temp)) > 1 and len(np.unique(y_temp)) < 50 else None
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _get_model_and_params(self, config: TrainingConfig) -> Tuple[Any, Dict[str, Any]]:
        """Get model class and hyperparameter grid"""
        
        model_type = config.model_type.lower()
        target_type = config.target_type
        
        # Determine if classification or regression
        is_classification = target_type in ['binary_return', 'multi_class_return', 'binary_breakout']
        
        if model_type == 'random_forest':
            if is_classification:
                model_class = RandomForestClassifier(
                    random_state=config.random_state,
                    class_weight=config.class_weight
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:
                model_class = RandomForestRegressor(
                    random_state=config.random_state
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            if is_classification:
                model_class = xgb.XGBClassifier(
                    random_state=config.random_state,
                    eval_metric='logloss'
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:
                model_class = xgb.XGBRegressor(
                    random_state=config.random_state
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
        
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            if is_classification:
                model_class = lgb.LGBMClassifier(
                    random_state=config.random_state,
                    class_weight=config.class_weight
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            else:
                model_class = lgb.LGBMRegressor(
                    random_state=config.random_state
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
        
        elif model_type == 'logistic_regression' and is_classification:
            model_class = LogisticRegression(
                random_state=config.random_state,
                class_weight=config.class_weight,
                max_iter=config.max_iterations
            )
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        
        elif model_type in ['linear_regression', 'ridge'] and not is_classification:
            if model_type == 'ridge':
                model_class = Ridge(random_state=config.random_state)
                param_grid = {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            else:
                model_class = LinearRegression()
                param_grid = {}
        
        else:
            # Fallback to random forest
            logger.warning(f"Model type {model_type} not recognized, using Random Forest")
            if is_classification:
                model_class = RandomForestClassifier(random_state=config.random_state)
                param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
            else:
                model_class = RandomForestRegressor(random_state=config.random_state)
                param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        
        return model_class, param_grid
    
    def _train_with_cross_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_class: Any,
        param_grid: Dict[str, Any],
        config: TrainingConfig
    ) -> Tuple[Any, Dict[str, float]]:
        """Train model with cross-validation and hyperparameter tuning"""
        
        # Set up cross-validation
        if config.use_time_series_split:
            cv = TimeSeriesSplit(n_splits=config.cross_validation_folds)
        else:
            cv = config.cross_validation_folds
        
        # Determine scoring metric
        is_classification = config.target_type in ['binary_return', 'multi_class_return', 'binary_breakout']
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
        
        if config.hyperparameter_tuning and param_grid:
            # Hyperparameter search
            if config.hyperparameter_search == 'random':
                search = RandomizedSearchCV(
                    model_class,
                    param_grid,
                    n_iter=20,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=config.random_state
                )
            else:
                search = GridSearchCV(
                    model_class,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
            
            # Fit with hyperparameter search
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # Cross-validation results
            cv_results = {
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'cv_scores': search.cv_results_['mean_test_score'].tolist()
            }
            
        else:
            # No hyperparameter tuning
            model_class.fit(X_train, y_train)
            best_model = model_class
            
            # Simple cross-validation
            cv_scores = cross_val_score(model_class, X_train, y_train, cv=cv, scoring=scoring)
            cv_results = {
                'best_score': cv_scores.mean(),
                'cv_scores': cv_scores.tolist(),
                'cv_std': cv_scores.std()
            }
        
        return best_model, cv_results
    
    def _evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        target_type: str
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        
        try:
            predictions = model.predict(X)
            
            is_classification = target_type in ['binary_return', 'multi_class_return', 'binary_breakout']
            
            if is_classification:
                # Classification metrics
                metrics = {
                    'accuracy': accuracy_score(y, predictions),
                    'precision': precision_score(y, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(y, predictions, average='weighted', zero_division=0),
                    'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
                }
                
                # Add probability scores if available
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X)
                        if probabilities.shape[1] == 2:  # Binary classification
                            from sklearn.metrics import roc_auc_score
                            metrics['auc'] = roc_auc_score(y, probabilities[:, 1])
                    except Exception:
                        pass
            
            else:
                # Regression metrics
                metrics = {
                    'mse': mean_squared_error(y, predictions),
                    'rmse': np.sqrt(mean_squared_error(y, predictions)),
                    'mae': mean_absolute_error(y, predictions),
                    'r2': r2_score(y, predictions)
                }
                
                # Additional regression metrics
                residuals = y - predictions
                metrics['mean_residual'] = np.mean(residuals)
                metrics['std_residual'] = np.std(residuals)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def _create_pipeline(
        self,
        model: Any,
        preprocessors: Dict[str, Any],
        config: TrainingConfig
    ) -> Pipeline:
        """Create sklearn pipeline with preprocessing and model"""
        
        steps = []
        
        # Add preprocessing steps
        if 'scaler' in preprocessors:
            steps.append(('scaler', preprocessors['scaler']))
        
        # Add model
        steps.append(('model', model))
        
        return Pipeline(steps)
    
    def _save_trained_model(self, training_results: Dict[str, Any]) -> None:
        """Save trained model using ModelStore"""
        
        if not self.model_store:
            return
        
        try:
            # Extract relevant information
            model = training_results['model']
            model_id = training_results['model_id']
            model_type = training_results['model_type']
            target_type = training_results['target_type']
            
            # Combine all metrics
            all_metrics = {}
            for split, metrics in training_results['metrics'].items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        all_metrics[f"{split}_{metric}"] = value
            
            # Save to model store
            self.model_store.save_model(
                model=model,
                model_id=model_id,
                model_type=model_type,
                target_type=target_type,
                metrics=all_metrics,
                feature_names=training_results['feature_names'],
                hyperparameters=training_results['best_params'],
                training_data_info=training_results['data_info'],
                description=f"Trained {model_type} for {target_type}"
            )
            
        except Exception as e:
            logger.error(f"Error saving model to store: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training sessions"""
        
        if not self.training_history:
            return {'total_models': 0}
        
        model_types = {}
        target_types = {}
        total_training_time = 0
        
        for result in self.training_history:
            # Count model types
            model_type = result['model_type']
            model_types[model_type] = model_types.get(model_type, 0) + 1
            
            # Count target types  
            target_type = result['target_type']
            target_types[target_type] = target_types.get(target_type, 0) + 1
            
            # Sum training time
            total_training_time += result.get('training_time', 0)
        
        return {
            'total_models': len(self.training_history),
            'model_types': model_types,
            'target_types': target_types,
            'total_training_time_seconds': total_training_time,
            'average_training_time_seconds': total_training_time / len(self.training_history),
            'latest_model': self.training_history[-1]['model_id']
        } 
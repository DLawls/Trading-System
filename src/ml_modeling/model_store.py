"""
Model Store for managing trained ML models with versioning and metadata
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import hashlib
import shutil
from pathlib import Path

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("Joblib not available. Using pickle for model serialization.")


class ModelMetadata:
    """Container for model metadata"""
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        target_type: str,
        created_at: datetime,
        model_version: str = "1.0",
        description: str = "",
        **kwargs
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.target_type = target_type
        self.created_at = created_at
        self.model_version = model_version
        self.description = description
        
        # Additional metadata
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (np.integer, np.floating)):
                result[key] = float(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary"""
        # Convert datetime strings back
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


class ModelStore:
    """
    Manages trained ML models with versioning and metadata
    """
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize the model store
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.models_dir = self.base_dir / "models"
        self.metadata_dir = self.base_dir / "metadata"
        self.experiments_dir = self.base_dir / "experiments"
        
        for dir_path in [self.models_dir, self.metadata_dir, self.experiments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelStore initialized at {self.base_dir}")
    
    def save_model(
        self,
        model: Any,
        model_id: str,
        model_type: str,
        target_type: str,
        metrics: Dict[str, float] = None,
        feature_names: List[str] = None,
        hyperparameters: Dict[str, Any] = None,
        training_data_info: Dict[str, Any] = None,
        description: str = "",
        tags: List[str] = None,
        model_version: str = "1.0"
    ) -> str:
        """
        Save a trained model with metadata
        
        Args:
            model: Trained model object
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'xgboost', 'random_forest')
            target_type: Type of target variable
            metrics: Performance metrics
            feature_names: List of feature names used in training
            hyperparameters: Model hyperparameters
            training_data_info: Information about training data
            description: Model description
            tags: List of tags for categorization
            model_version: Model version
            
        Returns:
            Full model path
        """
        
        # Generate unique filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # Save model using joblib if available, otherwise pickle
        try:
            if JOBLIB_AVAILABLE:
                joblib.dump(model, model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            target_type=target_type,
            created_at=datetime.utcnow(),
            model_version=model_version,
            description=description,
            model_path=str(model_path),
            model_filename=model_filename,
            metrics=metrics or {},
            feature_names=feature_names or [],
            hyperparameters=hyperparameters or {},
            training_data_info=training_data_info or {},
            tags=tags or [],
            file_size=os.path.getsize(model_path)
        )
        
        # Save metadata
        self._save_metadata(metadata)
        
        logger.info(f"Model {model_id} saved successfully")
        return str(model_path)
    
    def load_model(self, model_id: str, version: str = "latest") -> Tuple[Any, ModelMetadata]:
        """
        Load a model by ID and version
        
        Args:
            model_id: Model identifier
            version: Model version or "latest"
            
        Returns:
            Tuple of (model object, metadata)
        """
        
        # Find the model
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model {model_id} (version {version}) not found")
        
        model_path = Path(metadata.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        try:
            if JOBLIB_AVAILABLE:
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"Model {model_id} loaded successfully")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_metadata(self, model_id: str, version: str = "latest") -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model
        
        Args:
            model_id: Model identifier
            version: Model version or "latest"
            
        Returns:
            ModelMetadata object or None if not found
        """
        
        # List all metadata files for this model
        metadata_files = list(self.metadata_dir.glob(f"{model_id}_*.json"))
        
        if not metadata_files:
            return None
        
        if version == "latest":
            # Sort by creation time and get the latest
            metadata_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            metadata_file = metadata_files[0]
        else:
            # Find specific version
            metadata_file = None
            for file_path in metadata_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if data.get('model_version') == version:
                        metadata_file = file_path
                        break
                except Exception:
                    continue
            
            if not metadata_file:
                return None
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None
    
    def list_models(
        self,
        model_type: str = None,
        target_type: str = None,
        tags: List[str] = None
    ) -> List[ModelMetadata]:
        """
        List all models with optional filtering
        
        Args:
            model_type: Filter by model type
            target_type: Filter by target type
            tags: Filter by tags (must have all tags)
            
        Returns:
            List of ModelMetadata objects
        """
        
        metadata_files = list(self.metadata_dir.glob("*.json"))
        models = []
        
        for file_path in metadata_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                metadata = ModelMetadata.from_dict(data)
                
                # Apply filters
                if model_type and metadata.model_type != model_type:
                    continue
                
                if target_type and metadata.target_type != target_type:
                    continue
                
                if tags:
                    model_tags = getattr(metadata, 'tags', [])
                    if not all(tag in model_tags for tag in tags):
                        continue
                
                models.append(metadata)
                
            except Exception as e:
                logger.warning(f"Error loading metadata from {file_path}: {e}")
                continue
        
        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models
    
    def get_best_model(
        self,
        metric_name: str,
        model_type: str = None,
        target_type: str = None,
        higher_is_better: bool = True
    ) -> Optional[Tuple[str, ModelMetadata]]:
        """
        Get the best model based on a metric
        
        Args:
            metric_name: Name of the metric to optimize
            model_type: Filter by model type
            target_type: Filter by target type
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Tuple of (model_id, metadata) or None
        """
        
        models = self.list_models(model_type=model_type, target_type=target_type)
        
        if not models:
            return None
        
        # Filter models that have the metric
        valid_models = []
        for metadata in models:
            metrics = getattr(metadata, 'metrics', {})
            if metric_name in metrics:
                valid_models.append((metadata, metrics[metric_name]))
        
        if not valid_models:
            return None
        
        # Sort by metric
        valid_models.sort(key=lambda x: x[1], reverse=higher_is_better)
        best_metadata, best_score = valid_models[0]
        
        logger.info(f"Best model: {best_metadata.model_id} with {metric_name}={best_score}")
        return best_metadata.model_id, best_metadata
    
    def delete_model(self, model_id: str, version: str = "all") -> bool:
        """
        Delete a model and its metadata
        
        Args:
            model_id: Model identifier
            version: Model version or "all" to delete all versions
            
        Returns:
            True if successful
        """
        
        deleted_count = 0
        
        if version == "all":
            # Delete all versions
            model_files = list(self.models_dir.glob(f"{model_id}_*.pkl"))
            metadata_files = list(self.metadata_dir.glob(f"{model_id}_*.json"))
            
            for file_path in model_files + metadata_files:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        else:
            # Delete specific version
            metadata = self.get_model_metadata(model_id, version)
            if metadata:
                try:
                    # Delete model file
                    model_path = Path(metadata.model_path)
                    if model_path.exists():
                        model_path.unlink()
                        deleted_count += 1
                    
                    # Delete metadata file
                    metadata_files = list(self.metadata_dir.glob(f"{model_id}_*.json"))
                    for metadata_file in metadata_files:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                        if data.get('model_version') == version:
                            metadata_file.unlink()
                            deleted_count += 1
                            break
                            
                except Exception as e:
                    logger.error(f"Error deleting model {model_id} version {version}: {e}")
                    return False
        
        logger.info(f"Deleted {deleted_count} files for model {model_id}")
        return deleted_count > 0
    
    def update_model_metrics(
        self,
        model_id: str,
        new_metrics: Dict[str, float],
        version: str = "latest"
    ) -> bool:
        """
        Update metrics for an existing model
        
        Args:
            model_id: Model identifier
            new_metrics: New metrics to add/update
            version: Model version
            
        Returns:
            True if successful
        """
        
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            logger.error(f"Model {model_id} version {version} not found")
            return False
        
        # Update metrics
        current_metrics = getattr(metadata, 'metrics', {})
        current_metrics.update(new_metrics)
        metadata.metrics = current_metrics
        
        # Save updated metadata
        try:
            self._save_metadata(metadata)
            logger.info(f"Updated metrics for model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            return False
    
    def create_model_experiment(
        self,
        experiment_name: str,
        experiment_config: Dict[str, Any],
        description: str = ""
    ) -> str:
        """
        Create a new experiment for tracking related models
        
        Args:
            experiment_name: Name of the experiment
            experiment_config: Experiment configuration
            description: Experiment description
            
        Returns:
            Experiment ID
        """
        
        experiment_id = f"{experiment_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        experiment_path = self.experiments_dir / f"{experiment_id}.json"
        
        experiment_data = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'created_at': datetime.utcnow().isoformat(),
            'description': description,
            'config': experiment_config,
            'models': []
        }
        
        try:
            with open(experiment_path, 'w') as f:
                json.dump(experiment_data, f, indent=2)
            
            logger.info(f"Created experiment {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def add_model_to_experiment(
        self,
        experiment_id: str,
        model_id: str,
        metrics: Dict[str, float] = None
    ) -> bool:
        """
        Add a model to an experiment
        
        Args:
            experiment_id: Experiment identifier
            model_id: Model identifier
            metrics: Optional metrics for this model in the experiment
            
        Returns:
            True if successful
        """
        
        experiment_path = self.experiments_dir / f"{experiment_id}.json"
        
        if not experiment_path.exists():
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        try:
            with open(experiment_path, 'r') as f:
                experiment_data = json.load(f)
            
            # Add model to experiment
            model_entry = {
                'model_id': model_id,
                'added_at': datetime.utcnow().isoformat(),
                'metrics': metrics or {}
            }
            
            experiment_data['models'].append(model_entry)
            
            with open(experiment_path, 'w') as f:
                json.dump(experiment_data, f, indent=2)
            
            logger.info(f"Added model {model_id} to experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding model to experiment: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary statistics about stored models"""
        
        models = self.list_models()
        
        if not models:
            return {'total_models': 0}
        
        # Count by type
        model_types = {}
        target_types = {}
        total_size = 0
        
        for metadata in models:
            # Model types
            model_type = metadata.model_type
            model_types[model_type] = model_types.get(model_type, 0) + 1
            
            # Target types
            target_type = metadata.target_type
            target_types[target_type] = target_types.get(target_type, 0) + 1
            
            # File size
            file_size = getattr(metadata, 'file_size', 0)
            total_size += file_size
        
        return {
            'total_models': len(models),
            'model_types': model_types,
            'target_types': target_types,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'latest_model': models[0].model_id if models else None,
            'storage_path': str(self.base_dir)
        }
    
    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save metadata to file"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        metadata_filename = f"{metadata.model_id}_{timestamp}.json"
        metadata_path = self.metadata_dir / metadata_filename
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            logger.debug(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def cleanup_old_models(self, keep_latest: int = 5, days_old: int = 30) -> int:
        """
        Clean up old model versions
        
        Args:
            keep_latest: Number of latest models to keep per model_id
            days_old: Delete models older than this many days
            
        Returns:
            Number of models deleted
        """
        
        deleted_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Group models by model_id
        models_by_id = {}
        for metadata in self.list_models():
            model_id = metadata.model_id
            if model_id not in models_by_id:
                models_by_id[model_id] = []
            models_by_id[model_id].append(metadata)
        
        # Clean up each model group
        for model_id, model_list in models_by_id.items():
            # Sort by creation time (newest first)
            model_list.sort(key=lambda m: m.created_at, reverse=True)
            
            # Keep the latest N models
            models_to_check = model_list[keep_latest:]
            
            for metadata in models_to_check:
                # Delete if older than cutoff
                if metadata.created_at < cutoff_date:
                    if self.delete_model(model_id, metadata.model_version):
                        deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old models")
        return deleted_count 
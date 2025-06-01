"""
Test script for ML Modeling components
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_ml_modeling():
    """Test ML modeling components."""
    
    print("🤖 Testing ML Modeling Components...")
    
    # Test 1: Import all components
    print("\n📦 Testing imports...")
    try:
        from src.ml_modeling.target_builder import TargetBuilder, TargetType
        print("   ✅ TargetBuilder imported")
        
        from src.ml_modeling.model_trainer import ModelTrainer, TrainingConfig
        print("   ✅ ModelTrainer imported")
        
        from src.ml_modeling.model_store import ModelStore
        print("   ✅ ModelStore imported")
        
        from src.ml_modeling.model_predictor import ModelPredictor
        print("   ✅ ModelPredictor imported")
        
        from src.ml_modeling.ensemble_manager import EnsembleManager
        print("   ✅ EnsembleManager imported")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Instantiate components
    print("\n🏗️ Testing instantiation...")
    try:
        target_builder = TargetBuilder()
        print("   ✅ TargetBuilder instantiated")
        
        model_store = ModelStore(base_dir="data/test_models")
        print("   ✅ ModelStore instantiated")
        
        model_trainer = ModelTrainer(model_store)
        print("   ✅ ModelTrainer instantiated")
        
        model_predictor = ModelPredictor(model_store)
        print("   ✅ ModelPredictor instantiated")
        
        ensemble_manager = EnsembleManager(model_store, model_predictor)
        print("   ✅ EnsembleManager instantiated")
        
    except Exception as e:
        print(f"   ❌ Instantiation failed: {e}")
        return False
    
    # Test 3: Create sample data and targets
    print("\n📊 Testing target creation...")
    try:
        # Create sample market data
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 20),
            'high': np.random.uniform(105, 115, 20),
            'low': np.random.uniform(95, 105, 20),
            'close': np.random.uniform(100, 110, 20),
            'volume': np.random.uniform(1000000, 2000000, 20)
        }, index=dates)
        
        print(f"   📈 Created sample market data: {df.shape}")
        
        # Test target creation
        target_config = {
            'type': 'binary_return',
            'lookforward_periods': 1,
            'threshold': 0.01
        }
        
        df_with_targets, metadata = target_builder.create_targets(df, target_config)
        target_cols = [col for col in df_with_targets.columns if col.startswith('target_')]
        
        print(f"   🎯 Created {len(target_cols)} target features")
        print(f"   📊 Sample targets: {target_cols[:3]}")
        
    except Exception as e:
        print(f"   ❌ Target creation failed: {e}")
        return False
    
    # Test 4: Model Store functionality
    print("\n💾 Testing ModelStore...")
    try:
        # Test model summary
        summary = model_store.get_model_summary()
        print(f"   📋 Model store summary: {summary['total_models']} models")
        storage_path = summary.get('storage_path', model_store.base_dir)
        print(f"   📁 Storage path: {storage_path}")
        
    except Exception as e:
        print(f"   ❌ ModelStore test failed: {e}")
        return False
    
    print("\n✅ ML Modeling Components Test PASSED!")
    print("   📦 All imports successful")
    print("   🏗️ All instantiations successful")
    print("   🎯 Target creation working")
    print("   💾 ModelStore working")
    print("\n🎯 ML Modeling components are ready to use!")
    
    return True


if __name__ == "__main__":
    test_ml_modeling() 
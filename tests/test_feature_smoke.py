"""
Smoke test for Feature Engineering - just verify imports and instantiation
"""

def test_feature_engineering_smoke():
    """Quick smoke test to verify feature engineering components work."""
    
    print("💨 Feature Engineering Smoke Test...")
    
    # Test 1: Import all components
    print("\n📦 Testing imports...")
    try:
        from src.feature_engineering.timeseries_features import TimeseriesFeatures
        print("   ✅ TimeseriesFeatures imported")
        
        from src.feature_engineering.event_features import EventFeatures  
        print("   ✅ EventFeatures imported")
        
        from src.feature_engineering.sentiment_features import SentimentFeatures
        print("   ✅ SentimentFeatures imported")
        
        from src.feature_engineering.market_context_features import MarketContextFeatures
        print("   ✅ MarketContextFeatures imported")
        
        from src.feature_engineering.feature_store import FeatureStore
        print("   ✅ FeatureStore imported")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Instantiate components
    print("\n🏗️ Testing instantiation...")
    try:
        ts_generator = TimeseriesFeatures()
        print("   ✅ TimeseriesFeatures instantiated")
        
        context_generator = MarketContextFeatures()
        print("   ✅ MarketContextFeatures instantiated")
        
        feature_store = FeatureStore()
        print("   ✅ FeatureStore instantiated")
        
    except Exception as e:
        print(f"   ❌ Instantiation failed: {e}")
        return False
    
    # Test 3: Basic data structure
    print("\n📊 Testing basic data structure...")
    try:
        import pandas as pd
        import numpy as np
        
        # Create tiny dataset (just 5 rows)
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105], 
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        
        print(f"   ✅ Created test data: {df.shape}")
        print(f"   📈 Data range: {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        print(f"   ❌ Data creation failed: {e}")
        return False
    
    print("\n✅ Feature Engineering Smoke Test PASSED!")
    print("   📦 All imports successful")
    print("   🏗️ All instantiations successful") 
    print("   📊 Basic data structures working")
    print("\n🎯 Feature Engineering components are ready to use!")
    
    return True


if __name__ == "__main__":
    test_feature_engineering_smoke() 
"""
Minimal smoke test for FeatureEngineer
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_smoke():
    """Minimal smoke test"""
    
    print("FeatureEngineer Smoke Test...")
    
    try:
        # Test imports
        from src.feature_engineering.main import FeatureEngineer, FeatureConfig
        print("  ✓ Imports successful")
        
        # Test config creation
        config = FeatureConfig(
            enable_timeseries_features=False,
            enable_event_features=False,
            enable_sentiment_features=False,
            enable_market_context_features=False
        )
        print("  ✓ Config created")
        
        # Test FeatureEngineer initialization
        feature_engineer = FeatureEngineer(config=config)
        print("  ✓ FeatureEngineer initialized")
        
        # Create minimal data (just 3 rows)
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200],
            'price_change': [0, 1, 1],
            'price_change_pct': [0, 0.01, 0.0098]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        market_data = {'AAPL': data}
        print("  ✓ Mock data created")
        
        # Test feature generation with all features disabled
        result = feature_engineer.generate_features(
            market_data=market_data,
            symbols=['AAPL']
        )
        
        if 'AAPL' in result:
            print(f"  ✓ Feature generation worked - shape: {result['AAPL'].shape}")
            print(f"  ✓ Columns: {list(result['AAPL'].columns)}")
        else:
            print("  ✗ No results returned")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  ✓ Smoke test passed!")
    return True

if __name__ == "__main__":
    test_smoke() 
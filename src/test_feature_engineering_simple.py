"""
Simple, fast test for Feature Engineering pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering.feature_store import FeatureStore
from src.feature_engineering.timeseries_features import TimeseriesFeatures
from src.feature_engineering.event_features import EventFeatures
from src.feature_engineering.sentiment_features import SentimentFeatures
from src.feature_engineering.market_context_features import MarketContextFeatures


def test_feature_engineering_simple():
    """Simple, fast test of feature engineering components."""
    
    print("🔧 Simple Feature Engineering Test...")
    
    # Create minimal sample data (30 rows for speed)
    sample_data = create_minimal_market_data()
    print(f"📈 Created minimal sample data: {len(sample_data)} records")
    
    # Test 1: Basic Timeseries Features
    print("\n🔬 Testing TimeseriesFeatures...")
    ts_generator = TimeseriesFeatures()
    
    # Use minimal window sizes for speed
    ts_features = ts_generator.generate_features(sample_data.copy(), 'TSLA')
    new_features = len(ts_features.columns) - len(sample_data.columns)
    print(f"   ✅ Generated {new_features} timeseries features")
    
    # Test 2: Basic Market Context Features
    print("\n🔬 Testing MarketContextFeatures...")
    context_generator = MarketContextFeatures()
    context_features = context_generator.generate_features(sample_data.copy(), 'TSLA')
    new_features = len(context_features.columns) - len(sample_data.columns)
    print(f"   ✅ Generated {new_features} market context features")
    
    # Test 3: Quick FeatureStore test
    print("\n🔬 Testing FeatureStore...")
    feature_store = FeatureStore()
    
    # Generate features with minimal data
    complete_features = feature_store.generate_complete_features(
        data=sample_data.copy(),
        ticker='TSLA',
        use_cache=False,
        include_events=False,  # Skip events for speed
        include_sentiment=False  # Skip sentiment for speed
    )
    
    total_features = len(complete_features.columns) - len(sample_data.columns)
    print(f"   ✅ Generated {total_features} total features")
    print(f"   📊 Final dataset shape: {complete_features.shape}")
    
    # Test 4: Data Quality Check
    print("\n🔍 Basic Data Quality Check...")
    
    # Check for NaN values
    nan_count = complete_features.isnull().sum().sum()
    print(f"   📊 Total NaN values: {nan_count}")
    
    # Check for infinite values
    inf_count = np.isinf(complete_features.select_dtypes(include=[np.number])).sum().sum()
    print(f"   📊 Total infinite values: {inf_count}")
    
    # Show sample feature names
    feature_cols = [col for col in complete_features.columns if col not in sample_data.columns]
    print(f"   🔧 Sample features: {feature_cols[:5]}")
    
    print(f"\n✅ Simple Feature Engineering Test Complete!")
    print(f"   📊 Original columns: {len(sample_data.columns)}")
    print(f"   🔧 New features: {total_features}")
    print(f"   📈 Total columns: {len(complete_features.columns)}")
    
    return complete_features


def create_minimal_market_data() -> pd.DataFrame:
    """Create minimal OHLCV data for fast testing"""
    
    # Use 30 days instead of 10 for better feature generation
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    np.random.seed(42)  # For reproducibility
    base_price = 200.0
    data = []
    
    for i, date in enumerate(dates):
        # Add more realistic price movement
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        current_price = base_price * (1 + price_change)
        
        # More realistic OHLCV with some variation
        high_mult = 1 + abs(np.random.normal(0, 0.01))
        low_mult = 1 - abs(np.random.normal(0, 0.01))
        
        high = current_price * high_mult
        low = current_price * low_mult
        open_price = low + (high - low) * np.random.random()
        close = low + (high - low) * np.random.random()
        volume = int(np.random.uniform(500000, 2000000))
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        base_price = close
    
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    
    # Add simple price change for targets
    df['price_change_pct'] = df['close'].pct_change()
    
    return df


if __name__ == "__main__":
    test_feature_engineering_simple() 
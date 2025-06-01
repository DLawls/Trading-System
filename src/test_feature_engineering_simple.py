"""
Simple and fast test for FeatureEngineer
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.main import FeatureEngineer, FeatureConfig

def create_simple_market_data():
    """Create small amount of market data for fast testing"""
    
    # Just 5 days of daily data (much smaller)
    dates = pd.date_range(start='2023-01-01', periods=5, freq='1D')
    
    market_data = {}
    
    for symbol in ['AAPL']:  # Just one symbol
        # Simple price data
        prices = [100, 102, 98, 101, 105]  # Simple price sequence
        
        data = []
        for i, (timestamp, price) in enumerate(zip(dates, prices)):
            data.append({
                'timestamp': timestamp,
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'volume': 1000000
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Add required columns
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        
        market_data[symbol] = df
    
    return market_data

def test_feature_engineer_simple():
    """Simple test of FeatureEngineer"""
    
    print("Testing FeatureEngineer (Simple)...")
    
    # Create minimal data
    market_data = create_simple_market_data()
    print(f"  Market data: {len(market_data)} symbols, {len(market_data['AAPL'])} records each")
    
    # Test with only timeseries features (fastest)
    config = FeatureConfig(
        enable_timeseries_features=True,
        enable_event_features=False,  # Disable slow features
        enable_sentiment_features=False,
        enable_market_context_features=False,
        rolling_windows=[2, 3],  # Small windows
        max_features=10  # Limit features
    )
    
    # Initialize FeatureEngineer
    feature_engineer = FeatureEngineer(config=config)
    
    # Generate features
    print("  Generating timeseries features...")
    feature_data = feature_engineer.generate_features(
        market_data=market_data,
        symbols=['AAPL']
    )
    
    if 'AAPL' in feature_data:
        df = feature_data['AAPL']
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol']]
        print(f"  Generated {len(feature_cols)} features: {feature_cols[:5]}")
        print(f"  Data shape: {df.shape}")
        
        # Show sample
        print("\n  Sample data:")
        print(df[['close', 'price_change_pct'] + feature_cols[:3]].round(4))
    
    # Test feature summary
    summary = feature_engineer.get_feature_summary()
    print(f"\n  Feature summary:")
    print(f"    Total features: {summary['total_features']}")
    print(f"    Categories: {summary['feature_categories']}")
    
    print("\n  FeatureEngineer simple test completed successfully!")
    return True

if __name__ == "__main__":
    test_feature_engineer_simple() 
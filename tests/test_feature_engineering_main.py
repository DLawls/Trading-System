"""
Test script for the main FeatureEngineer class
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.main import FeatureEngineer, FeatureConfig
from src.event_detection.event_classifier import DetectedEvent, EventType

def create_mock_market_data():
    """Create mock market data for testing"""
    
    # Create 3 months of hourly data
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='1h')
    
    market_data = {}
    
    for symbol in ['AAPL', 'TSLA']:
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 2**32)
        
        base_price = np.random.uniform(100, 300)
        returns = np.random.normal(0.0001, 0.015, len(dates))
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(dates, prices)):
            volatility = 0.02
            high = price * (1 + abs(np.random.normal(0, volatility)))
            low = price * (1 - abs(np.random.normal(0, volatility)))
            open_price = prices[i-1] if i > 0 else price
            
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Add price change columns that feature generators expect
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        
        market_data[symbol] = df
    
    return market_data

def create_mock_news_data():
    """Create mock news data"""
    
    news_data = []
    current_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    
    while current_date <= end_date:
        # 1-2 articles per day
        for _ in range(np.random.randint(1, 3)):
            article_time = current_date + timedelta(
                hours=np.random.randint(9, 17),
                minutes=np.random.randint(0, 60)
            )
            
            sentiment_score = np.random.normal(0, 0.3)  # Random sentiment
            
            news_data.append({
                'timestamp': article_time,
                'title': f"Market news for {article_time.strftime('%Y-%m-%d')}",
                'sentiment_score': sentiment_score,
                'composite_sentiment': sentiment_score,
                'target_symbol': np.random.choice(['AAPL', 'TSLA', None])
            })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(news_data)

def create_mock_events():
    """Create mock events"""
    
    events = []
    current_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    
    while current_date <= end_date:
        if np.random.random() < 0.1:  # 10% chance of event per day
            event = DetectedEvent(
                event_type=np.random.choice(list(EventType)),
                confidence=np.random.uniform(0.6, 0.9),
                entity=np.random.choice(['AAPL', 'TSLA']),
                title=f"Event on {current_date.strftime('%Y-%m-%d')}",
                description="Mock event for testing",
                keywords_matched=['test', 'mock'],
                sentiment='neutral',
                metadata={'timestamp': current_date + timedelta(hours=np.random.randint(9, 17))}
            )
            events.append(event)
        
        current_date += timedelta(days=1)
    
    return events

def test_feature_engineer():
    """Test the FeatureEngineer class"""
    
    print("Testing FeatureEngineer...")
    
    # Create mock data
    print("  Creating mock data...")
    market_data = create_mock_market_data()
    news_data = create_mock_news_data()
    events = create_mock_events()
    
    print(f"    Market data: {len(market_data)} symbols")
    for symbol, df in market_data.items():
        print(f"      {symbol}: {len(df)} records")
    
    print(f"    News data: {len(news_data)} articles")
    print(f"    Events: {len(events)} events")
    
    # Test basic configuration
    print("\n  Testing basic feature generation...")
    config = FeatureConfig(
        enable_timeseries_features=True,
        enable_event_features=True,
        enable_sentiment_features=True,
        enable_market_context_features=False,  # Disable to avoid external API calls
        rolling_windows=[5, 10, 20],
        max_features=50  # Limit features for testing
    )
    
    # Initialize FeatureEngineer
    feature_engineer = FeatureEngineer(config=config)
    
    # Generate features
    print("    Generating features...")
    feature_data = feature_engineer.generate_features(
        market_data=market_data,
        news_data=news_data,
        events=events,
        symbols=['AAPL', 'TSLA']
    )
    
    print(f"    Generated features for {len(feature_data)} symbols")
    
    # Analyze results
    for symbol, df in feature_data.items():
        print(f"\n    {symbol} Features:")
        print(f"      Shape: {df.shape}")
        print(f"      Columns: {len(df.columns)}")
        
        # Show sample features
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol']]
        print(f"      Generated features: {len(feature_cols)}")
        
        if feature_cols:
            print(f"      Sample features: {feature_cols[:10]}")
    
    # Test feature summary
    print("\n  Testing feature summary...")
    summary = feature_engineer.get_feature_summary()
    print(f"    Total features: {summary['total_features']}")
    print(f"    Feature categories: {summary['feature_categories']}")
    print(f"    Symbols processed: {summary['symbols_processed']}")
    
    # Test training dataset creation
    print("\n  Testing training dataset creation...")
    try:
        X, y = feature_engineer.create_training_dataset(
            feature_data=feature_data,
            target_column='price_change_pct',
            lookforward_periods=1,
            min_samples=50
        )
        
        print(f"    Training data shape: X={X.shape}, y={y.shape}")
        print(f"    Features used: {len(X.columns)}")
        
        # Check for any NaN values
        nan_features = X.columns[X.isnull().any()].tolist()
        if nan_features:
            print(f"    Warning: Features with NaN values: {len(nan_features)}")
        else:
            print(f"    All features clean (no NaN values)")
            
    except Exception as e:
        print(f"    Training dataset creation failed: {e}")
    
    # Test feature importance
    print("\n  Testing feature importance...")
    try:
        importance = feature_engineer.get_feature_importance(
            feature_data=feature_data,
            target_column='price_change_pct'
        )
        
        for symbol, scores in importance.items():
            if scores:
                top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"    {symbol} top features:")
                for feature, score in top_features:
                    print(f"      {feature}: {score:.4f}")
                    
    except Exception as e:
        print(f"    Feature importance calculation failed: {e}")
    
    # Test with different configurations
    print("\n  Testing different configurations...")
    
    # Test with only timeseries features
    ts_config = FeatureConfig(
        enable_timeseries_features=True,
        enable_event_features=False,
        enable_sentiment_features=False,
        enable_market_context_features=False,
        rolling_windows=[10, 20]
    )
    
    ts_engineer = FeatureEngineer(config=ts_config)
    ts_features = ts_engineer.generate_features(
        market_data=market_data,
        symbols=['AAPL']
    )
    
    if 'AAPL' in ts_features:
        feature_cols = [col for col in ts_features['AAPL'].columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol']]
        print(f"    Timeseries-only features: {len(feature_cols)}")
    
    print("\n  FeatureEngineer testing completed successfully!")
    return True

if __name__ == "__main__":
    test_feature_engineer() 
"""
Test script for the complete Feature Engineering pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering.feature_store import FeatureStore
from src.feature_engineering.timeseries_features import TimeseriesFeatures
from src.feature_engineering.event_features import EventFeatures
from src.feature_engineering.sentiment_features import SentimentFeatures
from src.feature_engineering.market_context_features import MarketContextFeatures
from src.event_detection.event_store import EventStore


def test_feature_engineering_pipeline():
    """Test the complete feature engineering pipeline."""
    
    print("🔧 Testing Complete Feature Engineering Pipeline...")
    
    # Create sample market data
    sample_data = create_sample_market_data()
    print(f"\n📈 Created sample market data with {len(sample_data)} records")
    
    # Create sample news data for sentiment features
    sample_news = create_sample_news_data()
    print(f"📰 Created sample news data with {len(sample_news)} articles")
    
    # Initialize event store with some sample events
    event_store = EventStore()
    populate_sample_events(event_store)
    print(f"📊 Populated event store with sample events")
    
    # Test 1: Individual Feature Generators
    print("\n" + "="*80)
    print("🧪 TEST 1: Individual Feature Generators")
    print("="*80)
    
    # Test TimeseriesFeatures
    print("\n🔬 Testing TimeseriesFeatures...")
    ts_generator = TimeseriesFeatures()
    ts_features = ts_generator.generate_features(sample_data.copy(), 'TSLA')
    print(f"   ✅ Generated {len(ts_features.columns) - len(sample_data.columns)} timeseries features")
    
    # Show sample features
    timeseries_cols = [col for col in ts_features.columns if col not in sample_data.columns]
    print(f"   📊 Sample features: {timeseries_cols[:5]}")
    
    # Test EventFeatures
    print("\n🔬 Testing EventFeatures...")
    event_generator = EventFeatures(event_store)
    event_features = event_generator.generate_features(sample_data.copy(), 'TSLA')
    print(f"   ✅ Generated {len(event_features.columns) - len(sample_data.columns)} event features")
    
    # Show sample features
    event_cols = [col for col in event_features.columns if col not in sample_data.columns]
    print(f"   📊 Sample features: {event_cols[:5]}")
    
    # Test SentimentFeatures
    print("\n🔬 Testing SentimentFeatures...")
    sentiment_generator = SentimentFeatures(sample_news)
    sentiment_features = sentiment_generator.generate_features(sample_data.copy(), 'TSLA')
    print(f"   ✅ Generated {len(sentiment_features.columns) - len(sample_data.columns)} sentiment features")
    
    # Show sample features
    sentiment_cols = [col for col in sentiment_features.columns if col not in sample_data.columns]
    print(f"   📊 Sample features: {sentiment_cols[:5]}")
    
    # Test MarketContextFeatures
    print("\n🔬 Testing MarketContextFeatures...")
    context_generator = MarketContextFeatures()
    context_features = context_generator.generate_features(sample_data.copy(), 'TSLA')
    print(f"   ✅ Generated {len(context_features.columns) - len(sample_data.columns)} market context features")
    
    # Show sample features
    context_cols = [col for col in context_features.columns if col not in sample_data.columns]
    print(f"   📊 Sample features: {context_cols[:5]}")
    
    # Test 2: Complete FeatureStore Integration
    print("\n" + "="*80)
    print("🧪 TEST 2: Complete FeatureStore Integration")
    print("="*80)
    
    # Initialize FeatureStore
    feature_store = FeatureStore(
        event_store=event_store,
        news_data=sample_news,
        cache_dir="data/test_features"
    )
    
    # Generate complete feature set
    print("\n🏭 Generating complete feature set for TSLA...")
    complete_features = feature_store.generate_complete_features(
        data=sample_data.copy(),
        ticker='TSLA',
        use_cache=False  # Don't use cache for testing
    )
    
    total_features = len(complete_features.columns) - len(sample_data.columns)
    print(f"   ✅ Generated {total_features} total features")
    print(f"   📊 Final dataset shape: {complete_features.shape}")
    
    # Test 3: Feature Summary and Analysis
    print("\n" + "="*80)
    print("🧪 TEST 3: Feature Summary and Analysis")
    print("="*80)
    
    # Generate feature summary
    summary = feature_store.generate_feature_summary(complete_features, 'TSLA')
    print(f"\n📋 Feature Summary for TSLA:")
    print(f"   Total features: {summary['total_features']}")
    print(f"   Total samples: {summary['total_samples']}")
    print(f"   Date range: {summary['date_range']}")
    
    # Show feature breakdown by type
    print(f"\n🔧 Feature breakdown by type:")
    for feature_type, type_summary in summary['feature_types'].items():
        if isinstance(type_summary, dict) and 'total_features' in type_summary:
            print(f"   {feature_type}: {type_summary['total_features']} features")
    
    # Missing value statistics
    print(f"\n❌ Missing value statistics:")
    missing_stats = summary['missing_values']
    print(f"   Total missing: {missing_stats['total_missing']}")
    print(f"   Columns with missing: {missing_stats['columns_with_missing']}")
    print(f"   Missing percentage: {missing_stats['missing_percentage']:.2f}%")
    
    # Test 4: Feature Importance Analysis
    print("\n" + "="*80)
    print("🧪 TEST 4: Feature Importance Analysis")
    print("="*80)
    
    # Get feature importance
    importance_results = feature_store.get_feature_importance(complete_features)
    
    print(f"\n⭐ Feature importance by category:")
    for category, importance_dict in importance_results.items():
        if importance_dict:
            top_features = list(importance_dict.items())[:3]
            print(f"\n   {category.upper()}:")
            for feature, importance in top_features:
                print(f"     • {feature}: {importance:.4f}")
    
    # Test 5: Batch Processing
    print("\n" + "="*80)
    print("🧪 TEST 5: Batch Feature Generation")
    print("="*80)
    
    # Create data for multiple tickers
    tickers = ['TSLA', 'AAPL', 'MSFT']
    data_dict = {}
    
    for ticker in tickers:
        # Create slightly different data for each ticker
        ticker_data = create_sample_market_data()
        ticker_data['close'] *= np.random.uniform(0.8, 1.2)  # Add some variation
        data_dict[ticker] = ticker_data
    
    print(f"\n🔄 Processing {len(tickers)} tickers in batch...")
    
    # Generate features for all tickers
    batch_results = feature_store.generate_batch_features(
        tickers=tickers,
        data_dict=data_dict,
        include_events=False,  # Skip events for speed
        use_cache=False
    )
    
    print(f"   ✅ Processed {len(batch_results)} tickers successfully")
    
    for ticker, features_df in batch_results.items():
        feature_count = len(features_df.columns) - len(sample_data.columns)
        print(f"   📊 {ticker}: {feature_count} features, {len(features_df)} samples")
    
    # Test 6: ML Dataset Creation
    print("\n" + "="*80)
    print("🧪 TEST 6: ML Dataset Creation")
    print("="*80)
    
    # Create ML-ready dataset
    print(f"\n🤖 Creating ML dataset from batch results...")
    
    X, y = feature_store.create_ml_dataset(
        features_dict=batch_results,
        target_column='price_change_pct',
        lookforward_periods=1,
        min_samples=50
    )
    
    print(f"   ✅ Created ML dataset:")
    print(f"   📊 Features shape: {X.shape}")
    print(f"   🎯 Target shape: {y.shape}")
    print(f"   📈 Target statistics:")
    print(f"     Mean: {y.mean():.4f}")
    print(f"     Std: {y.std():.4f}")
    print(f"     Min: {y.min():.4f}")
    print(f"     Max: {y.max():.4f}")
    
    # Show sample features used
    print(f"\n🔧 Sample features in ML dataset:")
    for i, feature in enumerate(X.columns[:10]):
        print(f"   {i+1}. {feature}")
    if len(X.columns) > 10:
        print(f"   ... and {len(X.columns) - 10} more")
    
    # Test 7: Feature Quality Assessment
    print("\n" + "="*80)
    print("🧪 TEST 7: Feature Quality Assessment")
    print("="*80)
    
    # Analyze feature correlations with target
    print(f"\n🔍 Analyzing feature correlations with target...")
    
    # Calculate correlations
    feature_correlations = {}
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            corr = X[col].corr(y)
            if not pd.isna(corr):
                feature_correlations[col] = abs(corr)
    
    # Sort by absolute correlation
    sorted_correlations = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   📊 Top 10 features by correlation with target:")
    for i, (feature, corr) in enumerate(sorted_correlations[:10], 1):
        print(f"     {i:2d}. {feature:<30} {corr:.4f}")
    
    # Analyze feature distributions
    print(f"\n📈 Feature distribution analysis:")
    
    # Check for constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    print(f"   ⚠️  Constant features: {len(constant_features)}")
    
    # Check for features with high missing values
    high_missing = [col for col in X.columns if X[col].isnull().sum() / len(X) > 0.5]
    print(f"   ⚠️  High missing value features (>50%): {len(high_missing)}")
    
    # Check for highly correlated feature pairs
    correlation_matrix = X.corr()
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = abs(correlation_matrix.iloc[i, j])
            if corr_val > 0.95:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_val
                ))
    
    print(f"   🔗 Highly correlated feature pairs (>0.95): {len(high_corr_pairs)}")
    if high_corr_pairs:
        for feat1, feat2, corr in high_corr_pairs[:5]:
            print(f"     • {feat1} ↔ {feat2}: {corr:.3f}")
    
    # Test 8: Performance Benchmarking
    print("\n" + "="*80)
    print("🧪 TEST 8: Performance Benchmarking")
    print("="*80)
    
    # Time feature generation
    import time
    
    print(f"\n⏱️  Performance benchmarking...")
    
    start_time = time.time()
    perf_features = feature_store.generate_complete_features(
        data=sample_data.copy(),
        ticker='BENCHMARK',
        use_cache=False
    )
    end_time = time.time()
    
    processing_time = end_time - start_time
    features_per_second = len(perf_features.columns) / processing_time
    samples_per_second = len(perf_features) / processing_time
    
    print(f"   ⚡ Processing time: {processing_time:.2f} seconds")
    print(f"   🔧 Features per second: {features_per_second:.1f}")
    print(f"   📊 Samples per second: {samples_per_second:.1f}")
    print(f"   💾 Memory usage: {perf_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING PIPELINE TEST COMPLETE!")
    print("="*80)
    
    # Summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   ✅ Timeseries Features: Working")
    print(f"   ✅ Event Features: Working") 
    print(f"   ✅ Sentiment Features: Working")
    print(f"   ✅ Market Context Features: Working")
    print(f"   ✅ FeatureStore Integration: Working")
    print(f"   ✅ Batch Processing: Working")
    print(f"   ✅ ML Dataset Creation: Working")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   🔧 Total features generated: {len(complete_features.columns)}")
    print(f"   📈 ML dataset features: {X.shape[1]}")
    print(f"   🎯 ML dataset samples: {X.shape[0]}")
    print(f"   ⚡ Processing speed: {features_per_second:.1f} features/sec")
    
    return {
        'complete_features': complete_features,
        'ml_features': X,
        'ml_target': y,
        'batch_results': batch_results,
        'feature_summary': summary,
        'performance_metrics': {
            'processing_time': processing_time,
            'features_per_second': features_per_second,
            'total_features': len(complete_features.columns)
        }
    }


def create_sample_market_data() -> pd.DataFrame:
    """Create sample OHLCV market data for testing"""
    
    # Create 60 days of sample data
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducibility
    
    base_price = 200.0
    data = []
    
    for i, date in enumerate(dates):
        # Add some trend and noise
        trend = 0.1 * i / len(dates)  # Slight upward trend
        noise = np.random.normal(0, 0.02)  # 2% daily volatility
        
        # Calculate price
        price_change = trend + noise
        current_price = base_price * (1 + price_change)
        
        # Generate OHLCV
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        close = low + (high - low) * np.random.random()
        volume = int(np.random.uniform(1000000, 10000000))
        
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
    return df


def create_sample_news_data() -> pd.DataFrame:
    """Create sample news data for sentiment testing"""
    
    base_date = datetime(2024, 1, 1)
    news_articles = []
    
    sample_articles = [
        {
            'title': 'Tesla Reports Strong Q4 Earnings, Beats Expectations',
            'content': 'Tesla Inc reported strong fourth-quarter earnings that beat analyst expectations. The electric vehicle maker showed robust growth in deliveries and improved margins.',
            'ticker': 'TSLA'
        },
        {
            'title': 'Apple Launches Revolutionary New iPhone with AI Features',
            'content': 'Apple unveiled its latest iPhone with advanced AI capabilities. The new device features improved camera technology and enhanced battery life.',
            'ticker': 'AAPL'
        },
        {
            'title': 'Microsoft Azure Cloud Revenue Surges 30% Year-over-Year',
            'content': 'Microsoft Corporation reported impressive growth in its Azure cloud platform, with revenue increasing 30% compared to the previous year.',
            'ticker': 'MSFT'
        },
        {
            'title': 'Federal Reserve Signals Potential Interest Rate Cuts',
            'content': 'The Federal Reserve indicated it may consider interest rate cuts in upcoming meetings, citing economic conditions and inflation targets.',
            'ticker': None
        },
        {
            'title': 'Tesla Stock Downgraded by Major Investment Bank',
            'content': 'A major investment bank downgraded Tesla stock citing concerns about competition in the electric vehicle market and valuation concerns.',
            'ticker': 'TSLA'
        }
    ]
    
    # Replicate articles across different dates
    for i in range(20):  # Create 20 articles
        article = sample_articles[i % len(sample_articles)].copy()
        article['published_date'] = base_date + timedelta(days=i*3)
        article['source'] = f'Source_{i % 3 + 1}'
        article['url'] = f'https://example.com/article_{i}'
        news_articles.append(article)
    
    return pd.DataFrame(news_articles)


def populate_sample_events(event_store: EventStore):
    """Populate event store with sample events"""
    
    from src.event_detection.event_classifier import DetectedEvent, EventType
    from src.event_detection.impact_scorer import ImpactScore, ImpactLevel
    from src.event_detection.entity_linker import LinkedEntity, EntityType
    
    # Create sample events
    sample_events = [
        {
            'event': DetectedEvent(
                event_type=EventType.EARNINGS,
                confidence=0.9,
                entity='Tesla Inc',
                title='Tesla Q4 Earnings Beat',
                description='Tesla reports strong Q4 earnings',
                keywords_matched=['earnings', 'quarterly results', 'beats estimates']
            ),
            'entities': [
                LinkedEntity(
                    text='Tesla',
                    entity_type=EntityType.COMPANY,
                    confidence=0.95,
                    ticker='TSLA',
                    canonical_name='Tesla Inc'
                )
            ],
            'source_info': {
                'published_date': datetime(2024, 1, 15),
                'source': 'Reuters',
                'url': 'https://example.com/tesla-earnings'
            }
        },
        {
            'event': DetectedEvent(
                event_type=EventType.PRODUCT_LAUNCH,
                confidence=0.85,
                entity='Apple Inc',
                title='Apple Product Launch',
                description='Apple announces new iPhone with AI features',
                keywords_matched=['launches', 'announces new', 'product release']
            ),
            'entities': [
                LinkedEntity(
                    text='Apple',
                    entity_type=EntityType.COMPANY,
                    confidence=0.9,
                    ticker='AAPL',
                    canonical_name='Apple Inc'
                )
            ],
            'source_info': {
                'published_date': datetime(2024, 1, 20),
                'source': 'TechCrunch',
                'url': 'https://example.com/apple-launch'
            }
        }
    ]
    
    # Store events in event store
    for event_data in sample_events:
        # Create ImpactScore with the event parameter
        impact_score = ImpactScore(
            event=event_data['event'],
            impact_level=ImpactLevel.HIGH if event_data['event'].event_type == EventType.EARNINGS else ImpactLevel.MEDIUM,
            impact_score=0.8 if event_data['event'].event_type == EventType.EARNINGS else 0.6,
            price_target_change=5.0 if event_data['event'].event_type == EventType.EARNINGS else 3.0,
            volatility_increase=15.0 if event_data['event'].event_type == EventType.EARNINGS else 10.0,
            time_horizon='1-3 days' if event_data['event'].event_type == EventType.EARNINGS else '1-2 days',
            confidence=0.85 if event_data['event'].event_type == EventType.EARNINGS else 0.8
        )
        
        event_store.store_event(
            event_data['event'],
            impact_score,
            event_data['entities'],
            event_data['source_info']
        )


if __name__ == "__main__":
    test_feature_engineering_pipeline() 
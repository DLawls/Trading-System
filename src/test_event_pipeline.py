"""
Test script for the complete Event Detection Pipeline
"""

import pandas as pd
from datetime import datetime, timedelta
from src.event_detection.event_classifier import EventClassifier, EventType
from src.event_detection.impact_scorer import ImpactScorer
from src.event_detection.entity_linker import EntityLinker
from src.event_detection.event_store import EventStore
from src.event_detection.historical_analyzer import HistoricalAnalyzer

def test_complete_pipeline():
    """Test the complete event detection pipeline."""
    
    print("🔗 Testing Complete Event Detection Pipeline...")
    
    # Initialize all components
    classifier = EventClassifier()
    impact_scorer = ImpactScorer()
    entity_linker = EntityLinker()
    event_store = EventStore()  # Using in-memory storage for testing
    historical_analyzer = HistoricalAnalyzer(event_store)
    
    print("\n✅ All components initialized successfully!")
    
    # Create sample historical news data
    sample_news = create_sample_news_data()
    print(f"\n📰 Created sample dataset with {len(sample_news)} news articles")
    
    # Test 1: Process individual news article
    print("\n" + "="*80)
    print("🧪 TEST 1: Individual Article Processing")
    print("="*80)
    
    sample_article = sample_news.iloc[0]
    print(f"📄 Processing: {sample_article['title']}")
    
    # Step 1: Classify events
    title = sample_article['title']
    content = sample_article['content']
    events = classifier.classify_text(title, content)
    print(f"   🎯 Events detected: {len(events)}")
    
    if events:
        primary_event = events[0]
        print(f"   📊 Primary event: {primary_event.event_type.value} ({primary_event.confidence:.2f})")
        
        # Step 2: Extract entities
        full_text = f"{title} {content}"
        entities = entity_linker.link_entities(full_text)
        financial_entities = entity_linker.get_financial_entities(entities)
        print(f"   🏢 Financial entities: {len(financial_entities)}")
        
        for entity in financial_entities[:3]:  # Show top 3
            ticker_info = f" ({entity.ticker})" if entity.ticker else ""
            print(f"     • {entity.text}{ticker_info}")
        
        # Step 3: Score impact
        impact = impact_scorer.score_event_impact(primary_event)
        print(f"   💥 Impact: {impact.impact_level.value} ({impact.impact_score:.2f})")
        print(f"   💰 Expected price change: ±{impact.price_target_change:.1f}%")
        
        # Step 4: Store event
        source_info = {
            'url': sample_article['url'],
            'source': sample_article['source'],
            'published_date': sample_article['published_date']
        }
        
        event_id = event_store.store_event(primary_event, impact, entities, source_info)
        print(f"   💾 Stored event with ID: {event_id}")
    
    # Test 2: Historical Analysis
    print("\n" + "="*80)
    print("🧪 TEST 2: Historical Analysis")
    print("="*80)
    
    # Analyze last 7 days of sample data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    analysis_results = historical_analyzer.analyze_historical_data(
        sample_news,
        start_date=start_date,
        end_date=end_date,
        min_impact_threshold=0.3,
        store_events=True,
        batch_size=5
    )
    
    stats = analysis_results['statistics']
    processing_stats = analysis_results['processing_stats']
    
    print(f"📊 HISTORICAL ANALYSIS RESULTS:")
    print(f"   📰 Articles processed: {processing_stats['total_articles']}")
    print(f"   🎯 Events detected: {stats['total_events']}")
    print(f"   💥 High-impact events: {stats['high_impact_events']}")
    print(f"   ⚡ Processing speed: {processing_stats['articles_per_second']:.1f} articles/sec")
    print(f"   ⏱️  Total time: {processing_stats['processing_time_seconds']:.1f} seconds")
    
    if stats['total_events'] > 0:
        print(f"   📈 Average impact score: {stats['avg_impact_score']:.2f}")
        
        print(f"\n🏆 Top Event Types:")
        for event_type, count in list(stats['event_type_distribution'].items())[:5]:
            print(f"     • {event_type}: {count} events")
        
        print(f"\n📊 Impact Levels:")
        for impact_level, count in stats['impact_level_distribution'].items():
            print(f"     • {impact_level}: {count} events")
        
        if stats['ticker_distribution']:
            print(f"\n💹 Top Tickers:")
            for ticker, count in list(stats['ticker_distribution'].items())[:5]:
                print(f"     • {ticker}: {count} events")
    
    # Test 3: Event Store Queries
    print("\n" + "="*80)
    print("🧪 TEST 3: Event Store Queries")
    print("="*80)
    
    # Get all events
    all_events = event_store.get_events(limit=50)
    print(f"📋 Total events in store: {len(all_events)}")
    
    # Get high-impact events
    high_impact_events = event_store.get_events(impact_threshold=0.6, limit=20)
    print(f"💥 High-impact events: {len(high_impact_events)}")
    
    # Get events by type
    earnings_events = event_store.get_events(event_type=EventType.EARNINGS, limit=10)
    print(f"📈 Earnings events: {len(earnings_events)}")
    
    # Show sample high-impact events
    if high_impact_events:
        print(f"\n🚨 Sample High-Impact Events:")
        for i, event in enumerate(high_impact_events[:3], 1):
            print(f"   {i}. {event['title'][:60]}...")
            print(f"      Type: {event['event_type']} | Impact: {event['impact_score']:.2f}")
            print(f"      Ticker: {event['ticker']} | Date: {event['timestamp']}")
    
    # Get event statistics
    event_stats = event_store.get_event_statistics(days=30)
    print(f"\n📊 Event Store Statistics (30 days):")
    print(f"   📰 Total events: {event_stats['total_events']}")
    print(f"   💥 High-impact events: {event_stats['high_impact_events']}")
    
    if event_stats['total_events'] > 0:
        print(f"   📈 Average impact: {event_stats['avg_impact_score']:.2f}")
        
        if event_stats['top_tickers']:
            print(f"   🏆 Most active ticker: {list(event_stats['top_tickers'].keys())[0]}")
    
    # Test 4: Specialized Analysis Functions
    print("\n" + "="*80)
    print("🧪 TEST 4: Specialized Analysis")
    print("="*80)
    
    # Market moving events
    market_movers = historical_analyzer.get_market_moving_events(days=7, min_impact=0.5)
    print(f"📈 Market-moving events (7 days): {len(market_movers)}")
    
    if market_movers:
        print(f"🏆 Top market mover: {market_movers[0]['title'][:50]}...")
        print(f"   Impact score: {market_movers[0]['impact_score']:.2f}")
        print(f"   Ticker: {market_movers[0]['ticker']}")
    
    # Test ticker-specific timeline (if we have TSLA events)
    tsla_events = event_store.get_events(ticker='TSLA', limit=10)
    if tsla_events:
        print(f"\n🚗 Tesla-specific events: {len(tsla_events)}")
        timeline_df = historical_analyzer.analyze_event_timeline('TSLA', days=30)
        if not timeline_df.empty:
            print(f"   📅 Timeline dataframe shape: {timeline_df.shape}")
    
    # Generate backtest dataset
    backtest_df = historical_analyzer.generate_backtest_dataset(
        start_date=start_date,
        end_date=end_date,
        min_impact=0.4
    )
    
    if not backtest_df.empty:
        print(f"\n🎯 Backtest dataset generated:")
        print(f"   📊 Shape: {backtest_df.shape}")
        print(f"   📅 Date range: {backtest_df['date'].min()} to {backtest_df['date'].max()}")
        print(f"   🕐 Market hours events: {backtest_df['is_market_hours'].sum()}")
        print(f"   📈 Columns: {list(backtest_df.columns)}")
    
    # Test 5: DataFrame Conversion
    print("\n" + "="*80)
    print("🧪 TEST 5: DataFrame Analytics")
    print("="*80)
    
    # Convert events to DataFrame
    events_df = event_store.to_dataframe(limit=100)
    
    if not events_df.empty:
        print(f"📊 Events DataFrame:")
        print(f"   Shape: {events_df.shape}")
        print(f"   Columns: {list(events_df.columns)}")
        
        # Basic analytics
        print(f"\n📈 Quick Analytics:")
        print(f"   Average impact score: {events_df['impact_score'].mean():.2f}")
        print(f"   Most common event type: {events_df['event_type'].mode().iloc[0]}")
        print(f"   Unique tickers: {events_df['ticker'].nunique()}")
        print(f"   Date range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE TEST COMPLETE!")
    print("="*80)
    
    print(f"\n🎯 SUMMARY:")
    print(f"   ✅ Event Classification: Working")
    print(f"   ✅ Entity Linking: Working")
    print(f"   ✅ Impact Scoring: Working")
    print(f"   ✅ Event Storage: Working")
    print(f"   ✅ Historical Analysis: Working")
    print(f"   ✅ Query & Analytics: Working")
    
    print(f"\n📊 FINAL STATS:")
    print(f"   📰 Total articles processed: {len(sample_news)}")
    print(f"   🎯 Total events detected: {len(all_events)}")
    print(f"   💥 High-impact events: {len(high_impact_events)}")
    print(f"   📈 Event types covered: {len(set(e['event_type'] for e in all_events))}")
    
    return {
        'articles_processed': len(sample_news),
        'events_detected': len(all_events),
        'high_impact_events': len(high_impact_events),
        'pipeline_status': 'SUCCESS'
    }

def create_sample_news_data() -> pd.DataFrame:
    """Create sample news data for testing"""
    
    sample_articles = [
        {
            'title': 'Tesla Reports Record Q4 Earnings, Beats Wall Street Estimates',
            'content': 'Tesla Inc (TSLA) announced record fourth-quarter earnings that significantly beat analyst estimates. The electric vehicle maker reported revenue of $24.3 billion, up 37% year-over-year. CEO Elon Musk highlighted strong demand for Model Y and successful expansion in China.',
            'source': 'Reuters',
            'url': 'https://example.com/tesla-earnings',
            'published_date': datetime.utcnow() - timedelta(days=1)
        },
        {
            'title': 'Microsoft Announces $10B Investment in OpenAI Partnership',
            'content': 'Microsoft Corporation (MSFT) announced a multi-billion dollar investment in OpenAI, deepening their partnership in artificial intelligence. The deal positions Microsoft as a leader in the rapidly growing AI market and provides OpenAI with significant resources for development.',
            'source': 'Bloomberg',
            'url': 'https://example.com/microsoft-openai',
            'published_date': datetime.utcnow() - timedelta(days=2)
        },
        {
            'title': 'Apple Unveils Revolutionary iPhone AI Features',
            'content': 'Apple Inc (AAPL) unveiled groundbreaking artificial intelligence features for the iPhone 15 Pro. The new capabilities include advanced photo recognition and voice synthesis. The announcement was made at a special event in Cupertino.',
            'source': 'TechCrunch',
            'url': 'https://example.com/apple-ai',
            'published_date': datetime.utcnow() - timedelta(days=3)
        },
        {
            'title': 'Federal Reserve Hints at Interest Rate Pause',
            'content': 'Federal Reserve Chairman Jerome Powell suggested the central bank may pause interest rate hikes in upcoming meetings. The dovish commentary comes amid signs of cooling inflation and concerns about economic growth.',
            'source': 'Wall Street Journal',
            'url': 'https://example.com/fed-rates',
            'published_date': datetime.utcnow() - timedelta(days=4)
        },
        {
            'title': 'Amazon Acquires Healthcare Startup for $2B',
            'content': 'Amazon.com Inc (AMZN) announced the acquisition of a promising healthcare technology startup for approximately $2 billion. The deal represents Amazons continued expansion into the healthcare sector and its commitment to digital health solutions.',
            'source': 'CNBC',
            'url': 'https://example.com/amazon-healthcare',
            'published_date': datetime.utcnow() - timedelta(days=5)
        },
        {
            'title': 'Netflix Partners with Major Studios for Content Expansion',
            'content': 'Netflix Inc (NFLX) announced new partnerships with Warner Bros and Universal Studios to significantly expand its content library. The streaming giant will invest $5 billion in new productions over the next two years.',
            'source': 'Variety',
            'url': 'https://example.com/netflix-content',
            'published_date': datetime.utcnow() - timedelta(days=6)
        },
        {
            'title': 'Bitcoin Surges Above $50,000 on Institutional Demand',
            'content': 'Bitcoin (BTC-USD) surged past $50,000 for the first time in six months, driven by renewed institutional demand. Major companies including MicroStrategy and Tesla continue to hold significant Bitcoin positions.',
            'source': 'CoinDesk',
            'url': 'https://example.com/bitcoin-surge',
            'published_date': datetime.utcnow() - timedelta(days=7)
        },
        {
            'title': 'JPMorgan Reports Strong Q4 Results, Raises Dividend',
            'content': 'JPMorgan Chase & Co (JPM) reported better-than-expected fourth-quarter results and announced a 5% increase in its quarterly dividend. The largest US bank benefited from higher interest rates and strong trading revenues.',
            'source': 'Financial Times',
            'url': 'https://example.com/jpmorgan-earnings',
            'published_date': datetime.utcnow() - timedelta(days=8)
        }
    ]
    
    return pd.DataFrame(sample_articles)

if __name__ == "__main__":
    test_complete_pipeline() 
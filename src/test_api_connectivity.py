"""
Quick API Connectivity Test

Tests basic connectivity to Alpaca and News API to ensure everything is working.
"""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_alpaca_connectivity():
    """Test basic Alpaca API connectivity"""
    print("🔌 Testing Alpaca API connectivity...")
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        # Get API keys from environment
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        
        if not api_key or not api_secret:
            print("❌ Alpaca API keys not found in environment")
            return False
        
        print(f"  ✅ API Key found: {api_key[:8]}...")
        
        # Initialize client
        client = StockHistoricalDataClient(api_key, api_secret)
        print("  ✅ Client initialized")
        
        # Test with simple symbols that should work
        test_symbols = ['AAPL', 'MSFT', 'SPY']
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)
        
        print(f"  📊 Testing data retrieval for {test_symbols}")
        print(f"  📅 Time range: {start_time.date()} to {end_time.date()}")
        
        request = StockBarsRequest(
            symbol_or_symbols=test_symbols,
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time,
            feed='iex'  # Use free IEX data feed
        )
        
        bars = client.get_stock_bars(request)
        
        # Check what we got back
        if hasattr(bars, 'keys'):
            available_symbols = list(bars.keys())
            print(f"  ✅ Data received for symbols: {available_symbols}")
            
            for symbol in available_symbols:
                symbol_data = bars[symbol]
                print(f"    {symbol}: {len(symbol_data)} bars")
                if symbol_data:
                    latest = symbol_data[-1]
                    print(f"      Latest: ${latest.close:.2f} at {latest.timestamp}")
            
            return True
        else:
            print(f"  ⚠️  Unexpected response format: {type(bars)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Alpaca API error: {str(e)}")
        return False

async def test_news_api_connectivity():
    """Test basic News API connectivity"""
    print("\n📰 Testing News API connectivity...")
    
    try:
        import requests
        
        api_key = os.getenv('NEWS_API_KEY')
        
        if not api_key:
            print("❌ News API key not found in environment")
            return False
        
        print(f"  ✅ API Key found: {api_key[:8]}...")
        
        # Test simple request
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'apiKey': api_key,
            'country': 'us',
            'category': 'business',
            'pageSize': 5
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"  ✅ Successfully retrieved {len(articles)} articles")
            
            if articles:
                print("  📰 Sample headlines:")
                for i, article in enumerate(articles[:3]):
                    print(f"    {i+1}. {article['title'][:80]}...")
            
            return True
        else:
            print(f"  ❌ News API error: Status {response.status_code}")
            print(f"      Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ News API error: {str(e)}")
        return False

async def test_data_ingestion_manager():
    """Test our data ingestion manager with real APIs"""
    print("\n🔄 Testing Data Ingestion Manager...")
    
    try:
        from src.data_ingestion.main import DataIngestionManager
        
        # Initialize with environment variables
        manager = DataIngestionManager(
            alpaca_api_key=os.getenv('ALPACA_API_KEY'),
            alpaca_api_secret=os.getenv('ALPACA_API_SECRET'),
            alpaca_base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            news_api_key=os.getenv('NEWS_API_KEY'),
            symbols=['AAPL', 'MSFT']  # Use symbols that should work
        )
        
        print("  ✅ Data Ingestion Manager initialized")
        
        # Test market data fetch (don't start scheduler, just test fetch)
        print("  📊 Testing market data fetch...")
        market_data = manager.get_latest_market_data()
        
        if market_data and not market_data.empty:
            print(f"  ✅ Market data retrieved: {len(market_data)} records")
            print(f"      Symbols: {market_data['symbol'].unique()}")
        else:
            print("  ⚠️  No market data retrieved")
        
        # Test news fetch
        print("  📰 Testing news data fetch...")
        news_data = manager.get_latest_news()
        
        if news_data and not news_data.empty:
            print(f"  ✅ News data retrieved: {len(news_data)} articles")
        else:
            print("  ⚠️  No news data retrieved")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data Ingestion Manager error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all connectivity tests"""
    print("🏥 API Connectivity Health Check")
    print("=" * 50)
    
    results = []
    
    # Test each API
    results.append(await test_alpaca_connectivity())
    results.append(await test_news_api_connectivity())
    results.append(await test_data_ingestion_manager())
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("🎉 ALL API TESTS PASSED!")
        print("   Your trading system is ready for real data ingestion.")
        print("\n🚀 Next steps:")
        print("   1. Run comprehensive tests: python tests/run_tests.py")
        print("   2. Set up monitoring dashboard")
        print("   3. Begin paper trading validation")
    else:
        print("⚠️  Some API tests failed. Please check the errors above.")
        failed_count = len(results) - sum(results)
        print(f"   {failed_count}/{len(results)} tests failed")
    
    return all(results)

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        exit(1) 
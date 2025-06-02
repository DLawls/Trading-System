"""
Quick Test - Identify TSLA Issue and Basic API Status
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def test_alpaca_simple():
    """Quick Alpaca test with minimal data"""
    print("🔌 Quick Alpaca Test...")
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        client = StockHistoricalDataClient(
            os.getenv('ALPACA_API_KEY'), 
            os.getenv('ALPACA_API_SECRET')
        )
        
        # Test with just AAPL first (most reliable)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)  # Just 6 hours of data
        
        request = StockBarsRequest(
            symbol_or_symbols=['AAPL'],
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time,
            feed='iex'
        )
        
        bars = client.get_stock_bars(request)
        
        if 'AAPL' in bars and bars['AAPL']:
            print(f"  ✅ AAPL: {len(bars['AAPL'])} bars")
            return True
        else:
            print("  ❌ AAPL: No data")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

def test_tsla_specifically():
    """Test TSLA specifically to see what happens"""
    print("\n🚗 Testing TSLA specifically...")
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        client = StockHistoricalDataClient(
            os.getenv('ALPACA_API_KEY'), 
            os.getenv('ALPACA_API_SECRET')
        )
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)
        
        request = StockBarsRequest(
            symbol_or_symbols=['TSLA'],
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time,
            feed='iex'
        )
        
        bars = client.get_stock_bars(request)
        
        print(f"  Response type: {type(bars)}")
        print(f"  Available symbols: {list(bars.keys()) if hasattr(bars, 'keys') else 'No keys'}")
        
        if 'TSLA' in bars:
            print(f"  ✅ TSLA found: {len(bars['TSLA'])} bars")
            return True
        else:
            print("  ⚠️  TSLA not in response - might not be available via IEX feed")
            return False
            
    except Exception as e:
        print(f"  ❌ TSLA Error: {str(e)}")
        return False

def test_news_quick():
    """Quick news API test"""
    print("\n📰 Quick News Test...")
    
    try:
        import requests
        
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'apiKey': os.getenv('NEWS_API_KEY'),
            'country': 'us',
            'pageSize': 1  # Just 1 article
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Got {len(data.get('articles', []))} articles")
            return True
        else:
            print(f"  ❌ Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("⚡ Quick API Tests")
    print("=" * 30)
    
    results = []
    results.append(test_alpaca_simple())
    results.append(test_tsla_specifically())
    results.append(test_news_quick())
    
    print("\n" + "=" * 30)
    
    if all(results):
        print("🎉 All quick tests passed!")
    else:
        print("⚠️  Some issues found - but this helps us diagnose!")
        
    print("\n💡 Next: Fix any identified issues and run full tests") 
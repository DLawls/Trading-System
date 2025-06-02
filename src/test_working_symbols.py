"""
Test to identify working symbols with IEX feed
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def test_symbols_batch(symbols):
    """Test a batch of symbols"""
    print(f"🔍 Testing symbols: {symbols}")
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        client = StockHistoricalDataClient(
            os.getenv('ALPACA_API_KEY'), 
            os.getenv('ALPACA_API_SECRET')
        )
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=12)  # 12 hours window
        
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time,
            feed='iex'
        )
        
        bars = client.get_stock_bars(request)
        
        working_symbols = []
        for symbol in symbols:
            if symbol in bars and bars[symbol]:
                working_symbols.append(symbol)
                print(f"  ✅ {symbol}: {len(bars[symbol])} bars")
            else:
                print(f"  ❌ {symbol}: No data")
        
        return working_symbols
        
    except Exception as e:
        print(f"  ❌ Error testing batch: {str(e)}")
        return []

if __name__ == "__main__":
    print("🔎 Finding Working Symbols for IEX Feed")
    print("=" * 50)
    
    # Test common symbols in batches
    test_batches = [
        ['AAPL', 'MSFT', 'GOOGL'],
        ['AMZN', 'META', 'NVDA'], 
        ['TSLA', 'NFLX', 'DIS'],
        ['SPY', 'QQQ', 'IWM'],
        ['IBM', 'GE', 'JPM'],
        ['COST', 'WMT', 'PG']
    ]
    
    all_working = []
    
    for batch in test_batches:
        working = test_symbols_batch(batch)
        all_working.extend(working)
        print()
    
    print("=" * 50)
    print(f"🎯 WORKING SYMBOLS ({len(all_working)}):")
    print(f"   {all_working}")
    
    if len(all_working) >= 5:
        recommended = all_working[:8]  # Take first 8
        print(f"\n✅ RECOMMENDED SYMBOLS FOR CONFIG:")
        print(f'   symbols: {recommended}')
    else:
        print("\n⚠️  Too few working symbols found. May need to use SIP feed (paid).")
        
    print(f"\n💡 Update your config.yaml with the working symbols above.") 
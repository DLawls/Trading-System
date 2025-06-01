"""
Test script for EventScheduler functionality
"""

import asyncio
from datetime import datetime
from src.data_ingestion.events import EventScheduler

async def test_event_scheduler():
    """Test the EventScheduler with mock data."""
    
    print("Testing EventScheduler...")
    
    # Initialize scheduler
    scheduler = EventScheduler()
    
    # Test symbols (mix of stocks and crypto)
    test_symbols = ['TSLA', 'AAPL', 'MSFT', 'ETHUSD', 'BTCUSD']
    
    # Update events
    await scheduler.update_events(test_symbols)
    
    print(f"\n📅 Total Events Found: {len(scheduler.events)}")
    print(f"📊 Last Updated: {scheduler.last_update}")
    
    # Display all events
    print("\n🔮 All Upcoming Events:")
    for event in scheduler.events:
        time_until = (event.event_time - datetime.now()).total_seconds() / 3600
        print(f"  • {event.symbol} - {event.description}")
        print(f"    ⏰ {event.event_time.strftime('%Y-%m-%d %H:%M')} ({time_until:.1f}h)")
        print(f"    📈 Impact: {event.impact_level}")
        print()
    
    # Test filtering methods
    print("🚨 High Impact Events:")
    high_impact = scheduler.get_high_impact_events()
    for event in high_impact:
        print(f"  • {event.symbol}: {event.description}")
    
    print(f"\n⏱️  Events in Next 24h: {len(scheduler.get_upcoming_events(24))}")
    
    # Test symbol-specific events
    for symbol in ['TSLA', 'ETHUSD']:
        symbol_events = scheduler.get_events_by_symbol(symbol)
        if symbol_events:
            print(f"\n📊 {symbol} Events:")
            for event in symbol_events:
                print(f"  • {event.description} ({event.impact_level})")
    
    # Convert to DataFrame
    df = scheduler.to_dataframe()
    if not df.empty:
        print(f"\n📋 Events DataFrame Shape: {df.shape}")
        print("📋 Events Summary:")
        print(df[['symbol', 'event_type', 'impact_level', 'hours_until']].head())

if __name__ == "__main__":
    asyncio.run(test_event_scheduler()) 
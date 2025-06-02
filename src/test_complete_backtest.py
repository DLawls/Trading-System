"""
Complete Backtesting Demo - D-Laws Trading System

This script demonstrates the complete trading system by running a comprehensive
backtest using real market data simulation and multi-strategy signal generation.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from backtesting.historical_simulator import HistoricalEventSimulator
from backtesting.portfolio_simulator import PortfolioSimulator, FillModel
from backtesting.metrics_logger import MetricsLogger
from signal_generation.schemas import TradingSignal, SignalAction
from event_detection.event_classifier import EventClassifier
from event_detection.impact_scorer import ImpactScorer


class AdvancedSignalGenerator:
    """Advanced signal generator combining multiple strategies"""
    
    def __init__(self, strategies=['momentum', 'mean_reversion', 'event_driven']):
        self.strategies = strategies
        self.event_classifier = EventClassifier()
        self.impact_scorer = ImpactScorer()
        
    async def generate_signals(self, symbols, market_data, news_data=None, 
                             events=None, features=None, current_time=None):
        """Generate signals using multiple strategies"""
        
        all_signals = []
        
        for symbol in symbols:
            if symbol not in market_data or market_data[symbol].empty:
                continue
                
            df = market_data[symbol]
            if len(df) < 20:
                continue
                
            signals = []
            
            # 1. Momentum Strategy
            if 'momentum' in self.strategies:
                momentum_signals = self._generate_momentum_signals(
                    symbol, df, current_time
                )
                signals.extend(momentum_signals)
            
            # 2. Mean Reversion Strategy
            if 'mean_reversion' in self.strategies:
                reversion_signals = self._generate_mean_reversion_signals(
                    symbol, df, current_time
                )
                signals.extend(reversion_signals)
            
            # 3. Event-Driven Strategy
            if 'event_driven' in self.strategies and news_data:
                event_signals = await self._generate_event_signals(
                    symbol, df, news_data, current_time
                )
                signals.extend(event_signals)
            
            # Combine and rank signals
            final_signals = self._combine_signals(signals)
            all_signals.extend(final_signals)
        
        return all_signals
    
    def _generate_momentum_signals(self, symbol, df, current_time):
        """Generate momentum-based signals"""
        signals = []
        
        if len(df) >= 20:
            # Multiple timeframe momentum
            short_ma = df['close'].rolling(5).mean().iloc[-1]
            medium_ma = df['close'].rolling(10).mean().iloc[-1]
            long_ma = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Strong momentum (all MAs aligned)
            if short_ma > medium_ma > long_ma and current_price > short_ma * 1.01:
                signals.append({
                    'symbol': symbol,
                    'action': SignalAction.BUY,
                    'confidence': 0.8,
                    'position_size': 0.15,
                    'strategy': 'momentum',
                    'timestamp': current_time
                })
            
            # Strong downtrend
            elif short_ma < medium_ma < long_ma and current_price < short_ma * 0.99:
                signals.append({
                    'symbol': symbol,
                    'action': SignalAction.SELL,
                    'confidence': 0.7,
                    'position_size': 1.0,
                    'strategy': 'momentum',
                    'timestamp': current_time
                })
        
        return signals
    
    def _generate_mean_reversion_signals(self, symbol, df, current_time):
        """Generate mean reversion signals"""
        signals = []
        
        if len(df) >= 20:
            price = df['close'].iloc[-1]
            mean_price = df['close'].rolling(20).mean().iloc[-1]
            std_price = df['close'].rolling(20).std().iloc[-1]
            
            # Strong oversold condition
            if price < mean_price - 2.0 * std_price:
                signals.append({
                    'symbol': symbol,
                    'action': SignalAction.BUY,
                    'confidence': 0.75,
                    'position_size': 0.08,
                    'strategy': 'mean_reversion',
                    'timestamp': current_time
                })
            
            # Strong overbought condition
            elif price > mean_price + 2.0 * std_price:
                signals.append({
                    'symbol': symbol,
                    'action': SignalAction.SELL,
                    'confidence': 0.7,
                    'position_size': 0.5,
                    'strategy': 'mean_reversion',
                    'timestamp': current_time
                })
        
        return signals
    
    async def _generate_event_signals(self, symbol, df, news_data, current_time):
        """Generate event-driven signals based on news"""
        signals = []
        
        # Get recent news for this symbol
        recent_news = [
            news for news in news_data 
            if news.get('target_symbol') == symbol 
            and abs((news['published_at'] - current_time).total_seconds()) < 3600  # Last hour
        ]
        
        for news in recent_news:
            # Classify the event
            events = self.event_classifier.classify_article(
                news['title'], news['description']
            )
            
            for event in events:
                # Score the impact
                impact_result = self.impact_scorer.score_impact(
                    event, symbol, df['close'].iloc[-1]
                )
                
                # Generate signals based on impact
                if impact_result.impact_level in ['high', 'extreme']:
                    if 'positive' in impact_result.reasoning.lower():
                        signals.append({
                            'symbol': symbol,
                            'action': SignalAction.BUY,
                            'confidence': min(0.9, event.confidence + 0.2),
                            'position_size': 0.12 if impact_result.impact_level == 'extreme' else 0.08,
                            'strategy': 'event_driven',
                            'timestamp': current_time
                        })
                    elif 'negative' in impact_result.reasoning.lower():
                        signals.append({
                            'symbol': symbol,
                            'action': SignalAction.SELL,
                            'confidence': min(0.9, event.confidence + 0.1),
                            'position_size': 0.8 if impact_result.impact_level == 'extreme' else 0.5,
                            'strategy': 'event_driven',
                            'timestamp': current_time
                        })
        
        return signals
    
    def _combine_signals(self, signals):
        """Combine and filter signals"""
        if not signals:
            return []
        
        # Group by action
        buy_signals = [s for s in signals if s['action'] == SignalAction.BUY]
        sell_signals = [s for s in signals if s['action'] == SignalAction.SELL]
        
        final_signals = []
        
        # Choose best buy signal
        if buy_signals:
            best_buy = max(buy_signals, key=lambda x: x['confidence'])
            final_signals.append(TradingSignal(
                symbol=best_buy['symbol'],
                action=best_buy['action'],
                confidence=best_buy['confidence'],
                position_size=best_buy['position_size'],
                timestamp=best_buy['timestamp'],
                signal_id=f"{best_buy['strategy']}_{best_buy['symbol']}_{best_buy['timestamp']}"
            ))
        
        # Choose best sell signal
        if sell_signals:
            best_sell = max(sell_signals, key=lambda x: x['confidence'])
            final_signals.append(TradingSignal(
                symbol=best_sell['symbol'],
                action=best_sell['action'],
                confidence=best_sell['confidence'],
                position_size=best_sell['position_size'],
                timestamp=best_sell['timestamp'],
                signal_id=f"{best_sell['strategy']}_{best_sell['symbol']}_{best_sell['timestamp']}"
            ))
        
        return final_signals


def create_realistic_market_data(symbols, start_date, end_date, freq='1h'):
    """Create more realistic market data with trends and volatility"""
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    market_data = {}
    
    for i, symbol in enumerate(symbols):
        np.random.seed(42 + i)  # Reproducible but different for each symbol
        
        num_periods = len(date_range)
        base_price = 100 + i * 50  # Different starting prices
        
        # Create different market regimes
        regime_length = num_periods // 4
        regimes = []
        
        # Bull market
        regimes.extend([np.random.normal(0.001, 0.015, regime_length)])
        # Volatile sideways
        regimes.extend([np.random.normal(0.0002, 0.025, regime_length)])
        # Bear market
        regimes.extend([np.random.normal(-0.0008, 0.02, regime_length)])
        # Recovery
        remaining = num_periods - 3 * regime_length
        regimes.extend([np.random.normal(0.0012, 0.018, remaining)])
        
        returns = np.concatenate(regimes)
        
        # Calculate cumulative prices
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        df_data = []
        for j, (timestamp, close_price) in enumerate(zip(date_range, prices)):
            # More realistic intraday movements
            open_price = prices[j-1] if j > 0 else close_price
            
            # Random intraday range
            daily_range = abs(np.random.normal(0, 0.01)) * close_price
            high = max(open_price, close_price) + daily_range * np.random.random()
            low = min(open_price, close_price) - daily_range * np.random.random()
            
            # Volume with some correlation to price movement
            price_change = abs(close_price - open_price) / open_price if open_price > 0 else 0
            base_volume = np.random.randint(500000, 2000000)
            volume = int(base_volume * (1 + price_change * 10))
            
            df_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        market_data[symbol] = pd.DataFrame(df_data, index=date_range)
    
    return market_data


def create_realistic_news_data(start_date, end_date, symbols):
    """Create realistic news data with various event types"""
    
    news_templates = {
        'earnings': [
            "{} reports Q{} earnings: ${:.2f} EPS vs ${:.2f} expected",
            "{} beats earnings estimates with strong Q{} performance",
            "{} misses Q{} earnings expectations, revenue down {}%"
        ],
        'analyst': [
            "Morgan Stanley upgrades {} to overweight, raises target to ${}",
            "Goldman Sachs downgrades {} citing valuation concerns",
            "JPMorgan initiates {} coverage with buy rating"
        ],
        'product': [
            "{} announces breakthrough in artificial intelligence technology",
            "{} launches new product line expected to drive growth",
            "{} faces regulatory scrutiny over new product safety"
        ],
        'general': [
            "Federal Reserve signals potential interest rate changes",
            "Market volatility increases amid geopolitical tensions",
            "Tech sector shows strong momentum in trading session"
        ]
    }
    
    news_data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Generate 3-8 news articles per day
        num_articles = np.random.randint(3, 9)
        
        for _ in range(num_articles):
            article_time = current_date + timedelta(
                hours=np.random.randint(6, 22),
                minutes=np.random.randint(0, 60)
            )
            
            # 60% chance of company-specific news
            if np.random.random() < 0.6 and symbols:
                target_symbol = np.random.choice(symbols)
                news_type = np.random.choice(['earnings', 'analyst', 'product'])
                template = np.random.choice(news_templates[news_type])
                
                if news_type == 'earnings':
                    quarter = np.random.randint(1, 5)
                    eps_actual = round(np.random.uniform(0.5, 3.0), 2)
                    eps_expected = round(eps_actual + np.random.uniform(-0.5, 0.3), 2)
                    title = template.format(target_symbol, quarter, eps_actual, eps_expected)
                elif news_type == 'analyst':
                    if 'target' in template:
                        price_target = np.random.randint(150, 300)
                        title = template.format(target_symbol, price_target)
                    else:
                        title = template.format(target_symbol)
                else:
                    title = template.format(target_symbol)
                
                description = f"Detailed analysis of {target_symbol} performance and outlook..."
                
            else:
                target_symbol = None
                title = np.random.choice(news_templates['general'])
                description = "Market analysis and economic indicators suggest..."
            
            news_data.append({
                'title': title,
                'description': description,
                'published_at': article_time,
                'target_symbol': target_symbol,
                'source': 'Financial News Network'
            })
        
        current_date += timedelta(days=1)
    
    return news_data


async def run_comprehensive_backtest():
    """Run a comprehensive backtest demonstration"""
    
    print("🚀 Starting Comprehensive Backtesting Demo")
    print("=" * 60)
    
    # Configuration
    symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)  # 6 months
    initial_capital = 100000  # $100k
    
    print(f"📊 Backtest Period: {start_date.date()} to {end_date.date()}")
    print(f"💰 Initial Capital: ${initial_capital:,}")
    print(f"📈 Symbols: {', '.join(symbols)}")
    print()
    
    # Create realistic data
    print("📥 Generating market data...")
    market_data = create_realistic_market_data(symbols, start_date, end_date)
    
    print("📰 Generating news data...")
    news_data = create_realistic_news_data(start_date, end_date, symbols)
    
    print(f"✅ Generated {sum(len(df) for df in market_data.values())} market data points")
    print(f"✅ Generated {len(news_data)} news articles")
    print()
    
    # Create temporary directory for data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save market data
        market_dir = temp_path / "market_data"
        market_dir.mkdir()
        
        for symbol, df in market_data.items():
            df.to_csv(market_dir / f"{symbol}.csv")
        
        # Save news data
        news_df = pd.DataFrame(news_data)
        news_df.to_csv(temp_path / "news_data.csv", index=False)
        
        # Setup signal generator
        signal_generator = AdvancedSignalGenerator()
        
        # Configure backtest
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            symbols=symbols,
            data_directory=temp_dir,
            commission_rate=0.001,  # 0.1% commission
            slippage_rate=0.0005,   # 0.05% slippage
            step_frequency='1h'
        )
        
        # Run backtest
        print("🔄 Initializing backtest engine...")
        backtest_engine = BacktestEngine(config, signal_generator)
        
        print("⚡ Running backtest simulation...")
        results = await backtest_engine.run()
        
        print()
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        
        # Display key metrics
        if results and 'performance_metrics' in results:
            metrics = results['performance_metrics']
            
            print(f"💰 Final Portfolio Value: ${metrics.get('final_value', 0):,.2f}")
            print(f"📈 Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"🎯 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"📋 Total Trades: {metrics.get('total_trades', 0)}")
            
            if metrics.get('total_trades', 0) > 0:
                print(f"💹 Average Trade Return: {metrics.get('average_trade_return', 0)*100:.2f}%")
            
            print()
            
            # Portfolio composition
            if 'final_positions' in results:
                print("🏦 Final Portfolio Positions:")
                for symbol, position in results['final_positions'].items():
                    if position['quantity'] > 0:
                        value = position['quantity'] * position.get('current_price', 0)
                        print(f"  {symbol}: {position['quantity']:.0f} shares (${value:,.2f})")
                
                cash = results.get('final_cash', 0)
                print(f"  Cash: ${cash:,.2f}")
                print()
        
        # Strategy breakdown
        if 'trade_log' in results:
            trades = results['trade_log']
            strategy_performance = {}
            
            for trade in trades:
                strategy = getattr(trade, 'strategy', 'unknown')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'count': 0, 'total_pnl': 0}
                
                strategy_performance[strategy]['count'] += 1
                strategy_performance[strategy]['total_pnl'] += getattr(trade, 'pnl', 0)
            
            print("📊 Strategy Performance Breakdown:")
            for strategy, perf in strategy_performance.items():
                avg_pnl = perf['total_pnl'] / perf['count'] if perf['count'] > 0 else 0
                print(f"  {strategy.title()}: {perf['count']} trades, Avg PnL: ${avg_pnl:.2f}")
            print()
        
        print("✅ Backtest completed successfully!")
        
        return results


if __name__ == "__main__":
    # Run the comprehensive backtest
    try:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(run_comprehensive_backtest())
        
        print("\n🎉 Demo completed! The trading system has been successfully backtested.")
        print("   This demonstrates the complete pipeline from data ingestion to execution.")
        
    except Exception as e:
        print(f"❌ Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc() 
"""
Comprehensive Backtesting Framework Test

Tests the complete backtesting system including historical simulation,
portfolio management, signal generation, and performance analytics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import tempfile
import os

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.backtesting.historical_simulator import HistoricalEventSimulator
from src.backtesting.portfolio_simulator import PortfolioSimulator, FillModel
from src.backtesting.metrics_logger import MetricsLogger
from src.signal_generation.schemas import TradingSignal, SignalAction


class MockSignalGenerator:
    """Mock signal generator for testing"""
    
    def __init__(self, strategy_type: str = "momentum"):
        self.strategy_type = strategy_type
        
    async def generate_signals(self, symbols, market_data, news_data=None, 
                             events=None, features=None, current_time=None):
        """Generate mock signals based on simple strategies"""
        
        signals = []
        
        for symbol in symbols:
            if symbol in market_data and not market_data[symbol].empty:
                df = market_data[symbol]
                
                if len(df) >= 20:  # Need enough data for signal generation
                    
                    if self.strategy_type == "momentum":
                        # Simple momentum strategy
                        short_ma = df['close'].rolling(5).mean().iloc[-1]
                        long_ma = df['close'].rolling(20).mean().iloc[-1]
                        current_price = df['close'].iloc[-1]
                        
                        if short_ma > long_ma and current_price > short_ma * 1.02:
                            signals.append(TradingSignal(
                                symbol=symbol,
                                action=SignalAction.BUY,
                                confidence=0.7,
                                timestamp=current_time or datetime.now(),
                                signal_id=f"momentum_buy_{symbol}_{current_time}",
                                position_size=0.1  # 10% position
                            ))
                        elif short_ma < long_ma and current_price < short_ma * 0.98:
                            signals.append(TradingSignal(
                                symbol=symbol,
                                action=SignalAction.SELL,
                                confidence=0.6,
                                timestamp=current_time or datetime.now(),
                                signal_id=f"momentum_sell_{symbol}_{current_time}",
                                position_size=1.0  # Sell full position
                            ))
                    
                    elif self.strategy_type == "mean_reversion":
                        # Simple mean reversion strategy
                        price = df['close'].iloc[-1]
                        mean_price = df['close'].rolling(20).mean().iloc[-1]
                        std_price = df['close'].rolling(20).std().iloc[-1]
                        
                        if price < mean_price - 1.5 * std_price:
                            signals.append(TradingSignal(
                                symbol=symbol,
                                action=SignalAction.BUY,
                                confidence=0.8,
                                timestamp=current_time or datetime.now(),
                                signal_id=f"reversion_buy_{symbol}_{current_time}",
                                position_size=0.05  # 5% position
                            ))
                        elif price > mean_price + 1.5 * std_price:
                            signals.append(TradingSignal(
                                symbol=symbol,
                                action=SignalAction.SELL,
                                confidence=0.8,
                                timestamp=current_time or datetime.now(),
                                signal_id=f"reversion_sell_{symbol}_{current_time}",
                                position_size=1.0  # Sell full position
                            ))
        
        return signals


def create_mock_market_data(symbols, start_date, end_date, freq='1h'):
    """Create mock market data for testing"""
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    mock_data = {}
    
    for symbol in symbols:
        # Create realistic price movements
        np.random.seed(hash(symbol) % 2**32)  # Consistent but different for each symbol
        
        num_periods = len(date_range)
        base_price = np.random.uniform(50, 200)  # Random starting price
        
        # Generate price movements using random walk with drift
        returns = np.random.normal(0.0002, 0.02, num_periods)  # Small positive drift
        
        # Add some volatility clustering
        volatility = np.abs(np.random.normal(0.02, 0.01, num_periods))
        returns = returns * volatility
        
        # Calculate prices
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        df_data = []
        for i, (timestamp, price) in enumerate(zip(date_range, prices)):
            # Generate OHLC from close price
            daily_vol = volatility[i]
            high = price * (1 + abs(np.random.normal(0, daily_vol)))
            low = price * (1 - abs(np.random.normal(0, daily_vol)))
            open_price = prices[i-1] if i > 0 else price
            
            # Ensure OHLC consistency
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            volume = np.random.randint(100000, 1000000)
            
            df_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        mock_data[symbol] = pd.DataFrame(df_data, index=date_range)
    
    return mock_data


def create_mock_news_data(start_date, end_date, symbols):
    """Create mock news data for testing"""
    
    news_data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Generate 1-3 news articles per day
        num_articles = np.random.randint(1, 4)
        
        for _ in range(num_articles):
            # Random time during the day
            article_time = current_date + timedelta(
                hours=np.random.randint(9, 17),
                minutes=np.random.randint(0, 60)
            )
            
            # Choose random symbol or general market news
            if np.random.random() < 0.7 and symbols:
                target_symbol = np.random.choice(symbols)
                title = f"{target_symbol} reports quarterly earnings beat estimates"
                description = f"{target_symbol} announced strong quarterly results..."
            else:
                target_symbol = None
                title = "Federal Reserve signals potential rate changes"
                description = "The Federal Reserve indicated potential monetary policy adjustments..."
            
            news_data.append({
                'title': title,
                'description': description,
                'published_at': article_time,
                'target_symbol': target_symbol,
                'source': 'Mock News API'
            })
        
        current_date += timedelta(days=1)
    
    return news_data


@pytest.fixture
def mock_data_setup():
    """Setup mock data for testing"""
    
    symbols = ['AAPL', 'TSLA', 'MSFT']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)  # 3 months of data
    
    market_data = create_mock_market_data(symbols, start_date, end_date)
    news_data = create_mock_news_data(start_date, end_date, symbols)
    
    return {
        'symbols': symbols,
        'start_date': start_date,
        'end_date': end_date,
        'market_data': market_data,
        'news_data': news_data
    }


@pytest.fixture
def temp_data_dir(mock_data_setup):
    """Create temporary directory with mock data files"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save mock market data
        for symbol, data in mock_data_setup['market_data'].items():
            file_path = os.path.join(temp_dir, f"{symbol}_market_data.csv")
            data.to_csv(file_path)
        
        # Save mock news data
        news_df = pd.DataFrame(mock_data_setup['news_data'])
        news_file = os.path.join(temp_dir, "historical_news.csv")
        news_df.to_csv(news_file, index=False)
        
        yield temp_dir


class TestHistoricalEventSimulator:
    """Test the historical event simulator"""
    
    @pytest.mark.asyncio
    async def test_simulator_initialization(self, mock_data_setup):
        """Test simulator initialization"""
        
        simulator = HistoricalEventSimulator(
            start_date=mock_data_setup['start_date'],
            end_date=mock_data_setup['end_date'],
            symbols=mock_data_setup['symbols'],
            time_resolution='1h'
        )
        
        assert simulator.start_date == mock_data_setup['start_date']
        assert simulator.end_date == mock_data_setup['end_date']
        assert simulator.symbols == mock_data_setup['symbols']
        assert simulator.time_resolution == '1h'
    
    @pytest.mark.asyncio
    async def test_load_data_from_files(self, mock_data_setup, temp_data_dir):
        """Test loading data from files"""
        
        simulator = HistoricalEventSimulator(
            start_date=mock_data_setup['start_date'],
            end_date=mock_data_setup['end_date'],
            symbols=mock_data_setup['symbols']
        )
        
        await simulator.load_historical_data(data_directory=temp_data_dir)
        
        # Check that data was loaded
        assert len(simulator.historical_market_data) > 0
        assert len(simulator.historical_news) > 0
        assert len(simulator.preprocessed_events) > 0
        
        # Check data content
        for symbol in mock_data_setup['symbols']:
            assert symbol in simulator.historical_market_data
            assert not simulator.historical_market_data[symbol].empty
    
    @pytest.mark.asyncio
    async def test_simulation_execution(self, mock_data_setup, temp_data_dir):
        """Test running the simulation"""
        
        simulator = HistoricalEventSimulator(
            start_date=mock_data_setup['start_date'],
            end_date=mock_data_setup['start_date'] + timedelta(days=7),  # Short test
            symbols=['AAPL'],  # Single symbol for speed
            time_resolution='1h'
        )
        
        await simulator.load_historical_data(data_directory=temp_data_dir)
        
        simulation_steps = 0
        for current_time, state in simulator.simulate():
            simulation_steps += 1
            
            # Verify state structure
            assert isinstance(state.current_time, datetime)
            assert isinstance(state.available_market_data, dict)
            assert isinstance(state.available_news, list)
            assert isinstance(state.detected_events, list)
            
            # Stop after reasonable number of steps for testing
            if simulation_steps > 50:
                break
        
        assert simulation_steps > 0


class TestPortfolioSimulator:
    """Test the portfolio simulator"""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        
        portfolio = PortfolioSimulator(
            initial_capital=100000.0,
            commission_rate=0.001,
            fill_model=FillModel.REALISTIC
        )
        
        assert portfolio.initial_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.commission_rate == 0.001
        assert portfolio.fill_model == FillModel.REALISTIC
        assert len(portfolio.positions) == 0
    
    def test_market_data_update(self):
        """Test market data updates"""
        
        portfolio = PortfolioSimulator(initial_capital=100000.0)
        
        market_data = {
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 152.0,
            'volume': 1000000
        }
        
        portfolio.update_market_data('AAPL', market_data)
        
        assert 'AAPL' in portfolio.current_market_data
        assert portfolio.current_market_data['AAPL']['close'] == 152.0
    
    def test_signal_processing(self):
        """Test signal processing and order execution"""
        
        portfolio = PortfolioSimulator(
            initial_capital=100000.0,
            fill_model=FillModel.IMMEDIATE  # Use immediate fill for testing
        )
        
        # Update market data
        portfolio.update_market_data('AAPL', {
            'open': 150.0, 'high': 155.0, 'low': 148.0, 'close': 152.0, 'volume': 1000000
        })
        
        # Create buy signal
        buy_signal = TradingSignal(
            symbol='AAPL',
            action=SignalAction.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            signal_id='test_buy_1',
            position_size=0.1  # 10% of portfolio
        )
        
        # Process signal
        order = portfolio.process_signal(buy_signal, datetime.now())
        
        assert order is not None
        assert order.symbol == 'AAPL'
        assert order.side.value == 'buy'
        
        # Check that position was created and cash reduced
        assert 'AAPL' in portfolio.positions
        assert portfolio.cash < 100000.0
        assert len(portfolio.completed_transactions) > 0
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        
        portfolio = PortfolioSimulator(initial_capital=100000.0)
        
        # Simulate some portfolio activity
        portfolio.cash = 95000.0
        portfolio.portfolio_history = [
            type('PortfolioState', (), {
                'total_value': 100000.0,
                'timestamp': datetime.now() - timedelta(days=1)
            })(),
            type('PortfolioState', (), {
                'total_value': 105000.0,
                'timestamp': datetime.now()
            })()
        ]
        
        metrics = portfolio.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'total_trades' in metrics
        assert 'current_value' in metrics


class TestMetricsLogger:
    """Test the metrics logger"""
    
    def test_metrics_logger_initialization(self):
        """Test metrics logger initialization"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MetricsLogger(
                benchmark_symbol="SPY",
                output_directory=temp_dir
            )
            
            assert logger.benchmark_symbol == "SPY"
            assert str(logger.output_directory) == temp_dir
            assert len(logger.portfolio_history) == 0
    
    def test_performance_calculation(self):
        """Test performance metrics calculation"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MetricsLogger(output_directory=temp_dir)
            
            # Add mock portfolio history
            from src.backtesting.portfolio_simulator import PortfolioState
            
            states = []
            for i in range(100):
                # Simulate growing portfolio
                value = 100000 * (1 + i * 0.001)  # 0.1% growth per period
                
                state = PortfolioState(
                    timestamp=datetime.now() - timedelta(days=100-i),
                    cash=value * 0.1,
                    total_value=value,
                    positions={},
                    day_pnl=value * 0.001 if i > 0 else 0,
                    total_pnl=value - 100000,
                    drawdown=0,
                    max_drawdown=0,
                    transactions_today=0
                )
                
                logger.log_portfolio_state(state)
            
            # Calculate metrics
            metrics = logger.calculate_performance_metrics()
            
            assert metrics.total_return > 0
            assert metrics.annualized_return > 0
            assert metrics.sharpe_ratio >= 0
            assert metrics.volatility >= 0


class TestBacktestEngine:
    """Test the complete backtest engine"""
    
    @pytest.mark.asyncio
    async def test_backtest_engine_initialization(self, mock_data_setup):
        """Test backtest engine initialization"""
        
        config = BacktestConfig(
            start_date=mock_data_setup['start_date'],
            end_date=mock_data_setup['end_date'],
            symbols=mock_data_setup['symbols'],
            initial_capital=100000.0,
            use_live_data=False
        )
        
        signal_generator = MockSignalGenerator(strategy_type="momentum")
        
        engine = BacktestEngine(
            config=config,
            signal_generator=signal_generator
        )
        
        assert engine.config == config
        assert engine.signal_generator == signal_generator
        assert isinstance(engine.historical_simulator, HistoricalEventSimulator)
        assert isinstance(engine.portfolio_simulator, PortfolioSimulator)
        assert isinstance(engine.metrics_logger, MetricsLogger)
    
    @pytest.mark.asyncio
    async def test_complete_backtest_execution(self, mock_data_setup, temp_data_dir):
        """Test complete backtest execution"""
        
        with tempfile.TemporaryDirectory() as output_dir:
            config = BacktestConfig(
                start_date=mock_data_setup['start_date'],
                end_date=mock_data_setup['start_date'] + timedelta(days=30),  # 1 month test
                symbols=['AAPL'],  # Single symbol for speed
                initial_capital=100000.0,
                use_live_data=False,
                data_directory=temp_data_dir,
                output_directory=output_dir,
                time_resolution='1d'  # Daily for faster testing
            )
            
            signal_generator = MockSignalGenerator(strategy_type="momentum")
            
            engine = BacktestEngine(
                config=config,
                signal_generator=signal_generator
            )
            
            # Run backtest
            results = await engine.run_backtest()
            
            # Verify results
            assert isinstance(results.execution_time, float)
            assert results.execution_time > 0
            assert 'total_return' in results.performance_metrics
            assert len(results.portfolio_history) > 0
            assert os.path.exists(results.report_path)
            
            # Check that files were created
            assert os.path.exists(os.path.join(output_dir, "backtest_20230101_20230131.html"))
    
    @pytest.mark.asyncio
    async def test_different_strategies(self, mock_data_setup, temp_data_dir):
        """Test different trading strategies"""
        
        strategies = ["momentum", "mean_reversion"]
        results = {}
        
        for strategy in strategies:
            with tempfile.TemporaryDirectory() as output_dir:
                config = BacktestConfig(
                    start_date=mock_data_setup['start_date'],
                    end_date=mock_data_setup['start_date'] + timedelta(days=14),  # 2 weeks
                    symbols=['AAPL'],
                    initial_capital=100000.0,
                    use_live_data=False,
                    data_directory=temp_data_dir,
                    output_directory=output_dir,
                    time_resolution='1d'
                )
                
                signal_generator = MockSignalGenerator(strategy_type=strategy)
                engine = BacktestEngine(config=config, signal_generator=signal_generator)
                
                result = await engine.run_backtest()
                results[strategy] = result.performance_metrics['total_return']
        
        # Verify that we got results for both strategies
        assert 'momentum' in results
        assert 'mean_reversion' in results
        
        # Results should be different (even if slightly due to randomness)
        assert isinstance(results['momentum'], float)
        assert isinstance(results['mean_reversion'], float)


@pytest.mark.asyncio
async def test_backtesting_integration():
    """Integration test for the complete backtesting system"""
    
    print("\n🚀 Running Comprehensive Backtesting Integration Test...")
    
    # Create test configuration
    symbols = ['AAPL', 'TSLA']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 2, 28)  # 2 months
    
    # Create mock data
    market_data = create_mock_market_data(symbols, start_date, end_date, freq='1d')
    news_data = create_mock_news_data(start_date, end_date, symbols)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save mock data
        for symbol, data in market_data.items():
            data.to_csv(os.path.join(temp_dir, f"{symbol}_market_data.csv"))
        
        news_df = pd.DataFrame(news_data)
        news_df.to_csv(os.path.join(temp_dir, "historical_news.csv"), index=False)
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_capital=100000.0,
            commission_rate=0.001,
            max_position_size=0.2,  # 20% max position
            fill_model=FillModel.REALISTIC,
            slippage_rate=0.0005,
            time_resolution='1d',
            use_live_data=False,
            data_directory=temp_dir,
            output_directory=temp_dir
        )
        
        # Test momentum strategy
        print("  📈 Testing Momentum Strategy...")
        momentum_generator = MockSignalGenerator(strategy_type="momentum")
        momentum_engine = BacktestEngine(config=config, signal_generator=momentum_generator)
        momentum_results = await momentum_engine.run_backtest()
        
        print(f"    Return: {momentum_results.performance_metrics['total_return']:.2%}")
        print(f"    Sharpe: {momentum_results.performance_metrics['sharpe_ratio']:.2f}")
        print(f"    Trades: {momentum_results.performance_metrics['total_trades']}")
        
        # Test mean reversion strategy
        print("  📉 Testing Mean Reversion Strategy...")
        reversion_generator = MockSignalGenerator(strategy_type="mean_reversion")
        reversion_engine = BacktestEngine(config=config, signal_generator=reversion_generator)
        reversion_results = await reversion_engine.run_backtest()
        
        print(f"    Return: {reversion_results.performance_metrics['total_return']:.2%}")
        print(f"    Sharpe: {reversion_results.performance_metrics['sharpe_ratio']:.2f}")
        print(f"    Trades: {reversion_results.performance_metrics['total_trades']}")
        
        # Verify results
        assert momentum_results.execution_time > 0
        assert reversion_results.execution_time > 0
        assert len(momentum_results.portfolio_history) > 0
        assert len(reversion_results.portfolio_history) > 0
        
        print("  ✅ Integration test completed successfully!")
        print(f"  📊 Reports generated: {momentum_results.report_path}")


if __name__ == "__main__":
    # Run the integration test directly
    asyncio.run(test_backtesting_integration()) 
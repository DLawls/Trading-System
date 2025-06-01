"""
Backtest Engine

Main orchestrator for comprehensive backtesting that integrates
historical simulation, portfolio management, signal generation, and performance analytics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import asyncio

from .historical_simulator import HistoricalEventSimulator, SimulationState
from .portfolio_simulator import PortfolioSimulator, FillModel
from .metrics_logger import MetricsLogger
from ..signal_generation.main import SignalGenerator
from ..signal_generation.schemas import TradingSignal
from ..data_ingestion.market_data import MarketDataIngestor
from ..data_ingestion.news import NewsIngestor


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    # Time range
    start_date: datetime
    end_date: datetime
    
    # Assets
    symbols: List[str]
    benchmark_symbol: str = "SPY"
    
    # Portfolio settings
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    max_position_size: float = 0.1
    fill_model: FillModel = FillModel.REALISTIC
    slippage_rate: float = 0.0005
    
    # Simulation settings
    time_resolution: str = '1h'  # '1min', '5min', '1h', '1d'
    lookback_days: int = 30
    
    # Data sources
    use_live_data: bool = False
    data_directory: Optional[str] = None
    
    # Output
    output_directory: str = "backtest_results"
    save_state_history: bool = False


@dataclass
class BacktestResults:
    """Results from a completed backtest"""
    config: BacktestConfig
    performance_metrics: Dict[str, Any]
    portfolio_history: List[Any]
    transaction_history: List[Any]
    report_path: str
    execution_time: float


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the complete process:
    1. Historical data simulation
    2. Signal generation
    3. Portfolio execution
    4. Performance analytics
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        signal_generator: Optional[SignalGenerator] = None,
        alpaca_api_key: Optional[str] = None,
        alpaca_api_secret: Optional[str] = None,
        alpaca_base_url: Optional[str] = None,
        news_api_key: Optional[str] = None
    ):
        self.config = config
        self.signal_generator = signal_generator or SignalGenerator()
        
        # API credentials for live data
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_api_secret = alpaca_api_secret
        self.alpaca_base_url = alpaca_base_url
        self.news_api_key = news_api_key
        
        # Core components
        self.historical_simulator = HistoricalEventSimulator(
            start_date=config.start_date,
            end_date=config.end_date,
            symbols=config.symbols,
            lookback_days=config.lookback_days,
            time_resolution=config.time_resolution
        )
        
        self.portfolio_simulator = PortfolioSimulator(
            initial_capital=config.initial_capital,
            commission_rate=config.commission_rate,
            max_position_size=config.max_position_size,
            fill_model=config.fill_model,
            slippage_rate=config.slippage_rate
        )
        
        self.metrics_logger = MetricsLogger(
            benchmark_symbol=config.benchmark_symbol,
            output_directory=config.output_directory
        )
        
        # State tracking
        self.is_running = False
        self.current_time: Optional[datetime] = None
        self.state_history: List[Tuple[datetime, SimulationState]] = []
        
        logger.info(f"Initialized BacktestEngine: {config.start_date} to {config.end_date}")
    
    async def run_backtest(self) -> BacktestResults:
        """
        Execute the complete backtesting process
        
        Returns:
            BacktestResults with comprehensive performance data
        """
        
        start_time = datetime.now()
        logger.info("Starting backtest execution...")
        
        try:
            # 1. Load historical data
            await self._load_historical_data()
            
            # 2. Load benchmark data for comparison
            await self._load_benchmark_data()
            
            # 3. Run the main simulation loop
            await self._run_simulation()
            
            # 4. Calculate performance metrics
            performance_metrics = self.metrics_logger.calculate_performance_metrics()
            
            # 5. Generate comprehensive report
            report_path = self.metrics_logger.generate_report(
                f"backtest_{self.config.start_date.strftime('%Y%m%d')}_{self.config.end_date.strftime('%Y%m%d')}"
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 6. Compile results
            results = BacktestResults(
                config=self.config,
                performance_metrics=performance_metrics.__dict__,
                portfolio_history=[state.__dict__ for state in self.portfolio_simulator.portfolio_history],
                transaction_history=[txn.__dict__ for txn in self.portfolio_simulator.completed_transactions],
                report_path=report_path,
                execution_time=execution_time
            )
            
            logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            logger.info(f"Total Return: {performance_metrics.total_return:.2%}")
            logger.info(f"Sharpe Ratio: {performance_metrics.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {performance_metrics.max_drawdown:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        
        finally:
            self.is_running = False
    
    async def _load_historical_data(self) -> None:
        """Load historical market data and news"""
        
        logger.info("Loading historical data...")
        
        market_data_source = None
        news_data_source = None
        
        if self.config.use_live_data:
            # Use live API data sources
            if all([self.alpaca_api_key, self.alpaca_api_secret, self.alpaca_base_url]):
                market_data_source = MarketDataIngestor(
                    api_key=self.alpaca_api_key,
                    api_secret=self.alpaca_api_secret,
                    base_url=self.alpaca_base_url
                )
            
            if self.news_api_key:
                news_data_source = NewsIngestor(api_key=self.news_api_key)
        
        # Load data into historical simulator
        await self.historical_simulator.load_historical_data(
            market_data_source=market_data_source,
            news_data_source=news_data_source,
            data_directory=self.config.data_directory
        )
        
        logger.info("Historical data loaded successfully")
    
    async def _load_benchmark_data(self) -> None:
        """Load benchmark data for performance comparison"""
        
        if self.config.use_live_data and self.alpaca_api_key:
            try:
                market_data_source = MarketDataIngestor(
                    api_key=self.alpaca_api_key,
                    api_secret=self.alpaca_api_secret,
                    base_url=self.alpaca_base_url
                )
                
                from alpaca.data.timeframe import TimeFrame
                
                # Map time resolution
                timeframe_map = {
                    '1min': TimeFrame.Minute,
                    '5min': TimeFrame(5, TimeFrame.Unit.Minute),
                    '1h': TimeFrame.Hour,
                    '1d': TimeFrame.Day
                }
                timeframe = timeframe_map.get(self.config.time_resolution, TimeFrame.Hour)
                
                # Load benchmark data
                benchmark_data = await market_data_source.get_latest_bars(
                    symbols=[self.config.benchmark_symbol],
                    timeframe=timeframe,
                    limit=10000
                )
                
                if self.config.benchmark_symbol in benchmark_data:
                    # Filter to backtest date range
                    df = benchmark_data[self.config.benchmark_symbol]
                    mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
                    filtered_df = df[mask]
                    
                    if not filtered_df.empty:
                        self.metrics_logger.load_benchmark_data(filtered_df)
                        logger.info(f"Loaded benchmark data for {self.config.benchmark_symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to load benchmark data: {e}")
    
    async def _run_simulation(self) -> None:
        """Run the main simulation loop"""
        
        logger.info("Starting simulation loop...")
        self.is_running = True
        
        simulation_steps = 0
        signals_generated = 0
        orders_executed = 0
        
        # Run the historical simulation
        for current_time, simulation_state in self.historical_simulator.simulate():
            
            if not self.is_running:
                break
            
            self.current_time = current_time
            simulation_steps += 1
            
            # Update portfolio with latest market data
            self._update_portfolio_market_data(simulation_state)
            
            # Generate trading signals based on current state
            signals = await self._generate_signals(simulation_state)
            signals_generated += len(signals)
            
            # Execute signals through portfolio simulator
            for signal in signals:
                order = self.portfolio_simulator.process_signal(signal, current_time)
                if order and order.status.value == 'filled':
                    orders_executed += 1
                    
                    # Log transaction
                    if self.portfolio_simulator.completed_transactions:
                        latest_txn = self.portfolio_simulator.completed_transactions[-1]
                        self.metrics_logger.log_transaction(latest_txn)
            
            # Update portfolio state and log for metrics
            portfolio_state = self.portfolio_simulator.update_portfolio_state(current_time)
            self.metrics_logger.log_portfolio_state(portfolio_state)
            
            # Optional: Save detailed state history
            if self.config.save_state_history:
                self.state_history.append((current_time, simulation_state))
            
            # Progress logging
            if simulation_steps % 1000 == 0:
                logger.debug(f"Processed {simulation_steps} simulation steps")
        
        logger.info(f"Simulation completed: {simulation_steps} steps, {signals_generated} signals, {orders_executed} orders executed")
    
    def _update_portfolio_market_data(self, simulation_state: SimulationState) -> None:
        """Update portfolio simulator with latest market data"""
        
        for symbol, market_data_df in simulation_state.available_market_data.items():
            if not market_data_df.empty:
                # Get the latest market data point
                latest_data = market_data_df.iloc[-1].to_dict()
                
                # Update portfolio simulator
                self.portfolio_simulator.update_market_data(symbol, latest_data)
    
    async def _generate_signals(self, simulation_state: SimulationState) -> List[TradingSignal]:
        """Generate trading signals based on current simulation state"""
        
        try:
            # Prepare data for signal generation
            market_data = simulation_state.available_market_data
            news_data = pd.DataFrame(simulation_state.available_news) if simulation_state.available_news else pd.DataFrame()
            events = simulation_state.detected_events
            features = simulation_state.features
            
            # Generate signals
            signals = await self.signal_generator.generate_signals(
                symbols=self.config.symbols,
                market_data=market_data,
                news_data=news_data,
                events=events,
                features=features,
                current_time=simulation_state.current_time
            )
            
            return signals
            
        except Exception as e:
            logger.warning(f"Failed to generate signals: {e}")
            return []
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics during backtest execution"""
        
        if not self.portfolio_simulator.portfolio_history:
            return {}
        
        return self.portfolio_simulator.get_performance_metrics()
    
    def get_current_positions(self) -> pd.DataFrame:
        """Get current portfolio positions"""
        
        return self.portfolio_simulator.get_positions_summary()
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history"""
        
        return self.portfolio_simulator.get_transaction_history()
    
    def stop_backtest(self) -> None:
        """Stop the backtest execution"""
        
        self.is_running = False
        logger.info("Backtest stop requested")
    
    async def run_walk_forward_analysis(
        self,
        training_window_days: int = 252,
        testing_window_days: int = 63,
        step_size_days: int = 21
    ) -> List[BacktestResults]:
        """
        Run walk-forward analysis with multiple training/testing periods
        
        Args:
            training_window_days: Days of data for training
            testing_window_days: Days of data for testing
            step_size_days: Days to step forward for each iteration
            
        Returns:
            List of BacktestResults for each walk-forward period
        """
        
        logger.info(f"Starting walk-forward analysis...")
        
        results = []
        current_start = self.config.start_date
        
        while current_start + timedelta(days=training_window_days + testing_window_days) <= self.config.end_date:
            
            training_end = current_start + timedelta(days=training_window_days)
            testing_start = training_end
            testing_end = testing_start + timedelta(days=testing_window_days)
            
            logger.info(f"Walk-forward period: Training {current_start} to {training_end}, Testing {testing_start} to {testing_end}")
            
            # Create config for this period
            period_config = BacktestConfig(
                start_date=testing_start,
                end_date=testing_end,
                symbols=self.config.symbols,
                benchmark_symbol=self.config.benchmark_symbol,
                initial_capital=self.config.initial_capital,
                commission_rate=self.config.commission_rate,
                max_position_size=self.config.max_position_size,
                fill_model=self.config.fill_model,
                slippage_rate=self.config.slippage_rate,
                time_resolution=self.config.time_resolution,
                lookback_days=self.config.lookback_days,
                use_live_data=self.config.use_live_data,
                data_directory=self.config.data_directory,
                output_directory=f"{self.config.output_directory}/walk_forward",
                save_state_history=False
            )
            
            # Create engine for this period
            period_engine = BacktestEngine(
                config=period_config,
                signal_generator=self.signal_generator,
                alpaca_api_key=self.alpaca_api_key,
                alpaca_api_secret=self.alpaca_api_secret,
                alpaca_base_url=self.alpaca_base_url,
                news_api_key=self.news_api_key
            )
            
            # Run backtest for this period
            try:
                period_results = await period_engine.run_backtest()
                results.append(period_results)
                
            except Exception as e:
                logger.error(f"Walk-forward period failed: {e}")
            
            # Step forward
            current_start += timedelta(days=step_size_days)
        
        logger.info(f"Walk-forward analysis completed: {len(results)} periods")
        
        # Aggregate walk-forward results
        await self._generate_walk_forward_report(results)
        
        return results
    
    async def _generate_walk_forward_report(self, results: List[BacktestResults]) -> None:
        """Generate aggregated walk-forward analysis report"""
        
        if not results:
            return
        
        # Calculate aggregate metrics
        total_returns = [r.performance_metrics['total_return'] for r in results]
        sharpe_ratios = [r.performance_metrics['sharpe_ratio'] for r in results]
        max_drawdowns = [r.performance_metrics['max_drawdown'] for r in results]
        
        aggregate_stats = {
            'num_periods': len(results),
            'avg_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_period_return': min(total_returns),
            'best_period_return': max(total_returns),
            'positive_periods': sum(1 for r in total_returns if r > 0),
            'consistency_ratio': sum(1 for r in total_returns if r > 0) / len(total_returns)
        }
        
        logger.info("Walk-Forward Analysis Summary:")
        logger.info(f"  Periods: {aggregate_stats['num_periods']}")
        logger.info(f"  Avg Return: {aggregate_stats['avg_return']:.2%}")
        logger.info(f"  Avg Sharpe: {aggregate_stats['avg_sharpe']:.2f}")
        logger.info(f"  Consistency: {aggregate_stats['consistency_ratio']:.2%}")
        
        # Save detailed walk-forward report
        import json
        
        walk_forward_path = f"{self.config.output_directory}/walk_forward_analysis.json"
        with open(walk_forward_path, 'w') as f:
            json.dump({
                'aggregate_stats': aggregate_stats,
                'period_results': [r.__dict__ for r in results]
            }, f, indent=2, default=str)
        
        logger.info(f"Walk-forward analysis report saved: {walk_forward_path}")
    
    def reset(self) -> None:
        """Reset all components to initial state"""
        
        self.historical_simulator.reset()
        self.portfolio_simulator.reset()
        self.metrics_logger.reset()
        self.state_history = []
        self.current_time = None
        self.is_running = False
        
        logger.info("BacktestEngine reset to initial state") 
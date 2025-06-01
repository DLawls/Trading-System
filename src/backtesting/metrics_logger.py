"""
Metrics Logger

Comprehensive performance analytics and reporting for backtesting results.
Tracks P&L, Sharpe ratio, drawdown, alpha/beta analysis, and generates reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
from loguru import logger

from .portfolio_simulator import PortfolioState, Transaction


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95%)
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    
    # Other metrics
    calmar_ratio: float
    information_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


@dataclass
class PeriodAnalysis:
    """Analysis for a specific time period"""
    period: str
    start_date: datetime
    end_date: datetime
    returns: List[float]
    metrics: PerformanceMetrics


class MetricsLogger:
    """
    Comprehensive performance analytics and reporting system
    """
    
    def __init__(
        self,
        benchmark_symbol: str = "SPY",
        risk_free_rate: float = 0.02,
        output_directory: str = "backtest_results"
    ):
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Performance tracking
        self.portfolio_history: List[PortfolioState] = []
        self.transaction_history: List[Transaction] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        # Analysis results
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self.period_analyses: Dict[str, PeriodAnalysis] = {}
        
        logger.info(f"Initialized MetricsLogger with benchmark: {benchmark_symbol}")
    
    def log_portfolio_state(self, state: PortfolioState) -> None:
        """Log a portfolio state snapshot"""
        self.portfolio_history.append(state)
        
        # Calculate daily return
        if len(self.portfolio_history) > 1:
            previous_value = self.portfolio_history[-2].total_value
            daily_return = (state.total_value - previous_value) / previous_value
            self.daily_returns.append(daily_return)
    
    def log_transaction(self, transaction: Transaction) -> None:
        """Log a completed transaction"""
        self.transaction_history.append(transaction)
    
    def load_benchmark_data(self, benchmark_data: pd.DataFrame) -> None:
        """Load benchmark data for comparison"""
        self.benchmark_data = benchmark_data.copy()
        
        # Calculate benchmark returns
        if 'close' in benchmark_data.columns:
            self.benchmark_returns = benchmark_data['close'].pct_change().dropna().tolist()
        
        logger.info(f"Loaded benchmark data: {len(benchmark_data)} periods")
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not self.portfolio_history:
            logger.warning("No portfolio history available for metrics calculation")
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, cumulative_return=0.0,
                volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, var_95=0.0, win_rate=0.0, profit_factor=0.0,
                avg_win=0.0, avg_loss=0.0, total_trades=0, calmar_ratio=0.0,
                information_ratio=0.0
            )
        
        # Portfolio values
        initial_value = self.portfolio_history[0].total_value
        final_value = self.portfolio_history[-1].total_value
        
        # Calculate returns
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate time period in years
        time_period = (self.portfolio_history[-1].timestamp - self.portfolio_history[0].timestamp).days / 365.25
        annualized_return = (1 + total_return) ** (1 / max(time_period, 1/252)) - 1
        
        # Risk metrics
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio
            risk_free_daily = self.risk_free_rate / 252
            excess_returns = returns_array - risk_free_daily
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
            sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns_array, 5)
            
        else:
            volatility = sharpe_ratio = sortino_ratio = var_95 = 0.0
        
        # Drawdown metrics
        max_drawdown = max([state.max_drawdown for state in self.portfolio_history])
        
        # Trading metrics
        win_trades = [t for t in self.transaction_history if t.side.value == 'sell' and self._is_profitable_trade(t)]
        loss_trades = [t for t in self.transaction_history if t.side.value == 'sell' and not self._is_profitable_trade(t)]
        
        total_trades = len([t for t in self.transaction_history if t.side.value == 'sell'])
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0.0
        
        # Profit metrics
        total_wins = sum([self._calculate_trade_pnl(t) for t in win_trades])
        total_losses = abs(sum([self._calculate_trade_pnl(t) for t in loss_trades]))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
        avg_win = total_wins / len(win_trades) if len(win_trades) > 0 else 0.0
        avg_loss = total_losses / len(loss_trades) if len(loss_trades) > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Information ratio (vs benchmark)
        information_ratio = self._calculate_information_ratio()
        
        # Alpha and Beta (vs benchmark)
        alpha, beta = self._calculate_alpha_beta()
        
        self.performance_metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=total_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            alpha=alpha,
            beta=beta
        )
        
        return self.performance_metrics
    
    def _is_profitable_trade(self, transaction: Transaction) -> bool:
        """Determine if a sell transaction was profitable"""
        # This is simplified - in practice, you'd need to match with corresponding buy
        return transaction.total_cost < 0  # Negative cost means proceeds (profit)
    
    def _calculate_trade_pnl(self, transaction: Transaction) -> float:
        """Calculate P&L for a transaction"""
        # Simplified - would need proper trade matching in practice
        return -transaction.total_cost if transaction.side.value == 'sell' else 0.0
    
    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio vs benchmark"""
        
        if not self.benchmark_returns or len(self.daily_returns) != len(self.benchmark_returns):
            return 0.0
        
        portfolio_returns = np.array(self.daily_returns)
        benchmark_returns = np.array(self.benchmark_returns[:len(self.daily_returns)])
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error > 0:
            return np.mean(excess_returns) / tracking_error * np.sqrt(252)
        
        return 0.0
    
    def _calculate_alpha_beta(self) -> Tuple[Optional[float], Optional[float]]:
        """Calculate alpha and beta vs benchmark"""
        
        if not self.benchmark_returns or len(self.daily_returns) < 30:
            return None, None
        
        portfolio_returns = np.array(self.daily_returns)
        benchmark_returns = np.array(self.benchmark_returns[:len(self.daily_returns)])
        
        # Calculate beta using linear regression
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
            
            # Calculate alpha
            risk_free_daily = self.risk_free_rate / 252
            portfolio_mean = np.mean(portfolio_returns)
            benchmark_mean = np.mean(benchmark_returns)
            
            alpha = (portfolio_mean - risk_free_daily) - beta * (benchmark_mean - risk_free_daily)
            alpha_annualized = alpha * 252
            
            return alpha_annualized, beta
        
        return None, None
    
    def analyze_periods(self) -> Dict[str, PeriodAnalysis]:
        """Analyze performance by different time periods"""
        
        if not self.portfolio_history:
            return {}
        
        periods = {
            'monthly': self._analyze_monthly_performance(),
            'quarterly': self._analyze_quarterly_performance(),
            'yearly': self._analyze_yearly_performance()
        }
        
        self.period_analyses = periods
        return periods
    
    def _analyze_monthly_performance(self) -> List[PeriodAnalysis]:
        """Analyze monthly performance"""
        
        monthly_data = {}
        
        for i, state in enumerate(self.portfolio_history):
            month_key = state.timestamp.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'start_date': state.timestamp,
                    'start_value': state.total_value,
                    'end_date': state.timestamp,
                    'end_value': state.total_value
                }
            else:
                monthly_data[month_key]['end_date'] = state.timestamp
                monthly_data[month_key]['end_value'] = state.total_value
        
        monthly_analyses = []
        for month, data in monthly_data.items():
            monthly_return = (data['end_value'] - data['start_value']) / data['start_value']
            
            # Create simplified metrics for the period
            metrics = PerformanceMetrics(
                total_return=monthly_return,
                annualized_return=monthly_return * 12,  # Rough annualization
                cumulative_return=monthly_return,
                volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, var_95=0.0, win_rate=0.0, profit_factor=0.0,
                avg_win=0.0, avg_loss=0.0, total_trades=0, calmar_ratio=0.0,
                information_ratio=0.0
            )
            
            analysis = PeriodAnalysis(
                period=f"Month {month}",
                start_date=data['start_date'],
                end_date=data['end_date'],
                returns=[monthly_return],
                metrics=metrics
            )
            monthly_analyses.append(analysis)
        
        return monthly_analyses
    
    def _analyze_quarterly_performance(self) -> List[PeriodAnalysis]:
        """Analyze quarterly performance"""
        # Similar to monthly but grouped by quarters
        return []  # Simplified for now
    
    def _analyze_yearly_performance(self) -> List[PeriodAnalysis]:
        """Analyze yearly performance"""
        # Similar to monthly but grouped by years
        return []  # Simplified for now
    
    def generate_report(self, report_name: str = None) -> str:
        """Generate comprehensive performance report"""
        
        if report_name is None:
            report_name = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate metrics if not already done
        if self.performance_metrics is None:
            self.calculate_performance_metrics()
        
        # Generate HTML report
        report_path = self.output_directory / f"{report_name}.html"
        
        html_content = self._generate_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Generate plots
        self._generate_performance_plots(report_name)
        
        # Save raw data
        self._save_raw_data(report_name)
        
        logger.info(f"Generated comprehensive report: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self) -> str:
        """Generate HTML performance report"""
        
        metrics = self.performance_metrics
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #007acc; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f0f0f0; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Strategy Backtest Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Benchmark: {self.benchmark_symbol}</p>
            </div>
            
            <h2>Key Performance Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value {'positive' if metrics.total_return > 0 else 'negative'}">{metrics.total_return:.2%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if metrics.annualized_return > 0 else 'negative'}">{metrics.annualized_return:.2%}</div>
                    <div class="metric-label">Annualized Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value negative">{metrics.max_drawdown:.2%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.volatility:.2%}</div>
                    <div class="metric-label">Volatility</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.win_rate:.2%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>
            
            <h2>Detailed Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td class="{'positive' if metrics.total_return > 0 else 'negative'}">{metrics.total_return:.2%}</td></tr>
                <tr><td>Annualized Return</td><td class="{'positive' if metrics.annualized_return > 0 else 'negative'}">{metrics.annualized_return:.2%}</td></tr>
                <tr><td>Volatility</td><td>{metrics.volatility:.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{metrics.sharpe_ratio:.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">{metrics.max_drawdown:.2%}</td></tr>
                <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
                <tr><td>VaR (95%)</td><td class="negative">{metrics.var_95:.2%}</td></tr>
                <tr><td>Win Rate</td><td>{metrics.win_rate:.2%}</td></tr>
                <tr><td>Profit Factor</td><td>{metrics.profit_factor:.2f}</td></tr>
                <tr><td>Average Win</td><td class="positive">${metrics.avg_win:.2f}</td></tr>
                <tr><td>Average Loss</td><td class="negative">${metrics.avg_loss:.2f}</td></tr>
                <tr><td>Total Trades</td><td>{metrics.total_trades}</td></tr>
                <tr><td>Information Ratio</td><td>{metrics.information_ratio:.2f}</td></tr>
        """
        
        # Add alpha and beta if available
        if metrics.alpha is not None and metrics.beta is not None:
            html += f"""
                <tr><td>Alpha (vs {self.benchmark_symbol})</td><td class="{'positive' if metrics.alpha > 0 else 'negative'}">{metrics.alpha:.2%}</td></tr>
                <tr><td>Beta (vs {self.benchmark_symbol})</td><td>{metrics.beta:.2f}</td></tr>
            """
        
        html += """
            </table>
            
            <h2>Portfolio Evolution</h2>
            <p>See generated charts for visual analysis of portfolio performance over time.</p>
            
        </body>
        </html>
        """
        
        return html
    
    def _generate_performance_plots(self, report_name: str) -> None:
        """Generate performance visualization plots"""
        
        if not self.portfolio_history:
            return
        
        # Setup the plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trading Strategy Performance Analysis', fontsize=16)
        
        # Extract data for plotting
        timestamps = [state.timestamp for state in self.portfolio_history]
        portfolio_values = [state.total_value for state in self.portfolio_history]
        drawdowns = [state.drawdown for state in self.portfolio_history]
        
        # Plot 1: Portfolio Value Over Time
        axes[0, 0].plot(timestamps, portfolio_values, linewidth=2, color='blue')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown Over Time
        axes[0, 1].fill_between(timestamps, 0, [-d*100 for d in drawdowns], 
                               color='red', alpha=0.6)
        axes[0, 1].set_title('Drawdown Over Time')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Daily Returns Distribution
        if self.daily_returns:
            axes[1, 0].hist(self.daily_returns, bins=50, alpha=0.7, color='green')
            axes[1, 0].axvline(np.mean(self.daily_returns), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(self.daily_returns):.4f}')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Rolling Sharpe Ratio
        if len(self.daily_returns) > 60:  # Need at least 60 days for rolling calculation
            rolling_window = 60
            rolling_sharpe = []
            
            for i in range(rolling_window, len(self.daily_returns)):
                window_returns = self.daily_returns[i-rolling_window:i]
                risk_free_daily = self.risk_free_rate / 252
                excess_returns = np.array(window_returns) - risk_free_daily
                
                if np.std(excess_returns) > 0:
                    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0)
            
            rolling_dates = timestamps[rolling_window:]
            axes[1, 1].plot(rolling_dates, rolling_sharpe, linewidth=2, color='purple')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title(f'Rolling Sharpe Ratio ({rolling_window}-day)')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_directory / f"{report_name}_performance_charts.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated performance charts: {plot_path}")
    
    def _save_raw_data(self, report_name: str) -> None:
        """Save raw performance data to CSV files"""
        
        # Portfolio history
        portfolio_data = []
        for state in self.portfolio_history:
            portfolio_data.append({
                'timestamp': state.timestamp,
                'cash': state.cash,
                'total_value': state.total_value,
                'day_pnl': state.day_pnl,
                'total_pnl': state.total_pnl,
                'drawdown': state.drawdown,
                'max_drawdown': state.max_drawdown,
                'transactions_today': state.transactions_today
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.to_csv(self.output_directory / f"{report_name}_portfolio_history.csv", index=False)
        
        # Transaction history
        if self.transaction_history:
            transaction_data = []
            for txn in self.transaction_history:
                transaction_data.append({
                    'timestamp': txn.timestamp,
                    'symbol': txn.symbol,
                    'side': txn.side.value,
                    'quantity': txn.quantity,
                    'price': txn.price,
                    'commission': txn.commission,
                    'total_cost': txn.total_cost,
                    'order_id': txn.order_id
                })
            
            transaction_df = pd.DataFrame(transaction_data)
            transaction_df.to_csv(self.output_directory / f"{report_name}_transactions.csv", index=False)
        
        # Performance metrics as JSON
        if self.performance_metrics:
            metrics_dict = {
                'total_return': self.performance_metrics.total_return,
                'annualized_return': self.performance_metrics.annualized_return,
                'volatility': self.performance_metrics.volatility,
                'sharpe_ratio': self.performance_metrics.sharpe_ratio,
                'sortino_ratio': self.performance_metrics.sortino_ratio,
                'max_drawdown': self.performance_metrics.max_drawdown,
                'calmar_ratio': self.performance_metrics.calmar_ratio,
                'var_95': self.performance_metrics.var_95,
                'win_rate': self.performance_metrics.win_rate,
                'profit_factor': self.performance_metrics.profit_factor,
                'avg_win': self.performance_metrics.avg_win,
                'avg_loss': self.performance_metrics.avg_loss,
                'total_trades': self.performance_metrics.total_trades,
                'information_ratio': self.performance_metrics.information_ratio,
                'alpha': self.performance_metrics.alpha,
                'beta': self.performance_metrics.beta
            }
            
            with open(self.output_directory / f"{report_name}_metrics.json", 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
        
        logger.info(f"Saved raw performance data for {report_name}")
    
    def compare_strategies(self, other_logger: 'MetricsLogger', strategy_names: List[str]) -> str:
        """Compare performance between two strategies"""
        
        # This would generate a comparative analysis
        # Implementation would compare metrics side by side
        pass
    
    def reset(self) -> None:
        """Reset all logged data"""
        
        self.portfolio_history = []
        self.transaction_history = []
        self.daily_returns = []
        self.benchmark_returns = []
        self.performance_metrics = None
        self.period_analyses = {}
        
        logger.info("MetricsLogger reset to initial state") 
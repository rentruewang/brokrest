#!/usr/bin/env python3
"""
Automated Freqtrade Backtesting Script with plot-dataframe
Runs backtests, exports results, generates CSV and uses freqtrade plot-dataframe
"""

import subprocess
import json
import pandas as pd
import gzip
import shutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FreqtradeBacktester:
    """
    Automate Freqtrade backtesting and analysis with plot-dataframe
    """
    
    def __init__(self, 
                 strategy: str,
                 config_path: str = "C:/Users/youss/freqtrade-bot/freqtrade/user_data/config-live.json",
                 data_dir: str = "C:/Users/youss/freqtrade-bot/freqtrade/user_data/data/okx",
                 results_dir: str = "C:/Users/youss/freqtrade-bot/freqtrade/user_data/backtest_results"):
        """
        Initialize the backtester
        
        Args:
            strategy: Strategy name
            config_path: Path to config file
            data_dir: Directory containing historical data
            results_dir: Directory to store results
        """
        self.strategy = strategy
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique result filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_name = f"{strategy}_{timestamp}"
        self.result_path = self.results_dir / f"{self.result_name}.json"
        
        # Create plots directory
        self.plots_dir = self.results_dir / f"{self.result_name}_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def run_backtest(self, 
                     timerange: str,
                     timeframe: str = "15m",
                     max_open_trades: int = 3,
                     stake_amount: str = "unlimited",
                     enable_protections: bool = False,
                     breakdown: List[str] = None) -> bool:
        """
        Run Freqtrade backtest
        
        Args:
            timerange: Time range in format YYYYMMDD-YYYYMMDD
            timeframe: Timeframe (1m, 5m, 15m, 1h, etc)
            max_open_trades: Maximum number of open trades
            stake_amount: Stake amount or "unlimited"
            enable_protections: Enable strategy protections
            breakdown: List of breakdown options ['day', 'week', 'month']
        
        Returns:
            bool: Success status
        """
        logger.info(f"Starting backtest for {self.strategy}")
        logger.info(f"Time range: {timerange}")
        logger.info(f"Timeframe: {timeframe}")
        
        # Build command
        cmd = [
            "freqtrade", "backtesting",
            "--config", str(self.config_path),
            "--strategy", self.strategy,
            "--timerange", timerange,
            "--timeframe", timeframe,
            "--max-open-trades", str(max_open_trades),
            "--stake-amount", stake_amount,
            "--export", "trades",
            "--export-filename", str(self.result_path),
            "--datadir", str(self.data_dir),
            "--cache", "day"  # Use daily cache for faster repeated runs
        ]
        
        # Add optional parameters
        if enable_protections:
            cmd.append("--enable-protections")  
        
        if breakdown:
            for b in breakdown:
                cmd.extend(["--breakdown", b])
        
        # Add position stacking
        cmd.append("--enable-position-stacking")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run backtest
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log output
            logger.info("Backtest completed successfully")
            print("\n" + "="*60)
            print("BACKTEST OUTPUT:")
            print("="*60)
            print(process.stdout)
            
            # Save stdout to file
            output_file = self.results_dir / f"{self.result_name}_output.txt"
            with open(output_file, 'w') as f:
                f.write(process.stdout)
            
            # Store timerange and timeframe for plot generation
            self.timerange = timerange
            self.timeframe = timeframe
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Backtest failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def extract_results(self) -> Dict:
        """
        Extract and unzip backtest results
        
        Returns:
            Dict: Backtest results data
        """
        import zipfile
        import os
        print(f"Results directory: {os.listdir(self.results_dir)}")
        print(f"Result name: {self.result_name}")
        try:
            for file in os.listdir(self.results_dir):
                print(f"File: {file}")
                if self.result_name in file and file.endswith('.zip'):
                    print(f"File: {file}")
                    logger.info(f"Extracting results from {file}")
                    gz_path = self.results_dir / file
                    extract_to = self.results_dir / "extracted_results"
                    # Ensure the output directory exists
                    os.makedirs(extract_to, exist_ok=True)

                    # Open and extract
                    with zipfile.ZipFile(gz_path, 'r') as zf:
                        zf.extractall(path=extract_to)
                        print(f"Extracted all files to: {extract_to}")
                    with open(extract_to / file.replace('.zip', '.json'), 'r') as f:
                        data = json.load(f)
                    # logger.info(f"Loaded results: {len(data.get('trades', []))} trades")
                    return data
        except Exception as e:
            logger.error(f"Failed to extract results: {e}")
            return {}
        return {}
    
    def generate_trades_csv(self, results: Dict) -> pd.DataFrame:
        """
        Generate CSV file with realized trades
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            pd.DataFrame: Trades dataframe
        """
        trades = results['strategy'][self.strategy].get('trades', [])
        
        if not trades:
            logger.warning("No trades found in results")
            return pd.DataFrame()
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Add calculated columns
        if 'open_date' in trades_df.columns:
            trades_df['open_date'] = pd.to_datetime(trades_df['open_date'])
            trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])
            
            # Calculate trade duration in hours
            trades_df['duration_hours'] = (
                trades_df['close_date'] - trades_df['open_date']
            ).dt.total_seconds() / 3600
            
            # Add day of week
            trades_df['open_dow'] = trades_df['open_date'].dt.day_name()
            trades_df['close_dow'] = trades_df['close_date'].dt.day_name()
            
            # Add hour of day
            trades_df['open_hour'] = trades_df['open_date'].dt.hour
            trades_df['close_hour'] = trades_df['close_date'].dt.hour
        
        # Reorganize columns
        important_cols = [
            'pair', 'profit_ratio', 'profit_abs', 'open_date', 'close_date',
            'duration_hours', 'open_rate', 'close_rate', 'amount', 'fee_open',
            'fee_close', 'trade_duration', 'is_open', 'is_short', 'exit_reason',
            'enter_tag', 'min_rate', 'max_rate', 'stop_loss_ratio', 'stop_loss_abs',
            'initial_stop_loss_ratio', 'initial_stop_loss_abs', 'open_dow', 
            'close_dow', 'open_hour', 'close_hour'
        ]
        
        # Keep only existing columns
        cols_to_keep = [col for col in important_cols if col in trades_df.columns]
        trades_df = trades_df[cols_to_keep]
        
        # Save to CSV
        csv_path = self.results_dir / f"{self.result_name}_trades.csv"
        trades_df.to_csv(csv_path, index=False)
        logger.info(f"Saved trades CSV to {csv_path}")

        import seaborn as sns

        sns.set(style="whitegrid")

        # 1) Distribution of Profit Ratio
        plt.figure()
        sns.histplot(trades_df["profit_ratio"], kde=True)
        plt.title("Distribution of Trade Profit Ratio")
        plt.xlabel("Profit Ratio")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("distribution_profit_ratio.png")
        plt.show()

        # 2) Profit Ratio vs Trade Duration
        plt.figure()
        sns.scatterplot(x="duration_hours", y="profit_ratio", data=trades_df)
        plt.title("Profit Ratio vs Trade Duration")
        plt.xlabel("Duration (hours)")
        plt.ylabel("Profit Ratio")
        plt.tight_layout()
        plt.savefig("profit_vs_duration.png")
        plt.show()

        # 3) Profit Ratio by Open Day of Week
        plt.figure()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sns.boxplot(x="open_dow", y="profit_ratio", data=trades_df, order=order)
        plt.title("Profit Ratio by Open Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Profit Ratio")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("profit_by_day_of_week.png")
        plt.show()

        
        # Print summary
        print("\n" + "="*60)
        print("TRADES SUMMARY:")
        print("="*60)
        print(f"Total trades: {len(trades_df)}")
        print(f"Winning trades: {(trades_df['profit_ratio'] > 0).sum()}")
        print(f"Losing trades: {(trades_df['profit_ratio'] < 0).sum()}")
        print(f"Win rate: {(trades_df['profit_ratio'] > 0).sum() / len(trades_df) * 100:.2f}%")
        print(f"Average profit: {trades_df['profit_ratio'].mean() * 100:.2f}%")
        print(f"Total profit: {trades_df['profit_ratio'].sum() * 100:.2f}%")
        print(f"Best trade: {trades_df['profit_ratio'].max() * 100:.2f}%")
        print(f"Worst trade: {trades_df['profit_ratio'].min() * 100:.2f}%")
        print(f"Average duration: {trades_df['duration_hours'].mean():.1f} hours")
        
        return trades_df
    
    def generate_plot_dataframe(self, 
                               pairs: List[str] = None,
                               indicators1: List[str] = None,
                               indicators2: List[str] = None,
                               plot_limit: int = 750,
                               trade_source: str = "file") -> Dict[str, str]:
        """
        Generate plot-dataframe plots for selected pairs
        
        Args:
            pairs: List of pairs to plot (if None, plots all traded pairs)
            indicators1: Indicators to plot on main chart
            indicators2: Indicators to plot on separate chart
            plot_limit: Maximum number of candles to plot
            trade_source: Source of trades ('file' or 'DB')
            
        Returns:
            Dict[str, str]: Mapping of pair to plot file path
        """
        logger.info("Generating plot-dataframe plots...")
        
        # Get list of pairs from trades if not specified
        if pairs is None:
            results = self.extract_results()
            trades = results.get('trades', [])
            if trades:
                pairs = list(set([trade['pair'] for trade in trades]))
                logger.info(f"Found {len(pairs)} unique pairs in trades")
            else:
                logger.warning("No trades found to determine pairs")
                return {}
        
        # Limit number of pairs to plot (to avoid too many files)
        max_pairs = 10
        if len(pairs) > max_pairs:
            logger.warning(f"Too many pairs ({len(pairs)}), limiting to top {max_pairs} by trade count")
            # Get top pairs by trade count
            results = self.extract_results()
            trades = results.get('trades', [])
            pair_counts = {}
            for trade in trades:
                pair = trade['pair']
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            pairs = sorted(pair_counts.keys(), key=lambda x: pair_counts[x], reverse=True)[:max_pairs]
        
        plot_files = {}
        
        for pair in pairs:
            logger.info(f"Generating plot for {pair}...")
            
            # Build plot-dataframe command
            cmd = [
                "freqtrade", "plot-dataframe",
                "--config", str(self.config_path),
                "--strategy", self.strategy,
                "--pair", pair,
                "--timerange", self.timerange,
                "--timeframe", self.timeframe,
                # "--plot-limit", str(plot_limit),
                "--trade-source", trade_source
            ]
            
            # Add export trades file if using file source
            if trade_source == "file":
                cmd.extend(["--export-filename", str(self.result_path)])
            
            # Add indicators if specified
            if indicators1:
                cmd.extend(["--indicators1"] + indicators1)
            if indicators2:
                cmd.extend(["--indicators2"] + indicators2)
            
            try:
                # Run plot-dataframe
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # The plot file is saved in user_data/plot/
                plot_filename = f"{self.strategy}-{pair.replace('/', '_')}-{self.timeframe}.html"
                plot_path = Path("user_data/plot") / plot_filename
                
                if plot_path.exists():
                    # Copy to our results directory
                    dest_path = self.plots_dir / plot_filename
                    shutil.copy(plot_path, dest_path)
                    plot_files[pair] = str(dest_path)
                    logger.info(f"Plot saved to {dest_path}")
                else:
                    logger.warning(f"Plot file not found: {plot_path}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to generate plot for {pair}: {e}")
                logger.error(f"Error output: {e.stderr}")
        
        return plot_files
    
    def generate_plot_profit(self) -> Optional[str]:
        """
        Generate plot-profit graph
        
        Returns:
            Optional[str]: Path to plot file if successful
        """
        logger.info("Generating plot-profit...")
        
        # Build plot-profit command
        cmd = [
            "freqtrade", "plot-profit",
            "--config", str(self.config_path),
            "--strategy", self.strategy,
            "--timerange", self.timerange,
            "--trade-source", "file",
            "--export-filename", str(self.result_path)
        ]
        
        try:
            # Run plot-profit
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # The plot file is saved in user_data/plot/
            plot_filename = f"{self.strategy}-profit-plot.html"
            plot_path = Path("user_data/plot") / plot_filename
            
            if plot_path.exists():
                # Copy to our results directory
                dest_path = self.plots_dir / plot_filename
                shutil.copy(plot_path, dest_path)
                logger.info(f"Profit plot saved to {dest_path}")
                return str(dest_path)
            else:
                logger.warning(f"Profit plot file not found: {plot_path}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate profit plot: {e}")
            logger.error(f"Error output: {e.stderr}")
            return None
    
    def generate_custom_analysis_plots(self, trades_df: pd.DataFrame):
        """
        Generate additional custom analysis plots
        """
        if len(trades_df) == 0:
            logger.warning("No trades to plot")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Profit distribution
        ax1 = plt.subplot(2, 3, 1)
        trades_df['profit_percent'] = trades_df['profit_ratio'] * 100
        ax1.hist(trades_df['profit_percent'], bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.set_title('Profit Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Profit %')
        ax1.set_ylabel('Frequency')
        mean_profit = trades_df['profit_percent'].mean()
        ax1.axvline(x=mean_profit, color='g', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_profit:.2f}%')
        ax1.legend()
        
        # 2. Profit by pair
        ax2 = plt.subplot(2, 3, 2)
        pair_profits = trades_df.groupby('pair')['profit_ratio'].agg(['mean', 'count'])
        pair_profits['mean_percent'] = pair_profits['mean'] * 100
        pair_profits = pair_profits.sort_values('mean_percent', ascending=True).tail(15)
        
        colors = ['red' if x < 0 else 'green' for x in pair_profits['mean_percent']]
        pair_profits['mean_percent'].plot(kind='barh', ax=ax2, color=colors)
        ax2.set_title('Average Profit by Pair (Top 15)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Average Profit %')
        
        # Add trade count annotations
        for i, (idx, row) in enumerate(pair_profits.iterrows()):
            ax2.text(row['mean_percent'] + 0.1, i, f"({int(row['count'])})", 
                    va='center', fontsize=8)
        
        # 3. Daily profit
        ax3 = plt.subplot(2, 3, 3)
        trades_df['close_date_only'] = trades_df['close_date'].dt.date
        daily_profit = trades_df.groupby('close_date_only')['profit_ratio'].sum() * 100
        ax3.plot(daily_profit.index, daily_profit.values, marker='o', linewidth=1, markersize=4)
        ax3.set_title('Daily Profit', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Daily Profit %')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 4. Profit by hour of day
        ax4 = plt.subplot(2, 3, 4)
        hourly_profit = trades_df.groupby('open_hour')['profit_ratio'].agg(['mean', 'count'])
        hourly_profit['mean_percent'] = hourly_profit['mean'] * 100
        
        ax4.bar(hourly_profit.index, hourly_profit['mean_percent'], 
                color='skyblue', edgecolor='black')
        ax4.set_title('Average Profit by Hour of Day', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Hour')
        ax4.set_ylabel('Average Profit %')
        ax4.set_xticks(range(0, 24, 2))
        
        # 5. Exit reason analysis
        ax5 = plt.subplot(2, 3, 5)
        exit_reasons = trades_df['exit_reason'].value_counts()
        colors = plt.cm.Set3(range(len(exit_reasons)))
        ax5.pie(exit_reasons.values, labels=exit_reasons.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax5.set_title('Exit Reasons', fontsize=14, fontweight='bold')
        
        # 6. Win rate by day of week
        ax6 = plt.subplot(2, 3, 6)
        dow_stats = trades_df.groupby('open_dow').apply(
            lambda x: (x['profit_ratio'] > 0).sum() / len(x) * 100
        )
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats = dow_stats.reindex([d for d in days_order if d in dow_stats.index])
        
        ax6.bar(range(len(dow_stats)), dow_stats.values, color='lightcoral', edgecolor='black')
        ax6.set_xticks(range(len(dow_stats)))
        ax6.set_xticklabels(dow_stats.index, rotation=45)
        ax6.set_title('Win Rate by Day of Week', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Win Rate %')
        ax6.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{self.result_name}_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved analysis plots to {plot_path}")
        plt.close()
    
    def generate_statistics_report(self, trades_df: pd.DataFrame, results: Dict):
        """
        Generate detailed statistics report
        """
        stats_path = self.results_dir / f"{self.result_name}_statistics.txt"
        
        with open(stats_path, 'w') as f:
            f.write(f"FREQTRADE BACKTEST STATISTICS REPORT\n")
            f.write(f"Strategy: {self.strategy}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*30 + "\n")
            f.write(f"Total trades: {len(trades_df)}\n")
            f.write(f"Winning trades: {(trades_df['profit_ratio'] > 0).sum()}\n")
            f.write(f"Losing trades: {(trades_df['profit_ratio'] < 0).sum()}\n")
            f.write(f"Win rate: {(trades_df['profit_ratio'] > 0).sum() / len(trades_df) * 100:.2f}%\n")
            f.write(f"Average profit: {trades_df['profit_ratio'].mean() * 100:.2f}%\n")
            f.write(f"Median profit: {trades_df['profit_ratio'].median() * 100:.2f}%\n")
            f.write(f"Total profit: {trades_df['profit_ratio'].sum() * 100:.2f}%\n")
            f.write(f"Best trade: {trades_df['profit_ratio'].max() * 100:.2f}%\n")
            f.write(f"Worst trade: {trades_df['profit_ratio'].min() * 100:.2f}%\n")
            f.write(f"Standard deviation: {trades_df['profit_ratio'].std() * 100:.2f}%\n")
            f.write(f"Average trade duration: {trades_df['duration_hours'].mean():.1f} hours\n\n")
            
            # Performance by pair
            f.write("PERFORMANCE BY PAIR\n")
            f.write("-"*30 + "\n")
            pair_stats = trades_df.groupby('pair').agg({
                'profit_ratio': ['count', 'mean', 'sum', 'std'],
                'duration_hours': 'mean'
            }).round(4)
            f.write(pair_stats.to_string() + "\n\n")
            
            # Performance by exit reason
            f.write("PERFORMANCE BY EXIT REASON\n")
            f.write("-"*30 + "\n")
            exit_stats = trades_df.groupby('exit_reason').agg({
                'profit_ratio': ['count', 'mean', 'sum']
            }).round(4)
            f.write(exit_stats.to_string() + "\n\n")
            
            # Performance by entry tag (if available)
            if 'enter_tag' in trades_df.columns:
                f.write("PERFORMANCE BY ENTRY TAG\n")
                f.write("-"*30 + "\n")
                entry_stats = trades_df.groupby('enter_tag').agg({
                    'profit_ratio': ['count', 'mean', 'sum']
                }).round(4)
                f.write(entry_stats.to_string() + "\n\n")
        
        logger.info(f"Saved statistics report to {stats_path}")
    
    def create_plot_index(self, plot_files: Dict[str, str], profit_plot: str = None):
        """
        Create an HTML index file linking to all plots
        """
        index_path = self.plots_dir / "index.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.strategy} Backtest Results - {self.result_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                }}
                h2 {{
                    color: #666;
                    margin-top: 30px;
                }}
                .plot-link {{
                    display: block;
                    margin: 10px 0;
                    padding: 10px;
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-decoration: none;
                    color: #0066cc;
                }}
                .plot-link:hover {{
                    background-color: #f0f0f0;
                }}
                .summary {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>{self.strategy} Backtest Results</h1>
            <div class="summary">
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Time Range:</strong> {self.timerange}</p>
                <p><strong>Timeframe:</strong> {self.timeframe}</p>
            </div>
        """
        
        if profit_plot:
            html_content += f"""
            <h2>Profit Analysis</h2>
            <a href="{Path(profit_plot).name}" class="plot-link">📈 Overall Profit Plot</a>
        """
        
        if plot_files:
            html_content += """
            <h2>Individual Pair Plots</h2>
        """
            for pair, plot_path in sorted(plot_files.items()):
                html_content += f"""
            <a href="{Path(plot_path).name}" class="plot-link">📊 {pair}</a>
        """
        
        html_content += """
            <h2>Other Results</h2>
            <a href="../{0}_trades.csv" class="plot-link">📄 Trades CSV</a>
            <a href="../{0}_statistics.txt" class="plot-link">📊 Statistics Report</a>
            <a href="../{0}_output.txt" class="plot-link">📝 Backtest Output</a>
            <a href="{0}_analysis.png" class="plot-link">📈 Analysis Charts</a>
        </body>
        </html>
        """.format(self.result_name)
        
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created plot index at {index_path}")
        return str(index_path)
    
    def run_complete_analysis(self,
                            timerange: str,
                            timeframe: str = "15m",
                            pairs: List[str] = None,
                            indicators1: List[str] = None,
                            indicators2: List[str] = None,
                            plot_limit: int = 750,
                            **kwargs):
        """
        Run complete backtest and analysis workflow
        
        Args:
            timerange: Time range for backtest
            timeframe: Timeframe
            pairs: List of pairs to test (optional)
            indicators1: Indicators to plot on main chart
            indicators2: Indicators to plot on separate chart
            plot_limit: Maximum number of candles to plot
            **kwargs: Additional arguments for run_backtest
        """
        print("\n" + "="*60)
        print(f"STARTING BACKTEST ANALYSIS FOR {self.strategy}")
        print("="*60)
        
        # Update config if pairs specified
        if pairs:
            self.update_config_pairs(pairs)
        
        # Step 1: Run backtest
        print("\n📊 Step 1/5: Running backtest...")
        success = self.run_backtest(timerange, timeframe, **kwargs)
        
        if not success:
            logger.error("Backtest failed!")
            return
        
        # Step 2: Extract results
        print("\n📂 Step 2/5: Extracting results...")
        results = self.extract_results()
        
        if not results:
            logger.error("No results to analyze!")
            return
        
        # Step 3: Generate CSV
        print("\n📄 Step 3/5: Generating trades CSV...")
        trades_df = self.generate_trades_csv(results)
        
        if len(trades_df) == 0:
            logger.warning("No trades to analyze!")
            return
        
        # Step 4: Generate plots using plot-dataframe
        print("\n📈 Step 4/5: Generating plots...")
        
        # Default indicators if not specified
        if indicators1 is None:
            indicators1 = ['sma', 'ema', 'bb_lowerband', 'bb_upperband']
        if indicators2 is None:
            indicators2 = ['rsi', 'macd', 'macdsignal']
        
        # Generate individual pair plots
        plot_files = self.generate_plot_dataframe(
            pairs=None,  # Will use pairs from trades
            indicators1=indicators1,
            indicators2=indicators2,
            plot_limit=plot_limit
        )
        
        # Generate profit plot
        # profit_plot = self.generate_plot_profit()
        
        # Generate custom analysis plots
        self.generate_custom_analysis_plots(trades_df)
        
        # Step 5: Generate reports
        print("\n📊 Step 5/5: Generating reports...")
        self.generate_statistics_report(trades_df, results)
        
        # Create index HTML file
        index_file = self.create_plot_index(plot_files, profit_plot)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\n📁 Results saved to: {self.results_dir}")
        print(f"\n📊 Key files generated:")
        print(f"  - Trades CSV: {self.result_name}_trades.csv")
        print(f"  - Statistics: {self.result_name}_statistics.txt")
        print(f"  - Backtest output: {self.result_name}_output.txt")
        print(f"  - Analysis charts: {self.result_name}_plots/{self.result_name}_analysis.png")
        print(f"  - Plot index: {self.result_name}_plots/index.html")
        
        if plot_files:
            print(f"\n📈 Interactive plots generated for {len(plot_files)} pairs:")
            for pair in list(plot_files.keys())[:5]:
                print(f"  - {pair}")
            if len(plot_files) > 5:
                print(f"  ... and {len(plot_files) - 5} more")
        
        print(f"\n🌐 Open the index file to view all plots:")
        print(f"   file://{Path(index_file).absolute()}")
    
    def update_config_pairs(self, pairs: List[str]):
        """Update config file with specific pairs"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            config['exchange']['pair_whitelist'] = pairs
            
            # Save to temporary config
            temp_config = self.results_dir / f"{self.result_name}_config.json"
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update config path to use temporary
            self.config_path = temp_config
            logger.info(f"Created temporary config with {len(pairs)} pairs")


def main():
    """
    Example usage of the backtester
    """
    # Example 1: Basic usage
    backtester = FreqtradeBacktester(
        strategy="ichiV3",
        config_path="C:/Users/youss/freqtrade-bot/freqtrade/user_data/config-live.json"
    )
    
    # Run complete analysis with default settings
    backtester.run_complete_analysis(
        timerange="20250101-20250601",
        timeframe="15m",
        max_open_trades=5,
        stake_amount="unlimited",
        enable_protections=False,
        breakdown=["day", "week", "month"],
        # indicators1=["bb_lowerband", "bb_middleband", "bb_upperband", "ema_9", "ema_21"],
        # indicators2=["rsi", "macd", "macdsignal", "macdhist"],
        plot_limit=1000
    )
    
    # Example 2: Test specific pairs with custom indicators
    """
    backtester2 = FreqtradeBacktester(
        strategy="MyCustomStrategy",
        config_path="C:/Users/youss/freqtrade-bot/freqtrade/user_data/config-custom.json"
    )
    
    backtester2.run_complete_analysis(
        timerange="20240101-20240301",
        timeframe="5m",
        pairs=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        indicators1=["sma_20", "sma_50", "bb_lowerband", "bb_upperband"],
        indicators2=["rsi", "stoch", "volume"],
        plot_limit=500
    )
    """


if __name__ == "__main__":
    main()

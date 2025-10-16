#!/usr/bin/env python3
"""
Data Visualizer for Seasonal Trading Strategies
This tool helps visualize the data and understand the project structure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SeasonalDataVisualizer:
    def __init__(self, data_folder="SeasonalBackend/Data"):
        self.data_folder = Path(data_folder)
        self.data_files = list(self.data_folder.glob("*Raw.txt"))
        self.symbols = [f.stem.replace("Raw", "") for f in self.data_files]
        
    def load_symbol_data(self, symbol, limit=None):
        """Load data for a specific symbol"""
        file_path = self.data_folder / f"{symbol}Raw.txt"
        if not file_path.exists():
            return None
            
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            if limit:
                df = df.tail(limit)
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None
    
    def get_data_summary(self):
        """Get summary of all available data"""
        summary = []
        for symbol in self.symbols[:20]:  # Limit to first 20 for demo
            df = self.load_symbol_data(symbol, limit=100)
            if df is not None:
                summary.append({
                    'Symbol': symbol,
                    'Start_Date': df.index.min(),
                    'End_Date': df.index.max(),
                    'Records': len(df),
                    'Latest_Close': df['Close'].iloc[-1] if 'Close' in df.columns else None
                })
        return pd.DataFrame(summary)
    
    def plot_symbol_performance(self, symbols=['SPY', 'QQQ', 'GLD', 'TLT'], days=252*2):
        """Plot performance of selected symbols"""
        plt.figure(figsize=(15, 10))
        
        for i, symbol in enumerate(symbols):
            df = self.load_symbol_data(symbol, limit=days)
            if df is not None and 'Close' in df.columns:
                # Calculate cumulative returns
                returns = df['Close'].pct_change().fillna(0)
                cumulative = (1 + returns).cumprod()
                
                plt.subplot(2, 2, i+1)
                plt.plot(cumulative.index, cumulative.values)
                plt.title(f'{symbol} Performance (Last {days} days)')
                plt.ylabel('Cumulative Return')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('symbol_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_strategy_data_requirements(self):
        """Analyze what data each strategy needs"""
        strategies = {
            'DayOfTheWeek': ['ES', 'GC', 'CL', 'AD', 'BP'],
            'GateStrategies': ['SMH', 'SLV', 'GLD', 'QQQ', 'JNK'],
            'RankingStrategies': ['GDX', 'SMH', 'GLD', 'XLE', 'XHB', 'QQQ', 'XLK', 'SPY'],
            'SwitchStrategies': ['GLD', 'SMH', 'XLU'],
            'SeasonalMovingAverages': ['SMH', 'GLD', 'XLK', 'QQQ', 'SPY', 'XLE'],
            'TDOMSwitch': ['SMH', 'TLT', 'QQQ', 'GLD'],
            'WeeklySwitch': ['EEM', 'GLD', 'TLT', 'QQQ', 'XME', 'XLK', 'XLV', 'XLY'],
            'RebalanceEffect': ['TLT', 'IWM', 'SPY', 'SMH', 'HYG', 'EEM', 'DIA']
        }
        
        # Check data availability for each strategy
        availability = {}
        for strategy, required_symbols in strategies.items():
            available = []
            missing = []
            for symbol in required_symbols:
                if f"{symbol}Raw.txt" in [f.name for f in self.data_files]:
                    available.append(symbol)
                else:
                    missing.append(symbol)
            availability[strategy] = {
                'available': available,
                'missing': missing,
                'coverage': len(available) / len(required_symbols)
            }
        
        return availability
    
    def create_data_overview_plot(self):
        """Create overview plot of data availability"""
        summary = self.get_data_summary()
        
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Data coverage by symbol
        plt.subplot(2, 2, 1)
        summary['Records'].plot(kind='bar')
        plt.title('Data Records by Symbol')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=45)
        
        # Plot 2: Date range coverage
        plt.subplot(2, 2, 2)
        summary['Start_Date'] = pd.to_datetime(summary['Start_Date'])
        summary['End_Date'] = pd.to_datetime(summary['End_Date'])
        summary['Duration_Days'] = (summary['End_Date'] - summary['Start_Date']).dt.days
        summary['Duration_Days'].plot(kind='bar')
        plt.title('Data Duration by Symbol (Days)')
        plt.ylabel('Duration (Days)')
        plt.xticks(rotation=45)
        
        # Plot 3: Latest prices
        plt.subplot(2, 2, 3)
        summary['Latest_Close'].dropna().plot(kind='bar')
        plt.title('Latest Close Prices')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        
        # Plot 4: Data availability heatmap
        plt.subplot(2, 2, 4)
        availability = self.analyze_strategy_data_requirements()
        strategy_names = list(availability.keys())
        coverage_rates = [availability[s]['coverage'] for s in strategy_names]
        
        plt.bar(strategy_names, coverage_rates)
        plt.title('Data Coverage by Strategy')
        plt.ylabel('Coverage Rate')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_backend_requirements(self):
        """Generate requirements for backend service"""
        requirements = {
            'data_endpoints': {
                'description': 'Endpoints to serve strategy data',
                'endpoints': [
                    '/api/data/{symbol}',
                    '/api/data/{symbol}/latest',
                    '/api/data/{symbol}/range/{start_date}/{end_date}'
                ]
            },
            'strategy_endpoints': {
                'description': 'Endpoints for strategy execution',
                'endpoints': [
                    '/api/strategies/list',
                    '/api/strategies/{strategy_id}/execute',
                    '/api/strategies/{strategy_id}/performance',
                    '/api/strategies/{strategy_id}/alerts'
                ]
            },
            'alert_endpoints': {
                'description': 'Endpoints for alert generation',
                'endpoints': [
                    '/api/alerts/generate',
                    '/api/alerts/{strategy_id}',
                    '/api/alerts/history'
                ]
            }
        }
        return requirements

def main():
    """Main function to run the visualizer"""
    print("Seasonal Trading Strategies Data Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = SeasonalDataVisualizer()
    
    print(f"Found {len(visualizer.symbols)} data files")
    print(f"Sample symbols: {visualizer.symbols[:10]}")
    
    # Get data summary
    print("\nData Summary:")
    summary = visualizer.get_data_summary()
    print(summary.head(10))
    
    # Analyze strategy requirements
    print("\nStrategy Data Requirements:")
    availability = visualizer.analyze_strategy_data_requirements()
    for strategy, info in availability.items():
        print(f"{strategy}: {info['coverage']:.1%} coverage ({len(info['available'])}/{len(info['available']) + len(info['missing'])})")
        if info['missing']:
            print(f"  Missing: {info['missing']}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer.create_data_overview_plot()
    visualizer.plot_symbol_performance()
    
    # Generate backend requirements
    print("\nBackend Service Requirements:")
    requirements = visualizer.generate_backend_requirements()
    for category, info in requirements.items():
        print(f"\n{category.upper()}:")
        print(f"  {info['description']}")
        for endpoint in info['endpoints']:
            print(f"    {endpoint}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backend Architecture for Seasonal Trading Strategies Alert Service
This module provides the foundation for converting the 8 Python strategies into a robust backend service.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import pandas as pd
import numpy as np
from pathlib import Path
import asyncio
import json
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyType(str, Enum):
    DAY_OF_WEEK = "day_of_week"
    GATE = "gate"
    RANKING = "ranking"
    SWITCH = "switch"
    SEASONAL_MA = "seasonal_ma"
    TDOM_SWITCH = "tdom_switch"
    WEEKLY_SWITCH = "weekly_switch"
    REBALANCE_EFFECT = "rebalance_effect"

class AlertType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SWITCH = "switch"

class Alert(BaseModel):
    strategy_id: str
    strategy_name: str
    alert_type: AlertType
    symbol: str
    action: str  # "buy", "sell", "hold", "switch to"
    target_symbol: Optional[str] = None
    confidence: float
    timestamp: datetime
    reasoning: str
    performance_metrics: Dict[str, float]

class StrategyPerformance(BaseModel):
    strategy_id: str
    strategy_name: str
    cagr: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    last_updated: datetime

class DataManager:
    """Manages data loading and caching for all symbols"""
    
    def __init__(self, data_folder: str = "SeasonalBackend/Data"):
        self.data_folder = Path(data_folder)
        self.cache = {}
        self.last_update = {}
        
    def load_symbol_data(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load and cache symbol data"""
        cache_key = f"{symbol}_{limit}"
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load from file
        file_path = self.data_folder / f"{symbol}Raw.txt"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            if limit:
                df = df.tail(limit)
            
            # Cache the data
            self.cache[cache_key] = df
            self.last_update[symbol] = datetime.now()
            
            return df
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            raise
    
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """Get latest data point for a symbol"""
        df = self.load_symbol_data(symbol, limit=1)
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        return {
            'date': df.index[-1],
            'open': latest.get('Open', 0),
            'high': latest.get('High', 0),
            'low': latest.get('Low', 0),
            'close': latest.get('Close', 0),
            'volume': latest.get('Vol', 0)
        }
    
    def is_data_fresh(self, symbol: str, max_age_hours: int = 24) -> bool:
        """Check if data is fresh enough"""
        if symbol not in self.last_update:
            return False
        
        age = datetime.now() - self.last_update[symbol]
        return age.total_seconds() < max_age_hours * 3600

class StrategyEngine:
    """Core engine for executing trading strategies"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize all 8 strategies with their configurations"""
        return {
            "day_of_week": {
                "name": "Day of Week Strategies",
                "symbols": ["ES", "GC", "CL", "AD", "BP"],
                "description": "Day-of-week futures strategies with 1-day hold"
            },
            "gate": {
                "name": "Gate Strategies",
                "symbols": ["SMH", "SLV", "GLD", "QQQ", "JNK"],
                "description": "Gate-based strategies with A/B symbol selection"
            },
            "ranking": {
                "name": "Ranking Strategies",
                "symbols": ["GDX", "SMH", "GLD", "XLE", "XHB", "QQQ", "XLK", "SPY"],
                "description": "Monthly ranking-based strategies"
            },
            "switch": {
                "name": "Switch Strategies",
                "symbols": ["GLD", "SMH", "XLU"],
                "description": "Fixed month-to-symbol rotation"
            },
            "seasonal_ma": {
                "name": "Seasonal Moving Averages",
                "symbols": ["SMH", "GLD", "XLK", "QQQ", "SPY", "XLE"],
                "description": "Moving average-based monthly gate strategies"
            },
            "tdom_switch": {
                "name": "TDOM Switch",
                "symbols": ["SMH", "TLT", "QQQ", "GLD"],
                "description": "Half-month rotation by Trading Day of Month"
            },
            "weekly_switch": {
                "name": "Weekly Switch",
                "symbols": ["EEM", "GLD", "TLT", "QQQ", "XME", "XLK", "XLV", "XLY"],
                "description": "Week-of-month rotation strategies"
            },
            "rebalance_effect": {
                "name": "Rebalance Effect",
                "symbols": ["TLT", "IWM", "SPY", "SMH", "HYG", "EEM", "DIA"],
                "description": "Rebalance effect strategies"
            }
        }
    
    async def execute_strategy(self, strategy_id: str) -> List[Alert]:
        """Execute a specific strategy and return alerts"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        
        strategy_config = self.strategies[strategy_id]
        alerts = []
        
        try:
            if strategy_id == "day_of_week":
                alerts = await self._execute_day_of_week()
            elif strategy_id == "gate":
                alerts = await self._execute_gate_strategies()
            elif strategy_id == "ranking":
                alerts = await self._execute_ranking_strategies()
            elif strategy_id == "switch":
                alerts = await self._execute_switch_strategies()
            elif strategy_id == "seasonal_ma":
                alerts = await self._execute_seasonal_ma()
            elif strategy_id == "tdom_switch":
                alerts = await self._execute_tdom_switch()
            elif strategy_id == "weekly_switch":
                alerts = await self._execute_weekly_switch()
            elif strategy_id == "rebalance_effect":
                alerts = await self._execute_rebalance_effect()
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_id}: {e}")
            raise
    
    async def _execute_day_of_week(self) -> List[Alert]:
        """Execute day of week strategy"""
        # Implementation would call the actual DayOfTheWeek_Final.py logic
        # This is a simplified version
        alerts = []
        
        # Example logic: Check if it's a specific day of week
        today = datetime.now()
        if today.weekday() == 0:  # Monday
            alerts.append(Alert(
                strategy_id="day_of_week",
                strategy_name="Day of Week Strategy",
                alert_type=AlertType.BUY,
                symbol="ES",
                action="buy",
                confidence=0.8,
                timestamp=today,
                reasoning="Monday buy signal based on historical patterns",
                performance_metrics={"cagr": 0.12, "total_return": 0.15}
            ))
        
        return alerts
    
    async def _execute_gate_strategies(self) -> List[Alert]:
        """Execute gate strategies"""
        alerts = []
        
        # Example: SMH vs SLV in February
        current_month = datetime.now().month
        if current_month == 2:  # February
            alerts.append(Alert(
                strategy_id="gate",
                strategy_name="SMH vs SLV February Gate",
                alert_type=AlertType.SWITCH,
                symbol="SLV",
                action="switch to",
                target_symbol="SMH",
                confidence=0.75,
                timestamp=datetime.now(),
                reasoning="February gate strategy: switching from SLV to SMH",
                performance_metrics={"cagr": 0.18, "total_return": 0.22}
            ))
        
        return alerts
    
    async def _execute_ranking_strategies(self) -> List[Alert]:
        """Execute ranking strategies"""
        alerts = []
        
        # Example: Buy worst performer in current month
        current_month = datetime.now().month
        alerts.append(Alert(
            strategy_id="ranking",
            strategy_name="Monthly Ranking Strategy",
            alert_type=AlertType.BUY,
            symbol="XHB",
            action="buy",
            confidence=0.7,
            timestamp=datetime.now(),
            reasoning=f"Buying worst performer for month {current_month}",
            performance_metrics={"cagr": 0.14, "total_return": 0.18}
        ))
        
        return alerts
    
    async def _execute_switch_strategies(self) -> List[Alert]:
        """Execute switch strategies"""
        alerts = []
        
        # Example: GLD in specific months
        current_month = datetime.now().month
        if current_month in [1, 4, 8, 10, 12]:
            alerts.append(Alert(
                strategy_id="switch",
                strategy_name="GLD Fixed Month Rotation",
                alert_type=AlertType.BUY,
                symbol="GLD",
                action="buy",
                confidence=0.85,
                timestamp=datetime.now(),
                reasoning=f"GLD rotation for month {current_month}",
                performance_metrics={"cagr": 0.16, "total_return": 0.20}
            ))
        
        return alerts
    
    async def _execute_seasonal_ma(self) -> List[Alert]:
        """Execute seasonal moving average strategies"""
        alerts = []
        
        # Example: SMH vs GLD based on SMA
        try:
            smh_data = self.data_manager.load_symbol_data("SMH", limit=200)
            if len(smh_data) >= 200:
                sma_200 = smh_data['Close'].rolling(200).mean().iloc[-1]
                current_price = smh_data['Close'].iloc[-1]
                
                if current_price > sma_200:
                    alerts.append(Alert(
                        strategy_id="seasonal_ma",
                        strategy_name="SMH vs GLD SMA Strategy",
                        alert_type=AlertType.BUY,
                        symbol="SMH",
                        action="buy",
                        confidence=0.8,
                        timestamp=datetime.now(),
                        reasoning=f"SMH ({current_price:.2f}) > SMA200 ({sma_200:.2f})",
                        performance_metrics={"cagr": 0.17, "total_return": 0.21}
                    ))
                else:
                    alerts.append(Alert(
                        strategy_id="seasonal_ma",
                        strategy_name="SMH vs GLD SMA Strategy",
                        alert_type=AlertType.BUY,
                        symbol="GLD",
                        action="buy",
                        confidence=0.8,
                        timestamp=datetime.now(),
                        reasoning=f"SMH ({current_price:.2f}) < SMA200 ({sma_200:.2f}), switching to GLD",
                        performance_metrics={"cagr": 0.17, "total_return": 0.21}
                    ))
        except Exception as e:
            logger.error(f"Error in seasonal MA strategy: {e}")
        
        return alerts
    
    async def _execute_tdom_switch(self) -> List[Alert]:
        """Execute TDOM switch strategies"""
        alerts = []
        
        # Example: SMH for TDOM 1-12, TLT for TDOM 13-EOM
        today = datetime.now()
        tdom = today.day  # Simplified TDOM calculation
        
        if tdom <= 12:
            alerts.append(Alert(
                strategy_id="tdom_switch",
                strategy_name="SMH TDOM 1-12 Strategy",
                alert_type=AlertType.BUY,
                symbol="SMH",
                action="buy",
                confidence=0.8,
                timestamp=today,
                reasoning=f"TDOM {tdom}: Holding SMH for first half of month",
                performance_metrics={"cagr": 0.15, "total_return": 0.19}
            ))
        else:
            alerts.append(Alert(
                strategy_id="tdom_switch",
                strategy_name="TLT TDOM 13-EOM Strategy",
                alert_type=AlertType.BUY,
                symbol="TLT",
                action="buy",
                confidence=0.8,
                timestamp=today,
                reasoning=f"TDOM {tdom}: Holding TLT for second half of month",
                performance_metrics={"cagr": 0.15, "total_return": 0.19}
            ))
        
        return alerts
    
    async def _execute_weekly_switch(self) -> List[Alert]:
        """Execute weekly switch strategies"""
        alerts = []
        
        # Example: EEM, GLD, TLT, QQQ for weeks 1,2,3,4/5
        today = datetime.now()
        week_of_month = (today.day - 1) // 7 + 1
        week_of_month = min(week_of_month, 4)  # Cap at week 4
        
        symbols = ["EEM", "GLD", "TLT", "QQQ"]
        if week_of_month <= len(symbols):
            symbol = symbols[week_of_month - 1]
            alerts.append(Alert(
                strategy_id="weekly_switch",
                strategy_name="Weekly Rotation Strategy",
                alert_type=AlertType.BUY,
                symbol=symbol,
                action="buy",
                confidence=0.75,
                timestamp=today,
                reasoning=f"Week {week_of_month}: Holding {symbol}",
                performance_metrics={"cagr": 0.13, "total_return": 0.17}
            ))
        
        return alerts
    
    async def _execute_rebalance_effect(self) -> List[Alert]:
        """Execute rebalance effect strategies"""
        alerts = []
        
        # Example: Pick worst performer mid-month
        today = datetime.now()
        tdom = today.day
        
        if tdom == 15:  # Mid-month decision
            alerts.append(Alert(
                strategy_id="rebalance_effect",
                strategy_name="Rebalance Effect Strategy",
                alert_type=AlertType.BUY,
                symbol="IWM",
                action="buy",
                confidence=0.7,
                timestamp=today,
                reasoning="Mid-month rebalance: IWM is worst performer",
                performance_metrics={"cagr": 0.11, "total_return": 0.14}
            ))
        
        return alerts
    
    async def get_strategy_performance(self, strategy_id: str) -> StrategyPerformance:
        """Get performance metrics for a strategy"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        
        # This would calculate actual performance metrics
        # For now, returning example data
        return StrategyPerformance(
            strategy_id=strategy_id,
            strategy_name=self.strategies[strategy_id]["name"],
            cagr=0.15,
            total_return=0.20,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            win_rate=0.65,
            last_updated=datetime.now()
        )

# FastAPI Application
app = FastAPI(
    title="Seasonal Trading Strategies API",
    description="Backend service for seasonal trading strategy alerts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_manager = DataManager()
strategy_engine = StrategyEngine(data_manager)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Seasonal Trading Strategies API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/api/strategies")
async def list_strategies():
    """List all available strategies"""
    return {
        "strategies": [
            {
                "id": strategy_id,
                "name": config["name"],
                "description": config["description"],
                "symbols": config["symbols"]
            }
            for strategy_id, config in strategy_engine.strategies.items()
        ]
    }

@app.get("/api/strategies/{strategy_id}/execute")
async def execute_strategy(strategy_id: str):
    """Execute a specific strategy and return alerts"""
    try:
        alerts = await strategy_engine.execute_strategy(strategy_id)
        return {
            "strategy_id": strategy_id,
            "alerts": [alert.dict() for alert in alerts],
            "timestamp": datetime.now(),
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/{strategy_id}/performance")
async def get_strategy_performance(strategy_id: str):
    """Get performance metrics for a strategy"""
    try:
        performance = await strategy_engine.get_strategy_performance(strategy_id)
        return performance.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts/generate")
async def generate_all_alerts():
    """Generate alerts for all strategies"""
    all_alerts = []
    
    for strategy_id in strategy_engine.strategies.keys():
        try:
            alerts = await strategy_engine.execute_strategy(strategy_id)
            all_alerts.extend(alerts)
        except Exception as e:
            logger.error(f"Error generating alerts for {strategy_id}: {e}")
    
    return {
        "alerts": [alert.dict() for alert in all_alerts],
        "timestamp": datetime.now(),
        "count": len(all_alerts)
    }

@app.get("/api/data/{symbol}")
async def get_symbol_data(symbol: str, limit: Optional[int] = None):
    """Get data for a specific symbol"""
    try:
        df = data_manager.load_symbol_data(symbol, limit)
        return {
            "symbol": symbol,
            "data": df.to_dict('records'),
            "count": len(df),
            "last_updated": data_manager.last_update.get(symbol)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/data/{symbol}/latest")
async def get_latest_data(symbol: str):
    """Get latest data point for a symbol"""
    try:
        latest = data_manager.get_latest_data(symbol)
        return {
            "symbol": symbol,
            "latest": latest,
            "is_fresh": data_manager.is_data_fresh(symbol)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "data_folder": str(data_manager.data_folder),
        "strategies_count": len(strategy_engine.strategies)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

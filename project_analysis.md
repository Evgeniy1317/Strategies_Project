# Seasonal Trading Strategies Backend Analysis

## Project Overview
This project contains 8 Python trading strategies that need to be converted into a backend service/API for generating alerts. The strategies are seasonal trading algorithms that analyze historical data to make trading decisions.

## Data Structure
- **Location**: `SeasonalBackend/Data/`
- **Format**: CSV files with OHLC data
- **Symbols**: 100+ trading instruments including:
  - ETFs (SPY, QQQ, GLD, TLT, etc.)
  - Futures (AD, BC, BO, BP, etc.)
  - Forex pairs (FXAUDUSD, FXEURUSD, etc.)
  - Commodities (GC, CL, etc.)

## The 8 Python Strategies

### 1. DayOfTheWeek_Final.py
- **Purpose**: Day-of-week futures strategies
- **Logic**: Evaluates conditions on day t's close, enters next bar's open, exits following bar's open
- **Hold Period**: 1-day hold
- **Output**: Equity curves and performance metrics

### 2. GateStrategies_Final.py
- **Purpose**: Gate-based strategies with A/B symbol selection
- **Logic**: Hold symbol A during gate window (month or Q1), evaluate performance, then hold A or B based on gate return
- **Examples**: SMH vs SLV in February, GLD vs QQQ in October
- **Output**: Detail CSV, equity curves, performance vs SPY

### 3. RankingStrategies_Final.py
- **Purpose**: Monthly ranking-based strategies
- **Logic**: Rank symbols by historical monthly performance, buy worst/best performers
- **Modes**: "worst", "best", "worst2" (two worst equally weighted)
- **Output**: Daily detail, equity curves, summary metrics

### 4. SwitchStrategies_Final.py
- **Purpose**: Fixed month-to-symbol rotation
- **Logic**: Assign specific symbols to specific months, rotate based on calendar
- **Examples**: GLD in months [1,4,8,10,12], SMH in others
- **Output**: Monthly rotation details, equity performance

### 5. SeasonalMovingAverages_Final.py
- **Purpose**: Moving average-based monthly gate strategies
- **Logic**: Use 200D/50D SMA to decide between symbol pairs
- **Examples**: SMH vs GLD based on SMH > SMA200(SMH)
- **Output**: Monthly decisions, equity curves, performance metrics

### 6. TDOMSwitch_Final.py
- **Purpose**: Half-month rotation by Trading Day of Month
- **Logic**: Hold symbol A for TDOM 1-K, symbol B for TDOM K+1 to EOM
- **Examples**: SMH for TDOM 1-12, TLT for TDOM 13-EOM
- **Output**: Daily rotation details, equity performance

### 7. WeeklySwitchStrategies_Final.py
- **Purpose**: Week-of-month rotation strategies
- **Logic**: Assign symbols to trading weeks (1-4), rotate weekly
- **Examples**: EEM, GLD, TLT, QQQ for weeks 1,2,3,4/5
- **Output**: Weekly rotation details, equity curves

### 8. RebalanceEffect_Final.py
- **Purpose**: Rebalance effect strategies
- **Logic**: Pick worst performer mid-month, hold until month-end
- **Decision Points**: TDOM 7 or 15 for different strategies
- **Output**: Mid-month decisions, end-of-month performance

## Data Format
All data files follow this structure:
```
Date,Time,Open,High,Low,Close,Vol,OI
05/15/2001,17:00:00,0.18710,0.18710,0.18140,0.18710,315,24560
```

## Key Requirements for Backend Service
1. **Alert Generation**: Generate alerts when strategies should buy/sell
2. **Performance Metrics**: Return CAGR and Total Return for each strategy
3. **Data Updates**: Assume data updated by 18:00 EST daily
4. **API Response**: Return strategy recommendations with symbols to buy/sell
5. **No Lookahead**: All strategies are designed to work without future data

## Technical Considerations
- All strategies use intersection calendars (common trading days)
- Benchmark is typically SPY buy & hold
- Starting equity: $100,000
- Strategies handle missing data and different symbol availability
- Output includes detailed CSV files and equity curve plots

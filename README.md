# Seasonal Trading Strategies Backend Service

This project converts 8 existing Python trading strategies into a robust backend service/API for generating trading alerts.

## Project Overview

The project contains 8 seasonal trading strategies that analyze historical data to make trading decisions:

1. **Day of Week Strategies** - Day-of-week futures strategies with 1-day hold
2. **Gate Strategies** - Gate-based strategies with A/B symbol selection
3. **Ranking Strategies** - Monthly ranking-based strategies
4. **Switch Strategies** - Fixed month-to-symbol rotation
5. **Seasonal Moving Averages** - Moving average-based monthly gate strategies
6. **TDOM Switch** - Half-month rotation by Trading Day of Month
7. **Weekly Switch** - Week-of-month rotation strategies
8. **Rebalance Effect** - Rebalance effect strategies

## Data Structure

- **Location**: `SeasonalBackend/Data/`
- **Format**: CSV files with OHLC data
- **Symbols**: 100+ trading instruments including ETFs, futures, forex, and commodities
- **Update Schedule**: Data updated by 18:00 EST daily

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Production Backend Service

```bash
python production_server.py
```

This runs the production server with real CSV data integration.

### 3. Test the API

```bash
python test_backend.py
```

### 4. Visualize Data

```bash
python data_visualizer.py
```

## API Endpoints

### Core Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `GET /api/strategies` - List all strategies
- `GET /api/strategies/{strategy_id}/execute` - Execute specific strategy
- `GET /api/strategies/{strategy_id}/performance` - Get strategy performance
- `GET /api/alerts/generate` - Generate alerts for all strategies

### Data Endpoints

- `GET /api/data/{symbol}` - Get symbol data
- `GET /api/data/{symbol}/latest` - Get latest data point
- `GET /api/data/{symbol}/range/{start_date}/{end_date}` - Get data range

## Strategy Details

### 1. Day of Week Strategies
- **Symbols**: ES, GC, CL, AD, BP
- **Logic**: Evaluates conditions on day t's close, enters next bar's open
- **Hold Period**: 1-day hold

### 2. Gate Strategies
- **Symbols**: SMH, SLV, GLD, QQQ, JNK
- **Logic**: Hold symbol A during gate window, evaluate performance, then hold A or B
- **Examples**: SMH vs SLV in February, GLD vs QQQ in October

### 3. Ranking Strategies
- **Symbols**: GDX, SMH, GLD, XLE, XHB, QQQ, XLK, SPY
- **Logic**: Rank symbols by historical monthly performance, buy worst/best performers
- **Modes**: "worst", "best", "worst2" (two worst equally weighted)

### 4. Switch Strategies
- **Symbols**: GLD, SMH, XLU
- **Logic**: Assign specific symbols to specific months, rotate based on calendar
- **Examples**: GLD in months [1,4,8,10,12], SMH in others

### 5. Seasonal Moving Averages
- **Symbols**: SMH, GLD, XLK, QQQ, SPY, XLE
- **Logic**: Use 200D/50D SMA to decide between symbol pairs
- **Examples**: SMH vs GLD based on SMH > SMA200(SMH)

### 6. TDOM Switch
- **Symbols**: SMH, TLT, QQQ, GLD
- **Logic**: Hold symbol A for TDOM 1-K, symbol B for TDOM K+1 to EOM
- **Examples**: SMH for TDOM 1-12, TLT for TDOM 13-EOM

### 7. Weekly Switch
- **Symbols**: EEM, GLD, TLT, QQQ, XME, XLK, XLV, XLY
- **Logic**: Assign symbols to trading weeks (1-4), rotate weekly
- **Examples**: EEM, GLD, TLT, QQQ for weeks 1,2,3,4/5

### 8. Rebalance Effect
- **Symbols**: TLT, IWM, SPY, SMH, HYG, EEM, DIA
- **Logic**: Pick worst performer mid-month, hold until month-end
- **Decision Points**: TDOM 7 or 15 for different strategies

## Alert Types

- **BUY**: Buy signal for a specific symbol
- **SELL**: Sell signal for a specific symbol
- **HOLD**: Hold current position
- **SWITCH**: Switch from one symbol to another

## Performance Metrics

Each strategy returns:
- **CAGR**: Compound Annual Growth Rate
- **Total Return**: Total return over the period
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of winning trades

## Example Usage

### Generate All Alerts

```python
import requests

response = requests.get("http://localhost:8000/api/alerts/generate")
alerts = response.json()

for alert in alerts['alerts']:
    print(f"{alert['strategy_name']}: {alert['action']} {alert['symbol']}")
```

### Execute Specific Strategy

```python
response = requests.get("http://localhost:8000/api/strategies/gate/execute")
strategy_alerts = response.json()

for alert in strategy_alerts['alerts']:
    print(f"Alert: {alert['action']} {alert['symbol']} (confidence: {alert['confidence']})")
```

### Get Strategy Performance

```python
response = requests.get("http://localhost:8000/api/strategies/gate/performance")
performance = response.json()

print(f"CAGR: {performance['cagr']:.2%}")
print(f"Total Return: {performance['total_return']:.2%}")
```

## Data Visualization

The `data_visualizer.py` script provides:

- Data coverage analysis
- Symbol performance plots
- Strategy data requirements
- Backend service requirements

Run it to understand the data structure and requirements:

```bash
python data_visualizer.py
```

## Architecture

The backend service is built with:

- **FastAPI**: Modern, fast web framework for building APIs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Pydantic**: Data validation and serialization

## Key Features

- **No Lookahead**: All strategies work without future data
- **Real-time Alerts**: Generate alerts based on current market conditions
- **Performance Tracking**: Monitor strategy performance over time
- **Data Caching**: Efficient data loading and caching
- **RESTful API**: Clean, documented API endpoints
- **Error Handling**: Robust error handling and logging

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python backend_architecture.py
```

### Production Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run with uvicorn
uvicorn backend_architecture:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "backend_architecture:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing

Run the test suite:

```bash
python test_backend.py
```

This will test all API endpoints and demonstrate usage.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or support, please open an issue in the repository.

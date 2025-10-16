#!/usr/bin/env python3
"""
Test script for the Seasonal Trading Strategies Backend
This script tests the API endpoints and demonstrates usage.
"""

import requests
import json
from datetime import datetime
import time

class BackendTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """Test health endpoint"""
        print("Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                print(f"Response: {response.json()}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
    
    def test_list_strategies(self):
        """Test strategies listing"""
        print("\nTesting strategies listing...")
        try:
            response = requests.get(f"{self.base_url}/api/strategies")
            if response.status_code == 200:
                print("✅ Strategies listing successful")
                strategies = response.json()["strategies"]
                print(f"Found {len(strategies)} strategies:")
                for strategy in strategies:
                    print(f"  - {strategy['id']}: {strategy['name']}")
            else:
                print(f"❌ Strategies listing failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Strategies listing error: {e}")
    
    def test_strategy_execution(self, strategy_id="day_of_week"):
        """Test strategy execution"""
        print(f"\nTesting strategy execution for {strategy_id}...")
        try:
            response = requests.get(f"{self.base_url}/api/strategies/{strategy_id}/execute")
            if response.status_code == 200:
                print("✅ Strategy execution successful")
                result = response.json()
                print(f"Generated {result['count']} alerts")
                for alert in result['alerts']:
                    print(f"  - {alert['action']} {alert['symbol']} (confidence: {alert['confidence']})")
            else:
                print(f"❌ Strategy execution failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Strategy execution error: {e}")
    
    def test_strategy_performance(self, strategy_id="day_of_week"):
        """Test strategy performance metrics"""
        print(f"\nTesting strategy performance for {strategy_id}...")
        try:
            response = requests.get(f"{self.base_url}/api/strategies/{strategy_id}/performance")
            if response.status_code == 200:
                print("✅ Strategy performance successful")
                performance = response.json()
                print(f"CAGR: {performance['cagr']:.2%}")
                print(f"Total Return: {performance['total_return']:.2%}")
                print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            else:
                print(f"❌ Strategy performance failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Strategy performance error: {e}")
    
    def test_generate_all_alerts(self):
        """Test generating alerts for all strategies"""
        print("\nTesting all alerts generation...")
        try:
            response = requests.get(f"{self.base_url}/api/alerts/generate")
            if response.status_code == 200:
                print("✅ All alerts generation successful")
                result = response.json()
                print(f"Generated {result['count']} total alerts")
                
                # Group alerts by strategy
                strategy_alerts = {}
                for alert in result['alerts']:
                    strategy_id = alert['strategy_id']
                    if strategy_id not in strategy_alerts:
                        strategy_alerts[strategy_id] = []
                    strategy_alerts[strategy_id].append(alert)
                
                for strategy_id, alerts in strategy_alerts.items():
                    print(f"  {strategy_id}: {len(alerts)} alerts")
            else:
                print(f"❌ All alerts generation failed: {response.status_code}")
        except Exception as e:
            print(f"❌ All alerts generation error: {e}")
    
    def test_data_endpoints(self, symbol="SPY"):
        """Test data endpoints"""
        print(f"\nTesting data endpoints for {symbol}...")
        try:
            # Test latest data
            response = requests.get(f"{self.base_url}/api/data/{symbol}/latest")
            if response.status_code == 200:
                print("✅ Latest data endpoint successful")
                data = response.json()
                print(f"Latest {symbol} data: {data['latest']}")
            else:
                print(f"❌ Latest data endpoint failed: {response.status_code}")
            
            # Test data with limit
            response = requests.get(f"{self.base_url}/api/data/{symbol}?limit=5")
            if response.status_code == 200:
                print("✅ Data with limit successful")
                data = response.json()
                print(f"Retrieved {data['count']} records")
            else:
                print(f"❌ Data with limit failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Data endpoints error: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Backend API Tests")
        print("=" * 50)
        
        # Test health
        self.test_health()
        
        # Test strategies
        self.test_list_strategies()
        
        # Test individual strategy execution
        strategies = ["day_of_week", "gate", "ranking", "switch"]
        for strategy in strategies:
            self.test_strategy_execution(strategy)
            self.test_strategy_performance(strategy)
        
        # Test all alerts generation
        self.test_generate_all_alerts()
        
        # Test data endpoints
        self.test_data_endpoints("SPY")
        self.test_data_endpoints("QQQ")
        
        print("\n🎉 All tests completed!")

def main():
    """Main function"""
    print("Seasonal Trading Strategies Backend Tester")
    print("=" * 50)
    
    # Check if server is running
    tester = BackendTester()
    
    print("Make sure the backend server is running:")
    print("python backend_architecture.py")
    print("\nOr with uvicorn:")
    print("uvicorn backend_architecture:app --reload")
    print("\nPress Enter to continue with tests...")
    input()
    
    # Run tests
    tester.run_all_tests()

if __name__ == "__main__":
    main()

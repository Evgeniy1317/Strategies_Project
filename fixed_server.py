#!/usr/bin/env python3
"""
Fixed HTTP Server for Seasonal Trading Strategies
This version properly serves HTML files
"""

import http.server
import socketserver
import json
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs

class TradingStrategyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        try:
            if path == '/' or path == '/dashboard':
                # Serve the HTML dashboard
                self.serve_html_file('web_interface.html')
                return
            elif path == '/api/health':
                self.send_json_response({
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "server": "Simple HTTP Server"
                })
                return
            elif path == '/api/strategies':
                self.send_json_response({
                    "strategies": [
                        {
                            "id": "day_of_week",
                            "name": "Day of Week Strategies",
                            "description": "Day-of-week futures strategies with 1-day hold",
                            "symbols": ["ES", "GC", "CL", "AD", "BP"]
                        },
                        {
                            "id": "gate",
                            "name": "Gate Strategies",
                            "description": "Gate-based strategies with A/B symbol selection",
                            "symbols": ["SMH", "SLV", "GLD", "QQQ", "JNK"]
                        },
                        {
                            "id": "ranking",
                            "name": "Ranking Strategies",
                            "description": "Monthly ranking-based strategies",
                            "symbols": ["GDX", "SMH", "GLD", "XLE", "XHB", "QQQ", "XLK", "SPY"]
                        },
                        {
                            "id": "switch",
                            "name": "Switch Strategies",
                            "description": "Fixed month-to-symbol rotation",
                            "symbols": ["GLD", "SMH", "XLU"]
                        },
                        {
                            "id": "seasonal_ma",
                            "name": "Seasonal Moving Averages",
                            "description": "Moving average-based monthly gate strategies",
                            "symbols": ["SMH", "GLD", "XLK", "QQQ", "SPY", "XLE"]
                        },
                        {
                            "id": "tdom_switch",
                            "name": "TDOM Switch",
                            "description": "Half-month rotation by Trading Day of Month",
                            "symbols": ["SMH", "TLT", "QQQ", "GLD"]
                        },
                        {
                            "id": "weekly_switch",
                            "name": "Weekly Switch",
                            "description": "Week-of-month rotation strategies",
                            "symbols": ["EEM", "GLD", "TLT", "QQQ", "XME", "XLK", "XLV", "XLY"]
                        },
                        {
                            "id": "rebalance_effect",
                            "name": "Rebalance Effect",
                            "description": "Rebalance effect strategies",
                            "symbols": ["TLT", "IWM", "SPY", "SMH", "HYG", "EEM", "DIA"]
                        }
                    ]
                })
                return
            elif path.startswith('/api/strategies/') and path.endswith('/execute'):
                strategy_id = path.split('/')[-2]
                response = self._execute_strategy(strategy_id)
                self.send_json_response(response)
                return
            elif path.startswith('/api/strategies/') and path.endswith('/performance'):
                strategy_id = path.split('/')[-2]
                response = self._get_strategy_performance(strategy_id)
                self.send_json_response(response)
                return
            elif path == '/api/alerts/generate':
                response = self._generate_all_alerts()
                self.send_json_response(response)
                return
            else:
                self.send_error(404, "Endpoint not found")
                return
                
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def serve_html_file(self, filename):
        """Serve HTML files with proper content type"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"File not found: {filename}")
        except Exception as e:
            self.send_error(500, f"Error serving file: {str(e)}")
    
    def _execute_strategy(self, strategy_id):
        """Execute a specific strategy"""
        strategies = {
            "day_of_week": {
                "name": "Day of Week Strategy",
                "alerts": [
                    {
                        "strategy_id": "day_of_week",
                        "strategy_name": "Day of Week Strategy",
                        "alert_type": "buy",
                        "symbol": "ES",
                        "action": "buy",
                        "confidence": 0.8,
                        "timestamp": datetime.now().isoformat(),
                        "reasoning": "Monday buy signal based on historical patterns",
                        "performance_metrics": {"cagr": 0.12, "total_return": 0.15}
                    }
                ]
            },
            "gate": {
                "name": "Gate Strategy",
                "alerts": [
                    {
                        "strategy_id": "gate",
                        "strategy_name": "SMH vs SLV February Gate",
                        "alert_type": "switch",
                        "symbol": "SLV",
                        "action": "switch to",
                        "target_symbol": "SMH",
                        "confidence": 0.75,
                        "timestamp": datetime.now().isoformat(),
                        "reasoning": "February gate strategy: switching from SLV to SMH",
                        "performance_metrics": {"cagr": 0.18, "total_return": 0.22}
                    }
                ]
            },
            "ranking": {
                "name": "Ranking Strategy",
                "alerts": [
                    {
                        "strategy_id": "ranking",
                        "strategy_name": "Monthly Ranking Strategy",
                        "alert_type": "buy",
                        "symbol": "XHB",
                        "action": "buy",
                        "confidence": 0.7,
                        "timestamp": datetime.now().isoformat(),
                        "reasoning": "Buying worst performer for current month",
                        "performance_metrics": {"cagr": 0.14, "total_return": 0.18}
                    }
                ]
            }
        }
        
        if strategy_id in strategies:
            return {
                "strategy_id": strategy_id,
                "alerts": strategies[strategy_id]["alerts"],
                "timestamp": datetime.now().isoformat(),
                "count": len(strategies[strategy_id]["alerts"])
            }
        else:
            return {"error": f"Strategy {strategy_id} not found"}
    
    def _generate_all_alerts(self):
        """Generate alerts for all strategies"""
        all_alerts = []
        
        strategies = ["day_of_week", "gate", "ranking", "switch", "seasonal_ma", "tdom_switch", "weekly_switch", "rebalance_effect"]
        
        for strategy_id in strategies:
            strategy_result = self._execute_strategy(strategy_id)
            if "alerts" in strategy_result:
                all_alerts.extend(strategy_result["alerts"])
        
        return {
            "alerts": all_alerts,
            "timestamp": datetime.now().isoformat(),
            "count": len(all_alerts)
        }
    
    def _get_strategy_performance(self, strategy_id):
        """Get performance metrics for a strategy"""
        performance_data = {
            "day_of_week": {
                "strategy_id": "day_of_week",
                "strategy_name": "Day of Week Strategy",
                "cagr": 0.15,
                "total_return": 0.20,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.65,
                "last_updated": datetime.now().isoformat()
            },
            "gate": {
                "strategy_id": "gate",
                "strategy_name": "Gate Strategy",
                "cagr": 0.18,
                "total_return": 0.22,
                "sharpe_ratio": 1.4,
                "max_drawdown": 0.06,
                "win_rate": 0.70,
                "last_updated": datetime.now().isoformat()
            },
            "ranking": {
                "strategy_id": "ranking",
                "strategy_name": "Ranking Strategy",
                "cagr": 0.14,
                "total_return": 0.18,
                "sharpe_ratio": 1.1,
                "max_drawdown": 0.09,
                "win_rate": 0.62,
                "last_updated": datetime.now().isoformat()
            },
            "switch": {
                "strategy_id": "switch",
                "strategy_name": "Switch Strategy",
                "cagr": 0.16,
                "total_return": 0.20,
                "sharpe_ratio": 1.3,
                "max_drawdown": 0.07,
                "win_rate": 0.68,
                "last_updated": datetime.now().isoformat()
            },
            "seasonal_ma": {
                "strategy_id": "seasonal_ma",
                "strategy_name": "Seasonal Moving Averages",
                "cagr": 0.17,
                "total_return": 0.21,
                "sharpe_ratio": 1.35,
                "max_drawdown": 0.065,
                "win_rate": 0.69,
                "last_updated": datetime.now().isoformat()
            },
            "tdom_switch": {
                "strategy_id": "tdom_switch",
                "strategy_name": "TDOM Switch",
                "cagr": 0.15,
                "total_return": 0.19,
                "sharpe_ratio": 1.25,
                "max_drawdown": 0.075,
                "win_rate": 0.67,
                "last_updated": datetime.now().isoformat()
            },
            "weekly_switch": {
                "strategy_id": "weekly_switch",
                "strategy_name": "Weekly Switch",
                "cagr": 0.13,
                "total_return": 0.17,
                "sharpe_ratio": 1.15,
                "max_drawdown": 0.08,
                "win_rate": 0.64,
                "last_updated": datetime.now().isoformat()
            },
            "rebalance_effect": {
                "strategy_id": "rebalance_effect",
                "strategy_name": "Rebalance Effect",
                "cagr": 0.11,
                "total_return": 0.14,
                "sharpe_ratio": 1.05,
                "max_drawdown": 0.09,
                "win_rate": 0.60,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        if strategy_id in performance_data:
            return performance_data[strategy_id]
        else:
            return {"error": f"Strategy {strategy_id} not found"}
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    """Start the server"""
    PORT = 8000
    
    print(f"🚀 Starting Seasonal Trading Strategies Server on port {PORT}")
    print(f"📊 Open your browser and go to: http://localhost:{PORT}")
    print(f"🔗 API Documentation: http://localhost:{PORT}/api/strategies")
    print(f"⚡ Health Check: http://localhost:{PORT}/api/health")
    print(f"📈 Generate Alerts: http://localhost:{PORT}/api/alerts/generate")
    print("\nPress Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", PORT), TradingStrategyHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()

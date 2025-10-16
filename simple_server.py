#!/usr/bin/env python3
"""
Simple HTTP Server for Seasonal Trading Strategies
This is a basic version that works without FastAPI dependencies
"""

import http.server
import socketserver
import json
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import mimetypes

class TradingStrategyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        # Set CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            if path == '/':
                # Serve the HTML dashboard
                self.serve_html_file('web_interface.html')
                return
            elif path == '/dashboard':
                # Serve the HTML dashboard
                self.serve_html_file('web_interface.html')
                return
            elif path == '/api/health':
                response = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "server": "Simple HTTP Server"
                }
            elif path == '/api/strategies':
                response = {
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
                }
            elif path.startswith('/api/strategies/') and path.endswith('/execute'):
                strategy_id = path.split('/')[-2]
                response = self._execute_strategy(strategy_id)
            elif path == '/api/alerts/generate':
                response = self._generate_all_alerts()
            else:
                response = {"error": "Endpoint not found", "path": path}
            
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response, indent=2).encode())
    
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
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def serve_html_file(self, filename):
        """Serve HTML files"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<h1>404 - File not found</h1>')
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f'<h1>500 - Server Error: {str(e)}</h1>'.encode())

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

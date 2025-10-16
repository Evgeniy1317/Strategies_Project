#!/usr/bin/env python3
"""
Notification Service for Trading Alerts
This service can send alerts via email, SMS, or webhooks
"""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from datetime import datetime

class NotificationService:
    def __init__(self):
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email': 'your-email@gmail.com',
            'password': 'your-app-password'
        }
        
        self.webhook_urls = [
            'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK'
        ]
    
    def send_email_alert(self, alert, recipient_email):
        """Send email notification for trading alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email']
            msg['To'] = recipient_email
            msg['Subject'] = f"Trading Alert: {alert['strategy_name']}"
            
            body = f"""
            Trading Alert Generated!
            
            Strategy: {alert['strategy_name']}
            Action: {alert['action']} {alert['symbol']}
            Confidence: {alert['confidence']:.1%}
            Reasoning: {alert['reasoning']}
            
            Performance Metrics:
            - CAGR: {alert['performance_metrics']['cagr']:.1%}
            - Total Return: {alert['performance_metrics']['total_return']:.1%}
            
            Time: {alert['timestamp']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['email'], recipient_email, text)
            server.quit()
            
            print(f"Email alert sent to {recipient_email}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_slack_alert(self, alert):
        """Send Slack notification for trading alert"""
        try:
            slack_message = {
                "text": f"🚨 Trading Alert: {alert['strategy_name']}",
                "attachments": [
                    {
                        "color": "good" if alert['alert_type'] == 'buy' else "danger",
                        "fields": [
                            {"title": "Action", "value": f"{alert['action']} {alert['symbol']}", "short": True},
                            {"title": "Confidence", "value": f"{alert['confidence']:.1%}", "short": True},
                            {"title": "Reasoning", "value": alert['reasoning'], "short": False},
                            {"title": "CAGR", "value": f"{alert['performance_metrics']['cagr']:.1%}", "short": True},
                            {"title": "Total Return", "value": f"{alert['performance_metrics']['total_return']:.1%}", "short": True}
                        ]
                    }
                ]
            }
            
            response = requests.post(self.webhook_urls[0], json=slack_message)
            if response.status_code == 200:
                print("Slack alert sent successfully")
                return True
            else:
                print(f"Slack alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending Slack alert: {e}")
            return False
    
    def send_discord_alert(self, alert):
        """Send Discord notification for trading alert"""
        try:
            discord_message = {
                "content": f"🚨 **Trading Alert: {alert['strategy_name']}**",
                "embeds": [
                    {
                        "title": f"{alert['action']} {alert['symbol']}",
                        "description": alert['reasoning'],
                        "color": 0x00ff00 if alert['alert_type'] == 'buy' else 0xff0000,
                        "fields": [
                            {"name": "Confidence", "value": f"{alert['confidence']:.1%}", "inline": True},
                            {"name": "CAGR", "value": f"{alert['performance_metrics']['cagr']:.1%}", "inline": True},
                            {"name": "Total Return", "value": f"{alert['performance_metrics']['total_return']:.1%}", "inline": True}
                        ],
                        "timestamp": alert['timestamp']
                    }
                ]
            }
            
            response = requests.post(self.webhook_urls[1], json=discord_message)
            if response.status_code == 204:
                print("Discord alert sent successfully")
                return True
            else:
                print(f"Discord alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending Discord alert: {e}")
            return False
    
    def send_sms_alert(self, alert, phone_number):
        """Send SMS notification (requires Twilio or similar service)"""
        try:
            # This would require Twilio setup
            # from twilio.rest import Client
            
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(
            #     body=f"Trading Alert: {alert['action']} {alert['symbol']} - {alert['reasoning']}",
            #     from_='+1234567890',
            #     to=phone_number
            # )
            
            print(f"SMS alert would be sent to {phone_number}")
            return True
            
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False
    
    def send_all_notifications(self, alert, recipients):
        """Send alert to all configured notification channels"""
        results = {}
        
        # Email notifications
        if 'email' in recipients:
            for email in recipients['email']:
                results[f'email_{email}'] = self.send_email_alert(alert, email)
        
        # Slack notification
        if 'slack' in recipients and recipients['slack']:
            results['slack'] = self.send_slack_alert(alert)
        
        # Discord notification
        if 'discord' in recipients and recipients['discord']:
            results['discord'] = self.send_discord_alert(alert)
        
        # SMS notifications
        if 'sms' in recipients:
            for phone in recipients['sms']:
                results[f'sms_{phone}'] = self.send_sms_alert(alert, phone)
        
        return results

# Example usage
if __name__ == "__main__":
    # Example alert
    sample_alert = {
        "strategy_id": "ranking",
        "strategy_name": "Monthly Ranking Strategy",
        "alert_type": "buy",
        "symbol": "XLE",
        "action": "buy",
        "confidence": 0.7,
        "timestamp": datetime.now().isoformat(),
        "reasoning": "Buying worst performer for current month",
        "performance_metrics": {"cagr": 0.14, "total_return": 0.18}
    }
    
    # Notification service
    notification_service = NotificationService()
    
    # Recipients configuration
    recipients = {
        'email': ['trader1@example.com', 'trader2@example.com'],
        'slack': True,
        'discord': True,
        'sms': ['+1234567890']
    }
    
    # Send notifications
    results = notification_service.send_all_notifications(sample_alert, recipients)
    print(f"Notification results: {results}")

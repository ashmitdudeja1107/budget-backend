# backend/tasks/budget_alert.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class AlertConfig:
    """Configuration for budget alerts"""
    budget_threshold_warning: float = 0.7  # 70% of budget used
    budget_threshold_critical: float = 0.9  # 90% of budget used
    anomaly_threshold: float = 0.6  # Confidence threshold for anomaly alerts
    category_threshold: float = 0.8  # Category budget threshold
    enable_daily_summary: bool = True
    enable_budget_alerts: bool = True
    enable_anomaly_alerts: bool = True
    enable_category_alerts: bool = True

@dataclass
class Alert:
    """Represents a budget alert"""
    id: str
    type: str  # 'budget', 'anomaly', 'category', 'daily_summary'
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    timestamp: datetime
    data: Dict = None
    acknowledged: bool = False

class BudgetAlertSystem:
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
    
    def generate_alerts(self, expenses_df: pd.DataFrame, budget: float, 
                       category_budgets: Dict[str, float] = None) -> List[Alert]:
        """Generate all types of alerts based on expense data"""
        new_alerts = []
        
        if self.config.enable_budget_alerts:
            budget_alerts = self._check_budget_alerts(expenses_df, budget)
            new_alerts.extend(budget_alerts)
        
        if self.config.enable_anomaly_alerts:
            anomaly_alerts = self._check_anomaly_alerts(expenses_df)
            new_alerts.extend(anomaly_alerts)
        
        if self.config.enable_category_alerts and category_budgets:
            category_alerts = self._check_category_alerts(expenses_df, category_budgets)
            new_alerts.extend(category_alerts)
        
        if self.config.enable_daily_summary:
            summary_alert = self._generate_daily_summary(expenses_df, budget)
            if summary_alert:
                new_alerts.append(summary_alert)
        
        # Add new alerts to the system
        self.alerts.extend(new_alerts)
        
        return new_alerts
    
    def _check_budget_alerts(self, expenses_df: pd.DataFrame, budget: float) -> List[Alert]:
        """Check for budget-related alerts"""
        alerts = []
        total_spent = expenses_df['amount'].sum()
        budget_used_ratio = total_spent / budget
        
        alert_id = f"budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if budget_used_ratio >= 1.0:
            # Budget exceeded
            overspend = total_spent - budget
            alert = Alert(
                id=alert_id,
                type='budget',
                severity='critical',
                title='Budget Exceeded!',
                message=f'You have exceeded your budget by ₹{overspend:,.2f}. Total spent: ₹{total_spent:,.2f} of ₹{budget:,.2f}',
                timestamp=datetime.now(),
                data={
                    'budget': budget,
                    'spent': total_spent,
                    'overspend': overspend,
                    'percentage': budget_used_ratio * 100
                }
            )
            alerts.append(alert)
        
        elif budget_used_ratio >= self.config.budget_threshold_critical:
            # Critical threshold reached
            remaining = budget - total_spent
            alert = Alert(
                id=alert_id,
                type='budget',
                severity='critical',
                title='Budget Critical',
                message=f'You have used {budget_used_ratio*100:.1f}% of your budget. Only ₹{remaining:,.2f} remaining.',
                timestamp=datetime.now(),
                data={
                    'budget': budget,
                    'spent': total_spent,
                    'remaining': remaining,
                    'percentage': budget_used_ratio * 100
                }
            )
            alerts.append(alert)
        
        elif budget_used_ratio >= self.config.budget_threshold_warning:
            # Warning threshold reached
            remaining = budget - total_spent
            alert = Alert(
                id=alert_id,
                type='budget',
                severity='warning',
                title='Budget Warning',
                message=f'You have used {budget_used_ratio*100:.1f}% of your budget. ₹{remaining:,.2f} remaining.',
                timestamp=datetime.now(),
                data={
                    'budget': budget,
                    'spent': total_spent,
                    'remaining': remaining,
                    'percentage': budget_used_ratio * 100
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_anomaly_alerts(self, expenses_df: pd.DataFrame) -> List[Alert]:
        """Check for anomaly-related alerts"""
        alerts = []
        
        # Import ML manager here to avoid circular imports
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from utils.ml_helpers import MLModelManager
            
            ml_manager = MLModelManager()
            
            # Check recent expenses for anomalies (last 7 days)
            recent_date = expenses_df['date'].max() - timedelta(days=7)
            recent_expenses = expenses_df[expenses_df['date'] > recent_date]
            
            anomalies_found = []
            
            for _, expense in recent_expenses.iterrows():
                is_anomaly, confidence = ml_manager.detect_anomaly(expense['amount'])
                
                if is_anomaly and confidence >= self.config.anomaly_threshold:
                    anomalies_found.append({
                        'amount': expense['amount'],
                        'description': expense['description'],
                        'category': expense['category'],
                        'date': expense['date'],
                        'confidence': confidence
                    })
            
            if anomalies_found:
                # Create alert for anomalies
                alert_id = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if len(anomalies_found) == 1:
                    anomaly = anomalies_found[0]
                    title = 'Unusual Expense Detected'
                    message = f'Unusual expense: ₹{anomaly["amount"]:,.2f} for "{anomaly["description"]}" in {anomaly["category"]}'
                else:
                    title = f'{len(anomalies_found)} Unusual Expenses Detected'
                    total_anomaly_amount = sum(a['amount'] for a in anomalies_found)
                    message = f'Detected {len(anomalies_found)} unusual expenses totaling ₹{total_anomaly_amount:,.2f}'
                
                alert = Alert(
                    id=alert_id,
                    type='anomaly',
                    severity='warning',
                    title=title,
                    message=message,
                    timestamp=datetime.now(),
                    data={
                        'anomalies': anomalies_found,
                        'count': len(anomalies_found),
                        'total_amount': sum(a['amount'] for a in anomalies_found)
                    }
                )
                alerts.append(alert)
        
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
        
        return alerts
    
    def _check_category_alerts(self, expenses_df: pd.DataFrame, 
                              category_budgets: Dict[str, float]) -> List[Alert]:
        """Check for category budget alerts"""
        alerts = []
        
        # Calculate spending by category
        category_spending = expenses_df.groupby('category')['amount'].sum()
        
        for category, budget_limit in category_budgets.items():
            if category in category_spending:
                spent = category_spending[category]
                usage_ratio = spent / budget_limit
                
                alert_id = f"category_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if usage_ratio >= 1.0:
                    # Category budget exceeded
                    overspend = spent - budget_limit
                    alert = Alert(
                        id=alert_id,
                        type='category',
                        severity='critical',
                        title=f'{category} Budget Exceeded',
                        message=f'You have exceeded your {category} budget by ₹{overspend:,.2f}',
                        timestamp=datetime.now(),
                        data={
                            'category': category,
                            'budget': budget_limit,
                            'spent': spent,
                            'overspend': overspend,
                            'percentage': usage_ratio * 100
                        }
                    )
                    alerts.append(alert)
                
                elif usage_ratio >= self.config.category_threshold:
                    # Category budget warning
                    remaining = budget_limit - spent
                    alert = Alert(
                        id=alert_id,
                        type='category',
                        severity='warning',
                        title=f'{category} Budget Warning',
                        message=f'You have used {usage_ratio*100:.1f}% of your {category} budget. ₹{remaining:,.2f} remaining.',
                        timestamp=datetime.now(),
                        data={
                            'category': category,
                            'budget': budget_limit,
                            'spent': spent,
                            'remaining': remaining,
                            'percentage': usage_ratio * 100
                        }
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _generate_daily_summary(self, expenses_df: pd.DataFrame, budget: float) -> Optional[Alert]:
        """Generate daily spending summary"""
        today = datetime.now().date()
        
        # Get today's expenses
        today_expenses = expenses_df[expenses_df['date'].dt.date == today]
        
        if len(today_expenses) == 0:
            return None
        
        today_total = today_expenses['amount'].sum()
        today_count = len(today_expenses)
        
        # Calculate averages
        total_spent = expenses_df['amount'].sum()
        days_in_data = (expenses_df['date'].max() - expenses_df['date'].min()).days + 1
        daily_average = total_spent / days_in_data
        
        # Budget remaining
        remaining_budget = budget - total_spent
        
        # Top category today
        if len(today_expenses) > 0:
            top_category_today = today_expenses.groupby('category')['amount'].sum().idxmax()
            top_category_amount = today_expenses.groupby('category')['amount'].sum().max()
        else:
            top_category_today = 'None'
            top_category_amount = 0
        
        # Determine message tone
        if today_total > daily_average * 1.5:
            severity = 'warning'
            tone = 'high spending day'
        elif today_total < daily_average * 0.5:
            severity = 'info'
            tone = 'light spending day'
        else:
            severity = 'info'
            tone = 'normal spending day'
        
        alert = Alert(
            id=f"daily_summary_{today.strftime('%Y%m%d')}",
            type='daily_summary',
            severity=severity,
            title='Daily Spending Summary',
            message=f'Today was a {tone}. You spent ₹{today_total:,.2f} across {today_count} transactions.',
            timestamp=datetime.now(),
            data={
                'date': today.isoformat(),
                'total_spent_today': today_total,
                'transaction_count': today_count,
                'daily_average': daily_average,
                'top_category': top_category_today,
                'top_category_amount': top_category_amount,
                'remaining_budget': remaining_budget,
                'comparison_to_average': (today_total / daily_average) if daily_average > 0 else 1
            }
        )
        
        return alert
    
    def get_active_alerts(self, severity_filter: str = None) -> List[Alert]:
        """Get active (unacknowledged) alerts"""
        active_alerts = [alert for alert in self.alerts if not alert.acknowledged]
        
        if severity_filter:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity_filter]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.alert_history.append(alert)
                return True
        return False
    
    def acknowledge_all_alerts(self):
        """Acknowledge all active alerts"""
        for alert in self.alerts:
            alert.acknowledged = True
            self.alert_history.append(alert)
    
    def clear_old_alerts(self, days: int = 30):
        """Clear alerts older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Move old alerts to history
        old_alerts = [alert for alert in self.alerts if alert.timestamp < cutoff_date]
        self.alert_history.extend(old_alerts)
        
        # Keep recent alerts
        self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_date]
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert system status"""
        active_alerts = self.get_active_alerts()
        
        severity_counts = {
            'critical': len([a for a in active_alerts if a.severity == 'critical']),
            'warning': len([a for a in active_alerts if a.severity == 'warning']),
            'info': len([a for a in active_alerts if a.severity == 'info'])
        }
        
        type_counts = {}
        for alert in active_alerts:
            type_counts[alert.type] = type_counts.get(alert.type, 0) + 1
        
        return {
            'total_active_alerts': len(active_alerts),
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'last_alert_time': active_alerts[0].timestamp.isoformat() if active_alerts else None,
            'config': {
                'budget_threshold_warning': self.config.budget_threshold_warning,
                'budget_threshold_critical': self.config.budget_threshold_critical,
                'anomaly_threshold': self.config.anomaly_threshold,
                'category_threshold': self.config.category_threshold
            }
        }
    
    def export_alerts(self, filepath: str = None) -> str:
        """Export alerts to JSON file"""
        if filepath is None:
            filepath = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert alerts to serializable format
        alerts_data = []
        for alert in self.alerts + self.alert_history:
            alert_dict = {
                'id': alert.id,
                'type': alert.type,
                'severity': alert.severity,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data,
                'acknowledged': alert.acknowledged
            }
            alerts_data.append(alert_dict)
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        return filepath
    
    def import_alerts(self, filepath: str):
        """Import alerts from JSON file"""
        with open(filepath, 'r') as f:
            alerts_data = json.load(f)
        
        for alert_dict in alerts_data:
            alert = Alert(
                id=alert_dict['id'],
                type=alert_dict['type'],
                severity=alert_dict['severity'],
                title=alert_dict['title'],
                message=alert_dict['message'],
                timestamp=datetime.fromisoformat(alert_dict['timestamp']),
                data=alert_dict.get('data'),
                acknowledged=alert_dict['acknowledged']
            )
            
            if alert.acknowledged:
                self.alert_history.append(alert)
            else:
                self.alerts.append(alert)
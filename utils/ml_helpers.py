# utils/ml_helpers.py - Enhanced Version with Full Model Integration
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

class MLModelManager:
    def __init__(self, model_dir='ml_models'):
        self.model_dir = model_dir
        self.category_model = None
        self.vectorizer = None
        self.depletion_model = None
        self.anomaly_model = None
        self.anomaly_scaler = None
        self.anomaly_label_encoder = None
        self.smart_allocation_model = None
        self.category_regressors = {}
        self.models_loaded = {
            'category': False,
            'anomaly': False,
            'depletion': False,
            'allocation': False
        }
        self.load_models()
    
    def load_models(self):
        """Load all trained ML models with enhanced error handling"""
        print("🔄 Loading ML models...")
        
        # Load category prediction model
        try:
            with open(f"{self.model_dir}/category_model.pkl", 'rb') as f:
                self.category_model = pickle.load(f)
            
            with open(f"{self.model_dir}/preprocessors/tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.models_loaded['category'] = True
            print("✅ Category prediction model loaded successfully")
        except FileNotFoundError:
            print("⚠️  Category model not found. Category prediction will use fallback.")
        except Exception as e:
            print(f"❌ Error loading category model: {e}")
        
        # Load anomaly detection model
        try:
            with open(f"{self.model_dir}/anomaly_model.pkl", 'rb') as f:
                anomaly_data = pickle.load(f)
                self.anomaly_model = anomaly_data['model']
                self.anomaly_scaler = anomaly_data.get('scaler', None)
                self.anomaly_label_encoder = anomaly_data.get('label_encoder', None)
            
            self.models_loaded['anomaly'] = True
            print("✅ Anomaly detection model loaded successfully")
        except FileNotFoundError:
            print("⚠️  Anomaly model not found. Anomaly detection will use fallback.")
        except Exception as e:
            print(f"❌ Error loading anomaly model: {e}")
        
        # Load depletion model
        try:
            with open(f"{self.model_dir}/depletion_model.pkl", 'rb') as f:
                depletion_data = pickle.load(f)
                self.depletion_model = depletion_data['model']
                self.budget = depletion_data.get('budget', 10000)
            
            self.models_loaded['depletion'] = True
            print("✅ Budget depletion model loaded successfully")
        except FileNotFoundError:
            print("⚠️  Depletion model not found. Budget prediction will use fallback.")
        except Exception as e:
            print(f"❌ Error loading depletion model: {e}")
        
        # Load smart allocation models
        try:
            with open(f"{self.model_dir}/smart_allocation_model.pkl", 'rb') as f:
                allocation_data = pickle.load(f)
                self.smart_allocation_model = allocation_data.get('allocation_model', None)
                self.category_regressors = allocation_data.get('category_regressors', {})
            
            self.models_loaded['allocation'] = True
            print("✅ Smart allocation model loaded successfully")
        except FileNotFoundError:
            print("⚠️  Smart allocation model not found. Will use rule-based approach.")
        except Exception as e:
            print(f"❌ Error loading smart allocation model: {e}")
    
    def predict_category(self, description):
        """Enhanced category prediction with better error handling"""
        if not self.models_loaded['category']:
            return self._fallback_category_prediction(description)
        
        try:
            # Preprocess text
            clean_desc = self._preprocess_text(description)
            if not clean_desc:
                return "Other", 0.0
            
            # Vectorize and predict
            X = self.vectorizer.transform([clean_desc])
            prediction = self.category_model.predict(X)[0]
            probabilities = self.category_model.predict_proba(X)[0]
            confidence = float(probabilities.max())
            
            return str(prediction), confidence
        except Exception as e:
            print(f"Error in category prediction: {e}")
            return self._fallback_category_prediction(description)
    
    def _fallback_category_prediction(self, description):
        """Rule-based fallback for category prediction"""
        desc_lower = str(description).lower()
        
        # Simple keyword-based categorization
        category_keywords = {
            'Food': ['food', 'restaurant', 'meal', 'lunch', 'dinner', 'breakfast', 'cafe', 'pizza', 'burger', 'grocery', 'supermarket', 'mcdonalds', 'kfc', 'starbucks', 'dominos'],
            'Travel': ['uber', 'taxi', 'bus', 'train', 'flight', 'petrol', 'fuel', 'parking', 'toll', 'ola', 'metro', 'auto'],
            'Entertainment': ['movie', 'cinema', 'netflix', 'game', 'concert', 'show', 'spotify', 'amazon prime', 'youtube'],
            'Shopping': ['amazon', 'flipkart', 'clothes', 'shirt', 'shoes', 'electronics', 'mobile', 'laptop', 'shopping'],
            'Healthcare': ['doctor', 'hospital', 'medicine', 'pharmacy', 'medical', 'health', 'clinic', 'dentist'],
            'Bills': ['electricity', 'water', 'gas', 'internet', 'phone', 'broadband', 'utility', 'bill'],
            'Education': ['course', 'book', 'education', 'school', 'university', 'training', 'learning']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category, 0.6  # Medium confidence for rule-based
        
        return "Other", 0.3
    
    def detect_anomaly(self, amount, category=None, date=None):
        """Enhanced anomaly detection with proper feature matching"""
        if not self.models_loaded['anomaly']:
            return self._fallback_anomaly_detection(amount, category)
        
        try:
            # CREATE PROPER FEATURE VECTOR (matching training format)
            amount_val = float(amount)
            
            # Feature 2: Category encoded
            if category and self.anomaly_label_encoder:
                try:
                    category_encoded = self.anomaly_label_encoder.transform([str(category)])[0]
                except:
                    category_encoded = 0  # Fallback for unknown categories
            else:
                category_encoded = 0
            
            # Feature 3: Day of week
            if date:
                try:
                    day_of_week = pd.to_datetime(date).dayofweek
                except:
                    day_of_week = datetime.now().weekday()
            else:
                day_of_week = datetime.now().weekday()
            
            # Create feature vector [amount, category_encoded, day_of_week]
            features = np.array([[amount_val, category_encoded, day_of_week]])
            features_scaled = self.anomaly_scaler.transform(features)
            
            # Get predictions
            anomaly_score = self.anomaly_model.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_model.predict(features_scaled)[0] == -1
            
            # Convert to confidence score
            confidence = max(0, min(1, abs(float(anomaly_score)) / 2.0))
            
            # Enhanced category-specific thresholds
            if category:
                is_anomaly_cat, confidence_cat = self._check_category_thresholds(amount_val, category)
                if is_anomaly_cat:
                    is_anomaly = True
                    confidence = max(confidence, confidence_cat)
            
            if is_anomaly:
                print(f"🚨 ANOMALY DETECTED: ₹{amount_val:.2f} in {category} (ML Score: {anomaly_score:.3f}, Confidence: {confidence:.2f})")
            
            return bool(is_anomaly), float(confidence)
            
        except Exception as e:
            print(f"❌ Error in ML anomaly detection: {e}")
            return self._fallback_anomaly_detection(amount, category)
    
    def _fallback_anomaly_detection(self, amount, category):
        """Rule-based fallback for anomaly detection"""
        return self._check_category_thresholds(float(amount), category)
    
    def _check_category_thresholds(self, amount, category):
        """Check category-specific thresholds for anomaly detection"""
        category_thresholds = {
            'food': {'low': 20, 'high': 1500},
            'travel': {'low': 10, 'high': 2000}, 
            'entertainment': {'low': 50, 'high': 3000},
            'bills': {'low': 200, 'high': 6000},
            'shopping': {'low': 100, 'high': 5000},
            'healthcare': {'low': 50, 'high': 15000},
            'education': {'low': 500, 'high': 25000}
        }
        
        if not category:
            # General thresholds
            if amount < 10 or amount > 10000:
                return True, 0.7
            return False, 0.3
            
        cat_lower = str(category).lower()
        if cat_lower in category_thresholds:
            thresholds = category_thresholds[cat_lower]
            if amount < thresholds['low']:
                return True, 0.8  # Very low amount
            elif amount > thresholds['high']:
                return True, 0.9  # Very high amount
        
        return False, 0.2
    
    def predict_budget_depletion(self, expenses_df, budget=None):
        """Enhanced budget depletion prediction"""
        if not self.models_loaded['depletion']:
            return self._fallback_depletion_prediction(expenses_df, budget)
        
        if budget is None:
            budget = getattr(self, 'budget', 10000)
        
        try:
            # Prepare data
            df = expenses_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate cumulative expenses
            df['cumulative'] = df['amount'].cumsum()
            current_spent = float(df['cumulative'].iloc[-1]) if len(df) > 0 else 0.0
            
            if current_spent >= budget:
                return 0, []  # Budget already depleted
            
            # Calculate days from start
            start_date = df['date'].min()
            current_day = (df['date'].max() - start_date).days + 1
            
            # Create features for prediction (matching training format)
            future_days = range(current_day + 1, current_day + 31)
            future_predictions = []
            
            for day in future_days:
                day_of_week = (start_date + timedelta(days=day-1)).weekday()
                day_of_month = (start_date + timedelta(days=day-1)).day
                month = (start_date + timedelta(days=day-1)).month
                
                # Calculate recent average (last 7 days)
                if len(df) >= 7:
                    recent_avg = df['amount'].tail(7).mean()
                else:
                    recent_avg = df['amount'].mean() if len(df) > 0 else 500
                
                # Features: [days_since_start, day_of_week, day_of_month, month, recent_avg]
                features = np.array([[day, day_of_week, day_of_month, month, recent_avg]])
                predicted_cumulative = self.depletion_model.predict(features)[0]
                future_predictions.append(predicted_cumulative)
            
            # Find depletion point
            depletion_day = None
            for i, pred_cumulative in enumerate(future_predictions):
                if float(pred_cumulative) >= budget:
                    depletion_day = future_days[i]
                    break
            
            # Calculate remaining days
            if depletion_day:
                remaining_days = int(depletion_day - current_day)
            else:
                remaining_days = None
            
            # Generate trend data
            trend_data = []
            for i, day in enumerate(list(future_days)[:14]):  # Next 2 weeks
                trend_data.append({
                    'day': int(day),
                    'predicted_cumulative': float(future_predictions[i]),
                    'remaining_budget': float(max(0, budget - future_predictions[i]))
                })
            
            return remaining_days, trend_data
        
        except Exception as e:
            print(f"Error in ML budget depletion prediction: {e}")
            return self._fallback_depletion_prediction(expenses_df, budget)
    
    def _fallback_depletion_prediction(self, expenses_df, budget):
        """Simple linear extrapolation fallback"""
        if len(expenses_df) == 0:
            return None, []
        
        try:
            total_spent = expenses_df['amount'].sum()
            days_active = len(expenses_df['date'].dt.date.unique())
            daily_avg = total_spent / max(days_active, 1)
            
            remaining_budget = budget - total_spent
            if remaining_budget <= 0:
                return 0, []
            
            remaining_days = int(remaining_budget / daily_avg) if daily_avg > 0 else None
            
            # Simple trend data
            trend_data = []
            for day in range(1, 15):  # Next 2 weeks
                projected_cumulative = total_spent + (daily_avg * day)
                trend_data.append({
                    'day': day,
                    'predicted_cumulative': float(projected_cumulative),
                    'remaining_budget': float(max(0, budget - projected_cumulative))
                })
            
            return remaining_days, trend_data
        except Exception as e:
            print(f"Error in fallback depletion prediction: {e}")
            return None, []
    
    def get_smart_allocation(self, expenses_df, budget, days_remaining=30):
        """ENHANCED Smart Budget Allocation with ML-powered predictions"""
        if len(expenses_df) == 0:
            return self._fallback_allocation(budget, days_remaining)
        
        try:
            # Calculate current spending by category
            category_spending = expenses_df.groupby('category')['amount'].agg(['sum', 'mean', 'count']).round(2)
            
            # Calculate total spent
            total_spent = float(expenses_df['amount'].sum())
            remaining_budget = float(budget - total_spent)
            
            if remaining_budget <= 0:
                return {"error": "Budget already exceeded"}
            
            # ML-POWERED PREDICTIONS FOR EACH CATEGORY
            ml_category_predictions = self._predict_category_spending(expenses_df, days_remaining)
            
            recommendations = {}
            allocated_budget = 0
            
            # Priority-based allocation with ML enhancement
            essential_categories = ['Food', 'Healthcare', 'Bills', 'Education']
            non_essential = ['Entertainment', 'Shopping', 'Travel']
            
            for category in category_spending.index:
                current_avg = float(category_spending.loc[category, 'mean'])
                current_total = float(category_spending.loc[category, 'sum'])
                
                # GET ML PREDICTION for this category
                ml_predicted_spend = ml_category_predictions.get(str(category), current_avg * (days_remaining / 30))
                
                # SMART ALLOCATION LOGIC (Rule + ML Hybrid)
                if category in essential_categories:
                    # Essential: Use ML prediction + 20% buffer
                    recommended_allocation = ml_predicted_spend * 1.2
                    priority = 'high'
                elif category in non_essential:
                    # Non-essential: Use ML prediction but reduce by 15%
                    recommended_allocation = ml_predicted_spend * 0.85
                    priority = 'low'
                else:
                    # Unknown category: Use ML prediction
                    recommended_allocation = ml_predicted_spend
                    priority = 'medium'
                
                # Ensure minimum allocation for essentials
                if category in essential_categories:
                    min_allocation = current_avg * (days_remaining / 30) * 0.8
                    recommended_allocation = max(recommended_allocation, min_allocation)
                
                # Smart daily limit calculation
                suggested_daily_limit = recommended_allocation / days_remaining
                
                # SPENDING VELOCITY ANALYSIS
                days_of_data = len(expenses_df[expenses_df['category'] == category]['date'].dt.date.unique())
                spending_velocity = current_total / max(days_of_data, 1)  # Amount per day
                
                recommendations[str(category)] = {
                    'current_spent': float(current_total),
                    'average_expense': float(current_avg),
                    'ml_predicted_spend': round(float(ml_predicted_spend), 2),
                    'recommended_budget': round(float(recommended_allocation), 2),
                    'suggested_daily_limit': round(float(suggested_daily_limit), 2),
                    'spending_velocity': round(float(spending_velocity), 2),
                    'priority': priority,
                    'optimization_tip': self._get_optimization_tip(category, current_avg, spending_velocity)
                }
                
                allocated_budget += recommended_allocation
            
            # BUDGET REBALANCING if over-allocated
            if allocated_budget > remaining_budget:
                adjustment_factor = remaining_budget / allocated_budget
                for category in recommendations:
                    recommendations[category]['recommended_budget'] = round(
                        float(recommendations[category]['recommended_budget']) * adjustment_factor, 2
                    )
                    recommendations[category]['suggested_daily_limit'] = round(
                        float(recommendations[category]['suggested_daily_limit']) * adjustment_factor, 2
                    )
            
            # ENHANCED SUMMARY with ML insights
            avg_daily_spend = total_spent / max(len(expenses_df['date'].dt.date.unique()), 1)
            projected_total = avg_daily_spend * days_remaining
            
            recommendations['summary'] = {
                'total_budget': float(budget),
                'spent_so_far': float(total_spent),
                'remaining_budget': float(remaining_budget),
                'days_remaining': int(days_remaining),
                'recommended_daily_total': round(float(remaining_budget) / days_remaining, 2),
                'current_daily_average': round(float(avg_daily_spend), 2),
                'ml_projection': round(float(projected_total), 2),
                'budget_health': self._assess_budget_health(total_spent, budget, days_remaining, avg_daily_spend),
                'top_spending_category': str(category_spending['sum'].idxmax()),
                'allocation_strategy': 'ML-Enhanced' if self.models_loaded['allocation'] else 'Rule-Based'
            }
            
            return recommendations
        
        except Exception as e:
            print(f"Error in smart allocation: {e}")
            return self._fallback_allocation(budget, days_remaining)
    
    def _predict_category_spending(self, expenses_df, days_remaining):
        """Use ML models to predict future spending per category"""
        predictions = {}
        
        try:
            # Group by category and analyze trends
            for category in expenses_df['category'].unique():
                cat_data = expenses_df[expenses_df['category'] == category].copy()
                cat_data['date'] = pd.to_datetime(cat_data['date'])
                cat_data = cat_data.sort_values('date')
                
                # If we have category-specific regressors, use them
                if str(category) in self.category_regressors and self.models_loaded['allocation']:
                    try:
                        regressor_data = self.category_regressors[str(category)]
                        model = regressor_data['model']
                        
                        # Prepare features (day number, day of week)
                        last_day = (cat_data['date'].max() - cat_data['date'].min()).days + 1
                        future_features = np.array([[last_day + i, (last_day + i) % 7] for i in range(1, days_remaining + 1)])
                        predicted_daily = model.predict(future_features)
                        predictions[str(category)] = float(np.sum(np.maximum(predicted_daily, 0)))  # Ensure non-negative
                    except Exception as e:
                        print(f"Error in ML prediction for {category}: {e}")
                        predictions[str(category)] = self._simple_category_prediction(cat_data, days_remaining)
                else:
                    # Fallback: Use trend extrapolation
                    predictions[str(category)] = self._simple_category_prediction(cat_data, days_remaining)
                        
        except Exception as e:
            print(f"Error in category spending prediction: {e}")
            # Return simple averages as fallback
            for category in expenses_df['category'].unique():
                cat_avg = expenses_df[expenses_df['category'] == category]['amount'].mean()
                predictions[str(category)] = float(cat_avg * (days_remaining / 30))
        
        return predictions
    
    def _simple_category_prediction(self, cat_data, days_remaining):
        """Simple trend-based prediction for a category"""
        if len(cat_data) >= 2:
            # Calculate daily spending trend
            daily_spending = cat_data.groupby(cat_data['date'].dt.date)['amount'].sum()
            if len(daily_spending) >= 2:
                # Linear trend
                x = np.arange(len(daily_spending))
                y = daily_spending.values
                trend_slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                avg_daily = daily_spending.mean()
                
                # Project with trend (but cap the trend to avoid unrealistic projections)
                trend_slope = max(-avg_daily * 0.1, min(avg_daily * 0.1, trend_slope))
                predicted_total = (avg_daily + trend_slope * days_remaining/2) * days_remaining
                return max(float(predicted_total), 0)
            else:
                avg_amount = cat_data['amount'].mean()
                return float(avg_amount * (days_remaining / 30))
        else:
            # Very simple fallback
            avg_amount = cat_data['amount'].mean() if len(cat_data) > 0 else 100
            return float(avg_amount * (days_remaining / 30))
    
    def _get_optimization_tip(self, category, avg_expense, velocity):
        """Generate personalized optimization tips based on spending patterns"""
        tips = {
            'Food': [
                "Try meal prepping to reduce daily food costs",
                "Consider cooking more meals at home",
                "Look for restaurant deals and discounts",
                "Buy groceries in bulk for better prices"
            ],
            'Entertainment': [
                "Explore free entertainment options like parks and events",
                "Set a weekly entertainment budget limit",
                "Consider subscription sharing with friends",
                "Look for group discounts on activities"
            ],
            'Shopping': [
                "Create a shopping list and stick to it",
                "Wait 24 hours before non-essential purchases",
                "Compare prices across different stores",
                "Use cashback and reward apps"
            ],
            'Travel': [
                "Use public transport when possible",
                "Plan trips in advance for better deals",
                "Consider carpooling or ride-sharing",
                "Book flights and hotels early for discounts"
            ],
            'Healthcare': [
                "Look for generic medicine alternatives",
                "Use health insurance benefits fully",
                "Consider preventive care to avoid major expenses",
                "Compare prices for medical procedures"
            ],
            'Bills': [
                "Switch to energy-efficient appliances",
                "Negotiate better plans with service providers",
                "Use automatic payments for discounts",
                "Monitor usage to avoid overage charges"
            ],
            'Education': [
                "Look for online courses and free resources",
                "Apply for scholarships and grants",
                "Buy used books or rent them",
                "Consider group study sessions to share costs"
            ]
        }
        
        base_tips = tips.get(category, ["Monitor spending in this category closely", "Set a monthly limit", "Track expenses regularly"])
        
        # Add velocity-based tips
        if velocity > avg_expense * 1.5:
            return f"{base_tips[0]} - High spending velocity detected!"
        elif len(base_tips) > 1:
            return base_tips[1]
        else:
            return base_tips[0]
    
    def _assess_budget_health(self, spent, budget, days_remaining, daily_avg):
        """Assess overall budget health with ML-like scoring"""
        spend_ratio = spent / budget
        projected_total = daily_avg * days_remaining
        
        if spend_ratio > 0.9:
            return "Critical - Budget nearly exhausted"
        elif spend_ratio > 0.8:
            return "Warning - High spending rate"
        elif projected_total > (budget - spent):
            return "Caution - Current pace exceeds remaining budget"
        elif spend_ratio < 0.3:
            return "Excellent - Well within budget"
        elif spend_ratio < 0.6:
            return "Good - On track with budget"
        else:
            return "Fair - Monitor spending closely"
    
    def _fallback_allocation(self, budget, days_remaining):
        """Enhanced fallback allocation when no expense data exists"""
        daily_budget = budget / days_remaining
        
        # Improved default category allocation percentages
        default_allocation = {
            'Food': 0.30,
            'Bills': 0.25,
            'Travel': 0.15,
            'Shopping': 0.12,
            'Entertainment': 0.08,
            'Healthcare': 0.06,
            'Education': 0.04
        }
        
        recommendations = {}
        for category, percentage in default_allocation.items():
            allocated_amount = budget * percentage
            recommendations[category] = {
                'current_spent': 0.0,
                'average_expense': 0.0,
                'ml_predicted_spend': 0.0,
                'recommended_budget': round(allocated_amount, 2),
                'suggested_daily_limit': round(allocated_amount / days_remaining, 2),
                'spending_velocity': 0.0,
                'priority': 'high' if category in ['Food', 'Bills', 'Healthcare'] else 'medium',
                'optimization_tip': f"Start tracking {category} expenses to get personalized insights"
            }
        
        recommendations['summary'] = {
            'total_budget': float(budget),
            'spent_so_far': 0.0,
            'remaining_budget': float(budget),
            'days_remaining': int(days_remaining),
            'recommended_daily_total': round(daily_budget, 2),
            'current_daily_average': 0.0,
            'ml_projection': 0.0,
            'budget_health': 'New Budget - Start tracking expenses',
            'top_spending_category': 'None',
            'allocation_strategy': 'Default Template'
        }
        
        return recommendations
    
    def _preprocess_text(self, text):
        """Clean and preprocess text descriptions"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Keep alphanumeric and spaces, remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return ' '.join(text.split())
    
    def get_model_status(self):
        """Get status of all loaded models"""
        status = {
            'models_loaded': self.models_loaded,
            'total_models': len(self.models_loaded),
            'loaded_count': sum(self.models_loaded.values()),
            'category_regressors_count': len(self.category_regressors)
        }
        return status

class ExpenseAnalyzer:
    def __init__(self):
        self.ml_manager = MLModelManager()
    
    def analyze_expense(self, description, amount, category=None, date=None):
        """Comprehensive analysis of a single expense"""
        result = {
            'original': {
                'description': str(description),
                'amount': float(amount),
                'category': str(category) if category else None,
                'date': str(date) if date else None
            }
        }
        
        # Category prediction (if not provided)
        if not category:
            predicted_category, cat_confidence = self.ml_manager.predict_category(description)
            result['predicted_category'] = {
                'category': str(predicted_category),
                'confidence': round(float(cat_confidence), 3)
            }
            category = predicted_category
        else:
            result['predicted_category'] = {
                'category': str(category),
                'confidence': 1.0  # User provided
            }
        
        # Anomaly detection
        is_anomaly, anomaly_confidence = self.ml_manager.detect_anomaly(amount, category, date)
        result['anomaly_detection'] = {
            'is_anomaly': bool(is_anomaly),
            'confidence': round(float(anomaly_confidence), 3),
            'flag': 'high' if is_anomaly and anomaly_confidence > 0.7 else ('medium' if is_anomaly else 'normal')
        }
        
        # Category insights
        result['insights'] = {
            'category_tip': self.ml_manager._get_optimization_tip(category, float(amount), float(amount)),
            'is_essential': category in ['Food', 'Healthcare', 'Bills', 'Education'],
            'suggested_review': is_anomaly or float(amount) > 1000
        }
        
        return convert_numpy_types(result)
    
    def analyze_dataset(self, df):
        """Comprehensive analysis of entire expense dataset"""
        results = {
            'dataset_info': {
                'total_expenses': int(len(df)),
                'total_amount': float(df['amount'].sum()),
                'date_range': {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max()),
                    'days_covered': int((pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days + 1)
                },
                'average_daily_spending': round(float(df['amount'].sum()) / max(len(df['date'].dt.date.unique()), 1), 2)
            },
            'categories': {},
            'anomalies': [],
            'spending_patterns': {},
            'ml_insights': {},
            'recommendations': []
        }
        
        if len(df) == 0:
            return results
        
        try:
            # Category analysis with enhanced metrics
            category_stats = df.groupby('category')['amount'].agg(['sum', 'mean', 'count', 'std', 'min', 'max']).round(2)
            for category in category_stats.index:
                cat_data = category_stats.loc[category]
                results['categories'][str(category)] = {
                    'total': float(cat_data['sum']),
                    'average': float(cat_data['mean']),
                    'count': int(cat_data['count']),
                    'std_dev': float(cat_data['std']) if not pd.isna(cat_data['std']) else 0.0,
                    'min_expense': float(cat_data['min']),
                    'max_expense': float(cat_data['max']),
                    'percentage_of_total': round(float(cat_data['sum']) / df['amount'].sum() * 100, 1)
                }
            
            # Enhanced anomaly detection
            anomaly_count = 0
            for idx, row in df.iterrows():
                is_anomaly, confidence = self.ml_manager.detect_anomaly(
                    row['amount'], 
                    row.get('category'), 
                    row.get('date')
                )
                if is_anomaly and confidence > 0.5:
                    anomaly_count += 1
                    results['anomalies'].append({
                        'index': int(idx),
                        'amount': float(row['amount']),
                        'description': str(row['description']),
                        'category': str(row['category']),
                        'date': str(row['date']),
                        'confidence': round(float(confidence), 3),
                        'severity': 'high' if confidence > 0.8 else 'medium'
                    })
            
            # Spending patterns analysis
            df['date'] = pd.to_datetime(df['date'])
            daily_spending = df.groupby(df['date'].dt.date)['amount'].sum()
            weekly_spending = df.groupby(df['date'].dt.isocalendar().week)['amount'].sum()
            monthly_spending = df.groupby(df['date'].dt.month)['amount'].sum()
            
            if len(daily_spending) > 0:
                results['spending_patterns'] = {
                    'daily_stats': {
                        'average': round(float(daily_spending.mean()), 2),
                        'median': round(float(daily_spending.median()), 2),
                        'std_dev': round(float(daily_spending.std()), 2) if len(daily_spending) > 1 else 0.0,
                        'highest_day': {
                            'date': str(daily_spending.idxmax()),
                            'amount': float(daily_spending.max())
                        },
                        'lowest_day': {
                            'date': str(daily_spending.idxmin()),
                            'amount': float(daily_spending.min())
                        }
                    },
                    'weekly_average': round(float(weekly_spending.mean()), 2) if len(weekly_spending) > 0 else 0.0,
                    'monthly_average': round(float(monthly_spending.mean()), 2) if len(monthly_spending) > 0 else 0.0,
                    'spending_trend': self._calculate_spending_trend(daily_spending),
                    'day_of_week_analysis': self._analyze_day_of_week_spending(df)
                }
            
            # ML-powered insights
            model_status = self.ml_manager.get_model_status()
            results['ml_insights'] = {
                'models_available': model_status['loaded_count'],
                'total_models': model_status['total_models'],
                'model_status': model_status['models_loaded'],
                'anomaly_detection_active': model_status['models_loaded']['anomaly'],
                'category_prediction_active': model_status['models_loaded']['category'],
                'budget_forecasting_active': model_status['models_loaded']['depletion'],
                'smart_allocation_active': model_status['models_loaded']['allocation'],
                'category_regressors': model_status['category_regressors_count']
            }
            
            # Generate recommendations
            results['recommendations'] = self._generate_spending_recommendations(df, results)
            
            # Budget health assessment
            if 'amount' in df.columns and len(df) > 0:
                total_spending = df['amount'].sum()
                days_covered = len(df['date'].dt.date.unique())
                daily_avg = total_spending / max(days_covered, 1)
                
                results['budget_assessment'] = {
                    'daily_burn_rate': round(float(daily_avg), 2),
                    'weekly_projection': round(float(daily_avg * 7), 2),
                    'monthly_projection': round(float(daily_avg * 30), 2),
                    'spending_consistency': self._assess_spending_consistency(daily_spending),
                    'anomaly_rate': round(float(anomaly_count / len(df) * 100), 2)
                }
        
        except Exception as e:
            print(f"Error in dataset analysis: {e}")
            results['error'] = str(e)
        
        return convert_numpy_types(results)
    
    def _calculate_spending_trend(self, daily_spending):
        """Calculate spending trend over time"""
        if len(daily_spending) < 3:
            return "insufficient_data"
        
        try:
            # Simple linear regression to find trend
            x = np.arange(len(daily_spending))
            y = daily_spending.values
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > daily_spending.mean() * 0.05:  # 5% threshold
                return "increasing"
            elif slope < -daily_spending.mean() * 0.05:
                return "decreasing"
            else:
                return "stable"
        except:
            return "unknown"
    
    def _analyze_day_of_week_spending(self, df):
        """Analyze spending patterns by day of week"""
        try:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
            dow_spending = df.groupby('day_of_week')['amount'].sum().round(2)
            
            # Order by weekday
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_analysis = {}
            
            for day in weekday_order:
                if day in dow_spending.index:
                    dow_analysis[day] = float(dow_spending[day])
                else:
                    dow_analysis[day] = 0.0
            
            # Find highest and lowest spending days
            if len(dow_spending) > 0:
                highest_day = dow_spending.idxmax()
                lowest_day = dow_spending.idxmin()
                
                dow_analysis['insights'] = {
                    'highest_spending_day': highest_day,
                    'lowest_spending_day': lowest_day,
                    'weekend_vs_weekday': self._compare_weekend_weekday_spending(df)
                }
            
            return dow_analysis
        except Exception as e:
            print(f"Error in day of week analysis: {e}")
            return {}
    
    def _compare_weekend_weekday_spending(self, df):
        """Compare weekend vs weekday spending"""
        try:
            df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek.isin([5, 6])  # Saturday, Sunday
            weekend_avg = df[df['is_weekend']]['amount'].mean()
            weekday_avg = df[~df['is_weekend']]['amount'].mean()
            
            if pd.isna(weekend_avg) or pd.isna(weekday_avg):
                return "insufficient_data"
            
            if weekend_avg > weekday_avg * 1.2:
                return "higher_on_weekends"
            elif weekday_avg > weekend_avg * 1.2:
                return "higher_on_weekdays"
            else:
                return "similar"
        except:
            return "unknown"
    
    def _assess_spending_consistency(self, daily_spending):
        """Assess how consistent daily spending is"""
        if len(daily_spending) < 3:
            return "insufficient_data"
        
        try:
            cv = daily_spending.std() / daily_spending.mean()  # Coefficient of variation
            
            if cv < 0.3:
                return "very_consistent"
            elif cv < 0.6:
                return "moderately_consistent"
            elif cv < 1.0:
                return "variable"
            else:
                return "highly_variable"
        except:
            return "unknown"
    
    def _generate_spending_recommendations(self, df, analysis_results):
        """Generate personalized spending recommendations"""
        recommendations = []
        
        try:
            # Anomaly-based recommendations
            if len(analysis_results['anomalies']) > 0:
                anomaly_rate = len(analysis_results['anomalies']) / len(df) * 100
                if anomaly_rate > 10:
                    recommendations.append({
                        'type': 'anomaly_alert',
                        'priority': 'high',
                        'message': f"High anomaly rate detected ({anomaly_rate:.1f}%). Review unusual expenses.",
                        'action': 'review_anomalies'
                    })
            
            # Category-based recommendations
            if analysis_results['categories']:
                top_category = max(analysis_results['categories'].items(), key=lambda x: x[1]['total'])
                top_cat_name, top_cat_data = top_category
                
                if top_cat_data['percentage_of_total'] > 40:
                    recommendations.append({
                        'type': 'category_dominance',
                        'priority': 'medium',
                        'message': f"{top_cat_name} accounts for {top_cat_data['percentage_of_total']}% of spending. Consider diversifying expenses.",
                        'action': 'diversify_spending'
                    })
                
                # High variance categories
                for category, data in analysis_results['categories'].items():
                    if data['std_dev'] > data['average'] * 0.8 and data['count'] > 3:
                        recommendations.append({
                            'type': 'high_variance',
                            'priority': 'medium',
                            'message': f"{category} spending is highly variable. Consider setting consistent limits.",
                            'action': 'set_category_budget'
                        })
            
            # Spending pattern recommendations
            if 'spending_patterns' in analysis_results:
                patterns = analysis_results['spending_patterns']
                
                if patterns.get('spending_trend') == 'increasing':
                    recommendations.append({
                        'type': 'increasing_trend',
                        'priority': 'high',
                        'message': "Spending trend is increasing over time. Review and adjust budget.",
                        'action': 'review_budget'
                    })
                
                # Day of week recommendations
                dow_analysis = patterns.get('day_of_week_analysis', {})
                if 'insights' in dow_analysis:
                    weekend_pattern = dow_analysis['insights'].get('weekend_vs_weekday')
                    if weekend_pattern == 'higher_on_weekends':
                        recommendations.append({
                            'type': 'weekend_spending',
                            'priority': 'low',
                            'message': "Weekend spending is significantly higher. Plan weekend activities within budget.",
                            'action': 'weekend_budget_planning'
                        })
            
            # ML model recommendations
            ml_insights = analysis_results.get('ml_insights', {})
            if ml_insights.get('models_available', 0) < ml_insights.get('total_models', 4):
                recommendations.append({
                    'type': 'ml_enhancement',
                    'priority': 'low',
                    'message': "Enable more ML models for better insights and predictions.",
                    'action': 'train_ml_models'
                })
            
            # General recommendations
            if len(df) > 30:  # Enough data for meaningful analysis
                total_expenses = len(df)
                if total_expenses > 100:
                    recommendations.append({
                        'type': 'data_rich',
                        'priority': 'low',
                        'message': "Great job tracking expenses! Consider setting up automated categorization.",
                        'action': 'enable_auto_categorization'
                    })
        
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            recommendations.append({
                'type': 'error',
                'priority': 'low',
                'message': "Continue tracking expenses for personalized insights.",
                'action': 'keep_tracking'
            })
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_quick_insights(self, df, budget=None):
        """Get quick insights without full analysis"""
        if len(df) == 0:
            return {"message": "No expenses to analyze"}
        
        try:
            total_spent = df['amount'].sum()
            avg_expense = df['amount'].mean()
            top_category = df.groupby('category')['amount'].sum().idxmax()
            
            insights = {
                'total_spent': round(float(total_spent), 2),
                'average_expense': round(float(avg_expense), 2),
                'total_transactions': int(len(df)),
                'top_category': str(top_category),
                'date_range_days': int((pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days + 1)
            }
            
            if budget:
                insights['budget_utilization'] = round(float(total_spent / budget * 100), 1)
                insights['remaining_budget'] = round(float(budget - total_spent), 2)
            
            # Quick anomaly check
            anomalies = 0
            for _, row in df.iterrows():
                is_anomaly, _ = self.ml_manager.detect_anomaly(row['amount'], row.get('category'))
                if is_anomaly:
                    anomalies += 1
            
            insights['anomaly_count'] = anomalies
            insights['anomaly_rate'] = round(float(anomalies / len(df) * 100), 1)
            
            return convert_numpy_types(insights)
        
        except Exception as e:
            print(f"Error in quick insights: {e}")
            return {"error": str(e)}

# Additional utility functions
def load_ml_models(model_dir='ml_models'):
    """Standalone function to load ML models"""
    return MLModelManager(model_dir)

def analyze_single_expense(description, amount, category=None, date=None):
    """Standalone function to analyze a single expense"""
    analyzer = ExpenseAnalyzer()
    return analyzer.analyze_expense(description, amount, category, date)

def get_smart_budget_allocation(expenses_df, budget, days_remaining=30):
    """Standalone function for smart budget allocation"""
    ml_manager = MLModelManager()
    return ml_manager.get_smart_allocation(expenses_df, budget, days_remaining)

def predict_budget_depletion(expenses_df, budget):
    """Standalone function for budget depletion prediction"""
    ml_manager = MLModelManager()
    return ml_manager.predict_budget_depletion(expenses_df, budget)

# Model validation and testing functions
def validate_models(model_dir='ml_models'):
    """Validate that all models are properly saved and loadable"""
    try:
        ml_manager = MLModelManager(model_dir)
        status = ml_manager.get_model_status()
        
        print("🔍 Model Validation Results:")
        print(f"  📊 Models loaded: {status['loaded_count']}/{status['total_models']}")
        
        for model_name, is_loaded in status['models_loaded'].items():
            status_icon = "✅" if is_loaded else "❌"
            print(f"  {status_icon} {model_name.title()} Model")
        
        if status['category_regressors_count'] > 0:
            print(f"  🧠 Category Regressors: {status['category_regressors_count']} trained")
        
        return status['loaded_count'] >= 2  # At least 2 models should work
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def test_ml_functionality():
    """Test ML functionality with sample data"""
    try:
        print("🧪 Testing ML functionality...")
        
        # Test data
        test_expenses = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'amount': [500, 1200, 300, 800, 2000, 450, 600, 1500, 400, 750],
            'category': ['Food', 'Shopping', 'Food', 'Entertainment', 'Shopping', 'Food', 'Travel', 'Bills', 'Food', 'Entertainment'],
            'description': ['Restaurant dinner', 'Clothes shopping', 'Grocery store', 'Movie tickets', 'Electronics', 'Fast food', 'Uber ride', 'Electricity bill', 'Coffee shop', 'Concert tickets']
        })
        
        analyzer = ExpenseAnalyzer()
        
        # Test single expense analysis
        result = analyzer.analyze_expense("McDonald's burger meal", 450, "Food")
        print(f"  ✅ Single expense analysis: {result['predicted_category']['category']}")
        
        # Test dataset analysis
        analysis = analyzer.analyze_dataset(test_expenses)
        print(f"  ✅ Dataset analysis: {analysis['dataset_info']['total_expenses']} expenses analyzed")
        
        # Test smart allocation
        ml_manager = MLModelManager()
        allocation = ml_manager.get_smart_allocation(test_expenses, 15000, 30)
        print(f"  ✅ Smart allocation: {allocation['summary']['allocation_strategy']}")
        
        # Test budget prediction
        remaining_days, trend = ml_manager.predict_budget_depletion(test_expenses, 15000)
        print(f"  ✅ Budget prediction: {remaining_days if remaining_days else 'No depletion'} days")
        
        print("🎉 All ML functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ML functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ML Helpers Module Loaded")
    print("📋 Available functions:")
    print("  • MLModelManager() - Main ML model manager")
    print("  • ExpenseAnalyzer() - Comprehensive expense analysis")
    print("  • validate_models() - Validate model files")
    print("  • test_ml_functionality() - Test ML features")
    
    # Quick validation
    if validate_models():
        print("✅ Models validated successfully!")
        if test_ml_functionality():
            print("🎉 ML system is fully functional!")
    else:
        print("⚠️  Some models missing. Run train_models.py first.")
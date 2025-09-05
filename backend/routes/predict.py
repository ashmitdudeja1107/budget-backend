# backend/routes/predict.py
from fastapi import APIRouter, HTTPException, Depends,UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from io import StringIO
# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.ml_helpers import MLModelManager

# Create APIRouter (equivalent to Flask Blueprint)
prediction_router = APIRouter(prefix="/predict", tags=["predictions"])

# Initialize ML Manager
ml_manager = MLModelManager()

# Pydantic models for request/response validation
class CategoryPredictionRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Expense description")

class CategoryPredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    description: str
    success: bool = True

class BatchCategoryRequest(BaseModel):
    descriptions: List[str] = Field(..., min_items=1, description="List of expense descriptions")

class BatchCategoryResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_processed: int
    success: bool = True

class AnomalyDetectionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Expense amount")
    category: Optional[str] = None

class AnomalyDetectionResponse(BaseModel):
    amount: float
    is_anomaly: bool
    confidence: float
    risk_level: str
    category: Optional[str]
    recommendation: str
    success: bool = True

class BudgetDepletionRequest(BaseModel):
    expenses: List[Dict[str, Any]] = Field(..., min_items=1)
    budget: float = Field(default=10000, gt=0)

class BudgetDepletionResponse(BaseModel):
    budget: float
    total_spent: float
    remaining_budget: float
    predicted_depletion_days: Optional[int]
    daily_burn_rate: float
    trend_data: List[Dict[str, Any]]
    status: str
    success: bool = True

class SpendingForecastRequest(BaseModel):
    expenses: List[Dict[str, Any]] = Field(..., min_items=1)
    days: int = Field(default=30, gt=0, le=365)

class SpendingForecastResponse(BaseModel):
    forecast_days: int
    daily_forecast: List[Dict[str, Any]]
    category_forecast: Dict[str, Any]
    total_predicted: float
    daily_average: float
    success: bool = True

class SpendingPatternsRequest(BaseModel):
    expenses: List[Dict[str, Any]] = Field(..., min_items=1)

class OptimizeAllocationsRequest(BaseModel):
    expenses: List[Dict[str, Any]] = Field(..., min_items=1)
    budget: float = Field(default=10000, gt=0)
    target_days: int = Field(default=30, gt=0)

@prediction_router.post("/category", response_model=CategoryPredictionResponse)
async def predict_category(request: CategoryPredictionRequest):
    """Predict category for expense description"""
    try:
        # Predict category
        category, confidence = ml_manager.predict_category(request.description)
        
        return CategoryPredictionResponse(
            predicted_category=category,
            confidence=round(confidence, 3),
            description=request.description
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@prediction_router.post("/category/batch", response_model=BatchCategoryResponse)
async def predict_category_batch(request: BatchCategoryRequest):
    """Predict categories for multiple descriptions"""
    try:
        results = []
        for desc in request.descriptions:
            category, confidence = ml_manager.predict_category(str(desc))
            results.append({
                'description': desc,
                'predicted_category': category,
                'confidence': round(confidence, 3)
            })
        
        return BatchCategoryResponse(
            predictions=results,
            total_processed=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@prediction_router.post("/anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomaly(request: AnomalyDetectionRequest):
    """Detect if expense amount is anomalous"""
    try:
        # Detect anomaly
        is_anomaly, confidence = ml_manager.detect_anomaly(float(request.amount), request.category)
        
        # Determine risk level
        risk_level = 'low'
        if is_anomaly:
            if confidence > 0.8:
                risk_level = 'high'
            elif confidence > 0.5:
                risk_level = 'medium'
        
        return AnomalyDetectionResponse(
            amount=float(request.amount),
            is_anomaly=is_anomaly,
            confidence=round(confidence, 3),
            risk_level=risk_level,
            category=request.category,
            recommendation='Review this expense' if is_anomaly else 'Normal expense'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@prediction_router.post("/budget/depletion", response_model=BudgetDepletionResponse)
async def predict_budget_depletion(request: BudgetDepletionRequest):
    """Predict when budget will be depleted"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.expenses)
        
        # Ensure required columns
        required_cols = ['date', 'amount']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f'Missing required column: {col}')
        
        # Predict depletion
        remaining_days, trend_data = ml_manager.predict_budget_depletion(df, float(request.budget))
        
        # Calculate additional metrics
        total_spent = df['amount'].sum()
        remaining_budget = request.budget - total_spent
        daily_burn_rate = total_spent / len(df) if len(df) > 0 else 0
        
        return BudgetDepletionResponse(
            budget=float(request.budget),
            total_spent=float(total_spent),
            remaining_budget=float(remaining_budget),
            predicted_depletion_days=remaining_days,
            daily_burn_rate=float(daily_burn_rate),
            trend_data=trend_data,
            status='healthy' if remaining_budget > request.budget * 0.3 else 'warning' if remaining_budget > 0 else 'critical'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@prediction_router.post("/spending/forecast", response_model=SpendingForecastResponse)
async def forecast_spending(request: SpendingForecastRequest):
    """Forecast spending for next N days"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.expenses)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate daily averages by category
        df['day_of_week'] = df['date'].dt.day_name()
        
        # Overall daily average
        date_range = (df['date'].max() - df['date'].min()).days
        daily_avg = df['amount'].sum() / max(date_range, 1)
        
        # Category-wise daily averages
        category_daily_avg = df.groupby('category')['amount'].sum() / max(date_range, 1)
        
        # Generate forecast
        forecast_data = []
        total_forecast = 0
        
        for day in range(1, request.days + 1):
            # Simple linear projection (can be enhanced with seasonal patterns)
            daily_forecast = daily_avg
            
            # Add some randomness based on historical variance
            variance = df['amount'].std() * 0.1
            daily_forecast += np.random.normal(0, variance)
            daily_forecast = max(0, daily_forecast)  # Ensure non-negative
            
            total_forecast += daily_forecast
            
            forecast_data.append({
                'day': day,
                'predicted_amount': round(daily_forecast, 2),
                'cumulative_amount': round(total_forecast, 2)
            })
        
        # Category-wise forecast
        category_forecast = {}
        for category, avg_amount in category_daily_avg.items():
            category_forecast[category] = {
                'daily_average': round(avg_amount, 2),
                'total_forecast': round(avg_amount * request.days, 2)
            }
        
        return SpendingForecastResponse(
            forecast_days=request.days,
            daily_forecast=forecast_data,
            category_forecast=category_forecast,
            total_predicted=round(total_forecast, 2),
            daily_average=round(daily_avg, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@prediction_router.post("/insights/spending_patterns")
async def analyze_spending_patterns(filter_data: dict):
    """Analyze spending patterns from filter route response data and provide insights"""
    try:
        # Extract expenses data from filter route response
        if 'expenses' not in filter_data:
            raise HTTPException(status_code=400, detail="Invalid filter data format - 'expenses' key not found")
        
        expenses_list = filter_data['expenses']
        
        if not expenses_list:
            raise HTTPException(status_code=400, detail="No expense data found in filter response")
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(expenses_list)
        
        # Validate required columns
        required_columns = ['date', 'amount', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Filter data must contain columns: {', '.join(required_columns)}. Missing: {', '.join(missing_columns)}"
            )
        
        # Clean and process data
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['date', 'amount', 'category'])
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data found in filter response")
        
        # Add derived columns
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
        df['hour'] = df['date'].dt.hour
        
        insights = {}
        
        # Day of week patterns
        day_spending = df.groupby('day_of_week')['amount'].agg(['sum', 'mean', 'count'])
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_patterns = []
        
        for day in day_order:
            if day in day_spending.index:
                day_patterns.append({
                    'day': day,
                    'total': float(day_spending.loc[day, 'sum']),
                    'average': float(day_spending.loc[day, 'mean']),
                    'count': int(day_spending.loc[day, 'count'])
                })
        
        insights['day_of_week_patterns'] = day_patterns
        
        # Category patterns with integrated recommendations
        category_patterns = []
        category_stats = df.groupby('category')['amount'].agg(['sum', 'mean', 'count', 'std'])
        
        # Define category-specific thresholds
        category_thresholds = {
            'Bills': 40,      # Bills can be higher
            'Rent': 30,       # Rent can be higher
            'Food': 25,       # Food should be moderate
            'Transport': 15,  # Transport should be lower
            'Entertainment': 10,  # Entertainment should be minimal
            'Shopping': 15,   # Shopping should be controlled
            'Healthcare': 20, # Healthcare can vary
            'Travel': 10,     # Travel should be occasional
            'Other': 10,      # Other should be minimal
            'Snacks': 5,      # Snacks should be minimal
            'Utilities': 15   # Utilities should be controlled
        }
        
        for category in category_stats.index:
            percentage = float((category_stats.loc[category, 'sum'] / df['amount'].sum()) * 100)
            threshold = category_thresholds.get(category, 20)  # Default to 20% if not specified
            
            # Generate recommendation for this category
            if percentage <= threshold:
                recommendation = {
                    'status': 'good',
                    'message': f"Good! Your {category} spending is within budget at {percentage:.1f}% of total expenses",
                    'action': 'maintain'
                }
            elif percentage <= threshold * 1.5:
                recommendation = {
                    'status': 'warning',
                    'message': f"{category} spending at {percentage:.1f}% is slightly above ideal - consider minor adjustments",
                    'action': 'monitor'
                }
            else:
                recommendation = {
                    'status': 'alert',
                    'message': f"'{category}' represents {percentage:.1f}% of your spending - consider setting stricter limits",
                    'action': 'reduce'
                }
            
            category_patterns.append({
                'category': category,
                'total': float(category_stats.loc[category, 'sum']),
                'average': float(category_stats.loc[category, 'mean']),
                'count': int(category_stats.loc[category, 'count']),
                'std_dev': float(category_stats.loc[category, 'std']) if not pd.isna(category_stats.loc[category, 'std']) else 0,
                'percentage_of_total': percentage,
                'threshold': threshold,
                'recommendation': recommendation
            })
        
        insights['category_patterns'] = sorted(category_patterns, key=lambda x: x['total'], reverse=True)
        
        # Time-based patterns
        if 'hour' in df.columns and not df['hour'].isna().all():
            hourly_spending = df.groupby('hour')['amount'].agg(['sum', 'count'])
            time_patterns = []
            
            for hour in sorted(hourly_spending.index):
                time_patterns.append({
                    'hour': int(hour),
                    'total': float(hourly_spending.loc[hour, 'sum']),
                    'count': int(hourly_spending.loc[hour, 'count'])
                })
            
            insights['time_patterns'] = time_patterns
        
        # Spending trends
        daily_totals = df.groupby(df['date'].dt.date)['amount'].sum()
        
        # Calculate trend direction
        if len(daily_totals) > 1:
            trend_slope = np.polyfit(range(len(daily_totals)), daily_totals.values, 1)[0]
            trend_direction = 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        insights['trends'] = {
            'direction': trend_direction,
            'daily_variance': float(daily_totals.std()) if len(daily_totals) > 1 else 0,
            'highest_spending_day': {
                'date': str(daily_totals.idxmax()) if len(daily_totals) > 0 else None,
                'amount': float(daily_totals.max()) if len(daily_totals) > 0 else 0
            },
            'lowest_spending_day': {
                'date': str(daily_totals.idxmin()) if len(daily_totals) > 0 else None,
                'amount': float(daily_totals.min()) if len(daily_totals) > 0 else 0
            }
        }
        
        # Overall summary recommendations
        overall_recommendations = []
        high_spending_categories = [cat for cat in category_patterns if cat['recommendation']['status'] == 'alert']
        warning_categories = [cat for cat in category_patterns if cat['recommendation']['status'] == 'warning']
        good_categories = [cat for cat in category_patterns if cat['recommendation']['status'] == 'good']
        
        if high_spending_categories:
            overall_recommendations.append(f"Priority: Review spending on {', '.join([cat['category'] for cat in high_spending_categories])} - these categories are consuming excessive budget")
        
        if warning_categories:
            overall_recommendations.append(f"Monitor: Keep an eye on {', '.join([cat['category'] for cat in warning_categories])} - slightly above ideal thresholds")
        
        if good_categories:
            overall_recommendations.append(f"Well managed: {', '.join([cat['category'] for cat in good_categories])} are within healthy spending limits")
        
        insights['overall_recommendations'] = overall_recommendations
        
        # Use filter response metadata if available
        filter_metadata = {
            'total_records': filter_data.get('total_records', len(df)),
            'total_amount': filter_data.get('total_amount', float(df['amount'].sum())),
            'average_amount': filter_data.get('average_amount', float(df['amount'].mean()))
        }
        
        insights['summary'] = {
            'total_analyzed': len(df),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'total_amount': float(df['amount'].sum()),
            'categories_analyzed': len(category_patterns),
            'high_priority_categories': len(high_spending_categories),
            'warning_categories': len(warning_categories),
            'well_managed_categories': len(good_categories)
        }
        
        return {
            'insights': insights,
            'success': True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing spending patterns from filter data: {str(e)}")
@prediction_router.post("/optimize/allocations")
async def optimize_allocations(request: OptimizeAllocationsRequest):
    """Optimize budget allocations across categories"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.expenses)
        
        # Get smart allocation recommendations
        recommendations = ml_manager.get_smart_allocation(df, float(request.budget), request.target_days)
        
        if 'error' in recommendations:
            raise HTTPException(status_code=400, detail=recommendations['error'])
        
        # Calculate optimization score
        current_spending = df.groupby('category')['amount'].sum().to_dict()
        optimization_score = 0
        
        for category, rec in recommendations.items():
            if category != 'summary' and category in current_spending:
                current = current_spending[category]
                recommended = rec['recommended_budget']
                
                # Score based on how well the recommendation balances spending
                if current > 0:
                    ratio = recommended / current
                    # Optimal ratio is around 0.8-1.2 (slight reduction to increase)
                    if 0.8 <= ratio <= 1.2:
                        optimization_score += 20
                    elif 0.6 <= ratio < 0.8 or 1.2 < ratio <= 1.4:
                        optimization_score += 10
        
        optimization_score = min(100, optimization_score)  # Cap at 100
        
        return {
            'optimized_allocations': recommendations,
            'optimization_score': optimization_score,
            'target_days': request.target_days,
            'budget': float(request.budget),
            'success': True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@prediction_router.get("/health")
async def prediction_health():
    """Health check for prediction services"""
    return {
        'status': 'healthy',
        'services': {
            'category_prediction': ml_manager.category_model is not None,
            'anomaly_detection': ml_manager.anomaly_model is not None,
            'budget_depletion': ml_manager.depletion_model is not None
        },
        'timestamp': datetime.now().isoformat()
    }
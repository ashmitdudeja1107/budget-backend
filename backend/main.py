# backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import uuid
import tempfile
import shutil

# Import our custom modules
import sys
# Fix: Add current directory (where routes folder is) to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ml_helpers import MLModelManager, ExpenseAnalyzer
from utils.data_cleaner import CSVProcessor
from routes.predict import prediction_router
from routes.authroutes import router as auth_router  # Import the auth router
from tasks.budget_alert import BudgetAlertSystem

# Initialize FastAPI app
app = FastAPI(
    title="ML-Powered Expense Tracker API",
    description="Advanced expense tracking with ML predictions and insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize components
ml_manager = MLModelManager()
expense_analyzer = ExpenseAnalyzer()
csv_processor = CSVProcessor()
alert_system = BudgetAlertSystem()

# Global variable to store current dataset (in production, use a database)
current_dataset = None
current_budget = 10000  # Default budget

# Include routers
app.include_router(prediction_router, prefix="/api")
app.include_router(auth_router, prefix="/api")  # Add auth router

# Pydantic models
class BudgetSetRequest(BaseModel):
    budget: float = Field(..., gt=0, description="Budget amount")

class ExpenseAnalysisRequest(BaseModel):
    description: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)

class AddExpenseRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    amount: float = Field(..., gt=0)
    description: str = Field(..., min_length=1)
    category: Optional[str] = None

class FilterExpensesParams(BaseModel):
    category: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    anomalies_only: bool = False

@app.get("/")
async def home():
    """Home endpoint"""
    return {
        'message': 'ML-Powered Expense Tracker API',
        'version': '1.0.0',
        'features': [
            'CSV Upload & Processing',
            'Category Prediction',
            'Anomaly Detection',
            'Budget Depletion Prediction',
            'Smart Allocation Planning',
            'Google Authentication'
        ]
    }

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For numpy scalars
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return str(obj)
    else:
        return obj

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV file"""
    global current_dataset
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 16MB.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_file.write(contents)
            temp_filepath = temp_file.name
        
        try:
            # Process the CSV
            result = csv_processor.process_uploaded_csv(temp_filepath)
            
            if result['success']:
                current_dataset = result['data']
                
                # Generate initial analysis
                analysis = expense_analyzer.analyze_dataset(current_dataset)
                
                # Convert numpy types to native Python types for JSON serialization
                categories_counts = current_dataset['category'].value_counts()
                categories_dict = {str(k): int(v) for k, v in categories_counts.items()}
                
                response_data = {
                    'success': True,
                    'message': 'CSV uploaded and processed successfully',
                    'data_info': {
                        'records': int(len(current_dataset)),
                        'date_range': {
                            'start': current_dataset['date'].min().strftime('%Y-%m-%d'),
                            'end': current_dataset['date'].max().strftime('%Y-%m-%d')
                        },
                        'total_amount': float(current_dataset['amount'].sum()),
                        'categories': categories_dict
                    },
                    'quality_report': result['quality_report'],
                    'suggestions': result['suggestions'],
                    'analysis': analysis
                }
                
                # Convert all numpy types in the entire response
                return convert_numpy_types(response_data)
                
            else:
                raise HTTPException(status_code=400, detail=result['error'])
                
        finally:
            # Clean up temporary file
            os.unlink(temp_filepath)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get information about current dataset"""
    global current_dataset
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    try:
        info = {
            'total_records': len(current_dataset),
            'date_range': {
                'start': current_dataset['date'].min().strftime('%Y-%m-%d'),
                'end': current_dataset['date'].max().strftime('%Y-%m-%d'),
                'days': (current_dataset['date'].max() - current_dataset['date'].min()).days
            },
            'amount_stats': {
                'total': float(current_dataset['amount'].sum()),
                'average': float(current_dataset['amount'].mean()),
                'median': float(current_dataset['amount'].median()),
                'min': float(current_dataset['amount'].min()),
                'max': float(current_dataset['amount'].max())
            },
            'categories': current_dataset['category'].value_counts().to_dict(),
            'recent_expenses': current_dataset.nlargest(5, 'date')[['date', 'amount', 'description', 'category']].to_dict('records')
        }
        
        # Convert datetime objects to strings for JSON serialization
        for expense in info['recent_expenses']:
            expense['date'] = expense['date'].strftime('%Y-%m-%d')
            expense['amount'] = float(expense['amount'])
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/budget/set")
async def set_budget(request: BudgetSetRequest):
    """Set monthly budget"""
    global current_budget
    
    try:
        current_budget = float(request.budget)
        
        return {
            'success': True,
            'message': f'Budget set to ₹{current_budget:,.2f}',
            'budget': current_budget
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/budget/status")
async def get_budget_status():
    """Get current budget status and predictions"""
    global current_dataset, current_budget
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    try:
        total_spent = float(current_dataset['amount'].sum())
        remaining_budget = float(current_budget - total_spent)
        
        # Budget depletion prediction
        remaining_days, trend_data = ml_manager.predict_budget_depletion(current_dataset, current_budget)
        
        # Calculate spending rate
        date_range = (current_dataset['date'].max() - current_dataset['date'].min()).days
        daily_avg = float(total_spent / max(date_range, 1))
        
        status = {
            'budget': float(current_budget),
            'spent': float(total_spent),
            'remaining': float(remaining_budget),
            'percentage_used': float((total_spent / current_budget) * 100),
            'daily_average': float(daily_avg),
            'predicted_depletion_days': int(remaining_days) if remaining_days is not None else None,
            'trend_data': trend_data,
            'status': 'healthy' if remaining_budget > current_budget * 0.3 else 'warning' if remaining_budget > 0 else 'exceeded'
        }
        
        return convert_numpy_types(status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/allocations/smart")
async def get_smart_allocations(days: int = Query(30, gt=0, description="Days remaining")):
    """Get smart budget allocation recommendations"""
    global current_dataset, current_budget
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    try:
        recommendations = ml_manager.get_smart_allocation(
            current_dataset, 
            current_budget, 
            days
        )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/expenses/analyze")
async def analyze_single_expense(request: ExpenseAnalysisRequest):
    """Analyze a single expense entry"""
    try:
        analysis = expense_analyzer.analyze_expense(request.description, request.amount)
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/expenses/add")
async def add_expense(request: AddExpenseRequest):
    """Add a new expense to the dataset"""
    global current_dataset
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    try:
        print(f"Dataset size before adding: {len(current_dataset)}")  # Debug
        
        # Create new expense entry
        new_expense = {
            'date': pd.to_datetime(request.date),
            'amount': float(request.amount),
            'description': request.description,
            'category': request.category or 'Other'
        }
        
        # If category not provided, predict it
        if new_expense['category'] == 'Other' or not request.category:
            predicted_category, confidence = ml_manager.predict_category(new_expense['description'])
            new_expense['category'] = predicted_category
        
        # Add to dataset - More robust approach
        new_row = pd.DataFrame([new_expense])
        
        # Ensure column consistency
        if not current_dataset.empty:
            # Align columns
            for col in current_dataset.columns:
                if col not in new_row.columns:
                    new_row[col] = None
            for col in new_row.columns:
                if col not in current_dataset.columns:
                    current_dataset[col] = None
        
        current_dataset = pd.concat([current_dataset, new_row], ignore_index=True)
        current_dataset = current_dataset.sort_values('date').reset_index(drop=True)
        
        print(f"Dataset size after adding: {len(current_dataset)}")  # Debug
        
        # Analyze the new expense
        analysis = expense_analyzer.analyze_expense(new_expense['description'], new_expense['amount'])
        
        return {
            'success': True,
            'message': 'Expense added successfully',
            'new_expense': {
                'date': new_expense['date'].strftime('%Y-%m-%d'),
                'amount': new_expense['amount'],
                'description': new_expense['description'],
                'category': new_expense['category']
            },
            'analysis': analysis,
            'dataset_size': len(current_dataset)
        }
        
    except Exception as e:
        print(f"Error adding expense: {str(e)}")  # Debug
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/expenses/filter")
async def filter_expenses(
    category: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    anomalies_only: bool = Query(False)
):
    """Filter expenses by various criteria"""
    global current_dataset
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    try:
        print(f"Current dataset size: {len(current_dataset)}")  # Debug
        print(f"Dataset columns: {current_dataset.columns.tolist()}")  # Debug
        
        filtered_df = current_dataset.copy()
        
        # Apply filters
        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['date'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['date'] <= end_dt]
        
        if min_amount is not None:
            filtered_df = filtered_df[filtered_df['amount'] >= min_amount]
        
        if max_amount is not None:
            filtered_df = filtered_df[filtered_df['amount'] <= max_amount]
        
        if anomalies_only:
            # Detect anomalies for each row
            anomaly_flags = []
            for _, row in filtered_df.iterrows():
                is_anomaly, _ = ml_manager.detect_anomaly(row['amount'])
                anomaly_flags.append(is_anomaly)
            
            filtered_df = filtered_df[anomaly_flags]
        
        print(f"Filtered dataset size: {len(filtered_df)}")  # Debug
        
        # Convert to JSON-serializable format
        result = filtered_df.copy()
        result['date'] = result['date'].dt.strftime('%Y-%m-%d')
        result['amount'] = result['amount'].astype(float)
        
        # Handle NaN values properly
        total_amount = filtered_df['amount'].sum()
        average_amount = filtered_df['amount'].mean()
        
        # Replace NaN with appropriate defaults
        if pd.isna(total_amount):
            total_amount = 0.0
        if pd.isna(average_amount):
            average_amount = 0.0
            
        return {
            'total_records': len(result),
            'expenses': result.to_dict('records'),
            'total_amount': float(total_amount),
            'average_amount': float(average_amount)
        }
        
    except Exception as e:
        print(f"Error filtering expenses: {str(e)}")  # Debug
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/summary")
async def get_dashboard_summary():
    """Get comprehensive dashboard summary"""
    global current_dataset, current_budget
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    try:
        # Basic stats
        total_spent = float(current_dataset['amount'].sum())
        remaining_budget = current_budget - total_spent
        
        # Category breakdown
        category_stats = current_dataset.groupby('category')['amount'].agg(['sum', 'count']).round(2)
        categories = []
        for cat in category_stats.index:
            categories.append({
                'name': cat,
                'total': float(category_stats.loc[cat, 'sum']),
                'count': int(category_stats.loc[cat, 'count']),
                'percentage': (category_stats.loc[cat, 'sum'] / total_spent) * 100
            })
        
        # Sort categories by total amount (highest first)
        categories.sort(key=lambda x: x['total'], reverse=True)
        
        # Recent expenses (last 7 days)
        recent_date = current_dataset['date'].max() - timedelta(days=7)
        recent_expenses = current_dataset[current_dataset['date'] > recent_date]
        
        # Anomalies - Updated to include all detected anomalies
        anomalies = []
        debug_anomaly_count = 0
        
        for _, row in current_dataset.iterrows():
            is_anomaly, confidence = ml_manager.detect_anomaly(row['amount'])
            if is_anomaly:
                debug_anomaly_count += 1
                anomalies.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'amount': float(row['amount']),
                    'description': row['description'],
                    'category': row['category'],
                    'confidence': round(confidence, 3)
                })
        
        # Sort anomalies by confidence (highest first) and then by amount (highest first)
        anomalies.sort(key=lambda x: (x['confidence'], x['amount']), reverse=True)
        
        # Budget prediction
        remaining_days, trend_data = ml_manager.predict_budget_depletion(current_dataset, current_budget)
        
        summary = {
            'budget_overview': {
                'total_budget': current_budget,
                'spent': total_spent,
                'remaining': remaining_budget,
                'percentage_used': (total_spent / current_budget) * 100,
                'predicted_depletion_days': remaining_days
            },
            'category_breakdown': categories,
            'recent_activity': {
                'last_7_days_count': len(recent_expenses),
                'last_7_days_total': float(recent_expenses['amount'].sum()),
                'daily_average': float(recent_expenses['amount'].sum() / 7) if len(recent_expenses) > 0 else 0
            },
            'anomalies': {
                'count': len(anomalies),
                'items': anomalies[:10],  # Show top 10 anomalies
                'high_confidence_count': len([a for a in anomalies if a['confidence'] > 0.1]),
                'debug_total_detected': debug_anomaly_count  # For debugging - remove in production
            },
            'trends': {
                'highest_category': categories[0]['name'] if categories else 'N/A',
                'average_expense': float(current_dataset['amount'].mean()),
                'total_transactions': len(current_dataset)
            }
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/csv")
async def export_csv():
    """Export current dataset as CSV"""
    global current_dataset
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset to export")
    
    try:
        # Create export filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'expenses_export_{timestamp}.csv'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Export data
        export_df = current_dataset.copy()
        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
        export_df.to_csv(filepath, index=False)
        
        return FileResponse(
            path=filepath, 
            filename=filename,
            media_type='text/csv'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={'error': 'File too large. Maximum size is 16MB.'}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={'error': 'Internal server error'}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }

if __name__ == '__main__':
    import uvicorn
    
    print("Starting ML-Powered Expense Tracker API with FastAPI...")
    print("Available endpoints:")
    print("  POST /api/upload - Upload CSV file")
    print("  GET  /api/dataset/info - Get dataset information")
    print("  POST /api/budget/set - Set budget")
    print("  GET  /api/budget/status - Get budget status")
    print("  GET  /api/allocations/smart - Get smart allocations")
    print("  POST /api/expenses/analyze - Analyze single expense")
    print("  POST /api/expenses/add - Add new expense")
    print("  GET  /api/expenses/filter - Filter expenses")
    print("  GET  /api/dashboard/summary - Get dashboard summary")
    print("  GET  /api/export/csv - Export data as CSV")
    print("  POST /api/auth/google-signup - Google signup")
    print("  POST /api/auth/google-login - Google login")
    print("  GET  /api/auth/health - Auth health check")
    print("  GET  /health - Health check")
    print("  GET  /docs - Interactive API documentation")
    print("  GET  /redoc - Alternative API documentation")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
# database/models.py
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    expenses = relationship("Expense", back_populates="user", cascade="all, delete-orphan")
    budgets = relationship("Budget", back_populates="user", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(255), nullable=False)
    filename = Column(String(255))
    upload_date = Column(DateTime, default=datetime.utcnow)
    total_records = Column(Integer, default=0)
    total_amount = Column(Float, default=0.0)
    date_range_start = Column(Date)
    date_range_end = Column(Date)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="datasets")

class Expense(Base):
    __tablename__ = "expenses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    date = Column(Date, nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    is_anomaly = Column(Boolean, default=False)
    anomaly_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="expenses")

class Budget(Base):
    __tablename__ = "budgets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float, nullable=False)
    period = Column(String(20), default="monthly")  # monthly, weekly, yearly
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="budgets")

class CategoryMapping(Base):
    __tablename__ = "category_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    description_pattern = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    confidence = Column(Float, default=1.0)
    user_trained = Column(Boolean, default=False)  # If user manually corrected
    created_at = Column(DateTime, default=datetime.utcnow)

# database/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://username:password@localhost:5432/expense_tracker"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to create all tables
def create_tables():
    """Create all database tables"""
    from .models import Base  # Import here to avoid circular imports
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

# Function to drop all tables (for development)
def drop_tables():
    """Drop all database tables"""
    from .models import Base
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped successfully!")

# Initialize database on import (optional)
def init_database():
    """Initialize database with tables"""
    try:
        create_tables()
        return True
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        return False

# database/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc, asc
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import pandas as pd
from . import models

class UserCRUD:
    @staticmethod
    def get_or_create_user(db: Session, user_id: str = None) -> models.User:
        """Create a new user for each session (no cookie persistence)"""
        # Always create new user with random UUID
        user = models.User()
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

class ExpenseCRUD:
    @staticmethod
    def create_expense(db: Session, user_id: int, expense_data: dict) -> models.Expense:
        """Create a new expense"""
        expense = models.Expense(
            user_id=user_id,
            **expense_data
        )
        db.add(expense)
        db.commit()
        db.refresh(expense)
        return expense
    
    @staticmethod
    def get_user_expenses(
        db: Session, 
        user_id: int, 
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict] = None
    ) -> List[models.Expense]:
        """Get expenses for a user with optional filters"""
        query = db.query(models.Expense).filter(models.Expense.user_id == user_id)
        
        if filters:
            if filters.get('category'):
                query = query.filter(models.Expense.category == filters['category'])
            if filters.get('start_date'):
                query = query.filter(models.Expense.date >= filters['start_date'])
            if filters.get('end_date'):
                query = query.filter(models.Expense.date <= filters['end_date'])
            if filters.get('min_amount'):
                query = query.filter(models.Expense.amount >= filters['min_amount'])
            if filters.get('max_amount'):
                query = query.filter(models.Expense.amount <= filters['max_amount'])
            if filters.get('anomalies_only'):
                query = query.filter(models.Expense.is_anomaly == True)
        
        query = query.order_by(desc(models.Expense.date), desc(models.Expense.id))
        
        if limit:
            query = query.limit(limit)
        
        return query.offset(offset).all()
    
    @staticmethod
    def bulk_create_expenses(db: Session, user_id: int, expenses_data: List[dict], dataset_id: Optional[int] = None) -> List[models.Expense]:
        """Bulk create expenses from CSV upload"""
        expenses = []
        for expense_data in expenses_data:
            expense_data['user_id'] = user_id
            if dataset_id:
                expense_data['dataset_id'] = dataset_id
            expenses.append(models.Expense(**expense_data))
        
        db.add_all(expenses)
        db.commit()
        return expenses
    
    @staticmethod
    def get_expense_stats(db: Session, user_id: int, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict:
        """Get expense statistics for a user"""
        query = db.query(models.Expense).filter(models.Expense.user_id == user_id)
        
        if start_date:
            query = query.filter(models.Expense.date >= start_date)
        if end_date:
            query = query.filter(models.Expense.date <= end_date)
        
        expenses = query.all()
        
        if not expenses:
            return {
                'total_records': 0,
                'total_amount': 0.0,
                'average_amount': 0.0,
                'categories': {},
                'date_range': None
            }
        
        amounts = [e.amount for e in expenses]
        categories = {}
        
        for expense in expenses:
            if expense.category in categories:
                categories[expense.category]['count'] += 1
                categories[expense.category]['total'] += expense.amount
            else:
                categories[expense.category] = {'count': 1, 'total': expense.amount}
        
        return {
            'total_records': len(expenses),
            'total_amount': sum(amounts),
            'average_amount': sum(amounts) / len(amounts),
            'min_amount': min(amounts),
            'max_amount': max(amounts),
            'categories': categories,
            'date_range': {
                'start': min(e.date for e in expenses),
                'end': max(e.date for e in expenses)
            }
        }

class BudgetCRUD:
    @staticmethod
    def get_active_budget(db: Session, user_id: int) -> Optional[models.Budget]:
        """Get the active budget for a user"""
        today = date.today()
        return db.query(models.Budget).filter(
            and_(
                models.Budget.user_id == user_id,
                models.Budget.is_active == True,
                models.Budget.start_date <= today,
                models.Budget.end_date >= today
            )
        ).first()
    
    @staticmethod
    def create_or_update_budget(db: Session, user_id: int, amount: float, period: str = "monthly") -> models.Budget:
        """Create or update budget for a user"""
        # Deactivate existing budgets
        db.query(models.Budget).filter(
            and_(
                models.Budget.user_id == user_id,
                models.Budget.is_active == True
            )
        ).update({'is_active': False})
        
        # Calculate date range based on period
        today = date.today()
        if period == "monthly":
            start_date = today.replace(day=1)
            if today.month == 12:
                end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
        elif period == "weekly":
            days_since_monday = today.weekday()
            start_date = today - timedelta(days=days_since_monday)
            end_date = start_date + timedelta(days=6)
        else:  # yearly
            start_date = date(today.year, 1, 1)
            end_date = date(today.year, 12, 31)
        
        budget = models.Budget(
            user_id=user_id,
            amount=amount,
            period=period,
            start_date=start_date,
            end_date=end_date,
            is_active=True
        )
        
        db.add(budget)
        db.commit()
        db.refresh(budget)
        return budget

class DatasetCRUD:
    @staticmethod
    def create_dataset(db: Session, user_id: int, name: str, filename: str, metadata: dict) -> models.Dataset:
        """Create a new dataset record"""
        # Deactivate other datasets
        db.query(models.Dataset).filter(
            and_(
                models.Dataset.user_id == user_id,
                models.Dataset.is_active == True
            )
        ).update({'is_active': False})
        
        dataset = models.Dataset(
            user_id=user_id,
            name=name,
            filename=filename,
            total_records=metadata.get('total_records', 0),
            total_amount=metadata.get('total_amount', 0.0),
            date_range_start=metadata.get('date_range_start'),
            date_range_end=metadata.get('date_range_end'),
            is_active=True
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return dataset
    
    @staticmethod
    def get_active_dataset(db: Session, user_id: int) -> Optional[models.Dataset]:
        """Get the active dataset for a user"""
        return db.query(models.Dataset).filter(
            and_(
                models.Dataset.user_id == user_id,
                models.Dataset.is_active == True
            )
        ).first()
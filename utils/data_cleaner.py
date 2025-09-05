# utils/data_cleaner.py
import pandas as pd
import numpy as np
from datetime import datetime
import re

class ExpenseDataCleaner:
    def __init__(self):
        self.required_columns = ['date', 'amount', 'description']
        self.optional_columns = ['category']
        
    def clean_dataset(self, df):
        """Main cleaning function for expense dataset"""
        print(f"Original dataset shape: {df.shape}")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Step 1: Clean column names
        cleaned_df = self._clean_column_names(cleaned_df)
        
        # Step 2: Validate required columns
        cleaned_df = self._validate_columns(cleaned_df)
        
        # Step 3: Clean date column
        cleaned_df = self._clean_dates(cleaned_df)
        
        # Step 4: Clean amount column
        cleaned_df = self._clean_amounts(cleaned_df)
        
        # Step 5: Clean descriptions
        cleaned_df = self._clean_descriptions(cleaned_df)
        
        # Step 6: Handle categories
        cleaned_df = self._clean_categories(cleaned_df)
        
        # Step 7: Remove duplicates
        cleaned_df = self._remove_duplicates(cleaned_df)
        
        # Step 8: Sort by date
        cleaned_df = cleaned_df.sort_values('date').reset_index(drop=True)
        
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print(f"Removed {len(df) - len(cleaned_df)} rows during cleaning")
        
        return cleaned_df
    
    def _clean_column_names(self, df):
        """Clean and standardize column names"""
        # Convert to lowercase and remove spaces
        df.columns = df.columns.str.lower().str.strip()
        
        # Map common variations to standard names
        column_mapping = {
            'date': ['date', 'transaction_date', 'expense_date', 'dt'],
            'amount': ['amount', 'expense', 'cost', 'price', 'value', 'spend'],
            'description': ['description', 'desc', 'details', 'item', 'expense_desc', 'transaction_desc'],
            'category': ['category', 'cat', 'type', 'expense_type', 'class']
        }
        
        # Rename columns based on mapping
        for standard_name, variations in column_mapping.items():
            for col in df.columns:
                if col in variations:
                    df = df.rename(columns={col: standard_name})
                    break
        
        return df
    
    def _validate_columns(self, df):
        """Validate that required columns exist"""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _clean_dates(self, df):
        """Clean and standardize date column"""
        print("Cleaning dates...")
        
        # Try to convert to datetime
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        except:
            # Try different date formats
            date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
            
            for fmt in date_formats:
                try:
                    df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')
                    break
                except:
                    continue
        
        # Remove rows with invalid dates
        invalid_dates = df['date'].isna()
        if invalid_dates.sum() > 0:
            print(f"Removing {invalid_dates.sum()} rows with invalid dates")
            df = df[~invalid_dates].copy()
        
        # Remove future dates (assuming expenses can't be in the future)
        future_dates = df['date'] > datetime.now()
        if future_dates.sum() > 0:
            print(f"Removing {future_dates.sum()} rows with future dates")
            df = df[~future_dates].copy()
        
        return df
    
    def _clean_amounts(self, df):
        """Clean and validate amount column"""
        print("Cleaning amounts...")
        
        # Handle different amount formats
        if df['amount'].dtype == 'object':
            # Remove currency symbols and commas
            df['amount'] = df['amount'].astype(str).str.replace(r'[₹$€£,]', '', regex=True)
            df['amount'] = df['amount'].str.replace(r'[^\d.-]', '', regex=True)
        
        # Convert to numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows with invalid amounts
        invalid_amounts = df['amount'].isna()
        if invalid_amounts.sum() > 0:
            print(f"Removing {invalid_amounts.sum()} rows with invalid amounts")
            df = df[~invalid_amounts].copy()
        
        # Remove negative amounts (assuming all expenses are positive)
        negative_amounts = df['amount'] <= 0
        if negative_amounts.sum() > 0:
            print(f"Removing {negative_amounts.sum()} rows with negative/zero amounts")
            df = df[~negative_amounts].copy()
        
        # Handle outliers (amounts beyond reasonable range)
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outliers as values beyond 3*IQR from Q1 and Q3
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        extreme_outliers = (df['amount'] < lower_bound) | (df['amount'] > upper_bound)
        if extreme_outliers.sum() > 0:
            print(f"Found {extreme_outliers.sum()} extreme outliers (will be flagged but not removed)")
            df['is_outlier'] = extreme_outliers
        else:
            df['is_outlier'] = False
        
        return df
    
    def _clean_descriptions(self, df):
        """Clean and standardize description column"""
        print("Cleaning descriptions...")
        
        # Fill missing descriptions
        df['description'] = df['description'].fillna('Unknown Expense')
        
        # Convert to string and clean
        df['description'] = df['description'].astype(str)
        
        # Remove extra whitespace
        df['description'] = df['description'].str.strip()
        
        # Remove empty descriptions
        empty_desc = (df['description'] == '') | (df['description'] == 'nan')
        df.loc[empty_desc, 'description'] = 'Unknown Expense'
        
        # Standardize common variations
        standardizations = {
            r'\bmcdonalds?\b': 'McDonald\'s',
            r'\buber\b': 'Uber',
            r'\bamazon\b': 'Amazon',
            r'\bnetflix\b': 'Netflix',
            r'\bstarbucks?\b': 'Starbucks',
            r'\bgoogle\b': 'Google',
            r'\bapple\b': 'Apple',
            r'\bmicrosoft\b': 'Microsoft'
        }
        
        for pattern, replacement in standardizations.items():
            df['description'] = df['description'].str.replace(pattern, replacement, case=False, regex=True)
        
        return df
    
    def _clean_categories(self, df):
        """Clean and standardize category column"""
        print("Cleaning categories...")
        
        # If category column doesn't exist, create it
        if 'category' not in df.columns:
            df['category'] = 'Other'
        
        # Fill missing categories
        df['category'] = df['category'].fillna('Other')
        
        # Convert to string and clean
        df['category'] = df['category'].astype(str).str.strip().str.title()
        
        # Standardize category names
        category_mapping = {
            'Food & Dining': 'Food',
            'Food & Drink': 'Food',
            'Restaurants': 'Food',
            'Groceries': 'Food',
            'Dining': 'Food',
            
            'Transportation': 'Travel',
            'Gas & Fuel': 'Travel',
            'Public Transportation': 'Travel',
            'Taxi': 'Travel',
            'Flight': 'Travel',
            
            'Entertainment & Recreation': 'Entertainment',
            'Movies': 'Entertainment',
            'Music': 'Entertainment',
            'Games': 'Entertainment',
            
            'Shopping & Retail': 'Shopping',
            'Clothing': 'Shopping',
            'Electronics': 'Shopping',
            'Online Shopping': 'Shopping',
            
            'Medical': 'Healthcare',
            'Health': 'Healthcare',
            'Doctor': 'Healthcare',
            'Pharmacy': 'Healthcare',
            
            'Utilities': 'Bills',
            'Phone': 'Bills',
            'Internet': 'Bills',
            'Electricity': 'Bills',
            
            'Learning': 'Education',
            'Books': 'Education',
            'Course': 'Education',
            'Training': 'Education'
        }
        
        # Apply mapping
        for old_name, new_name in category_mapping.items():
            df.loc[df['category'].str.contains(old_name, case=False, na=False), 'category'] = new_name
        
        # Ensure only valid categories
        valid_categories = ['Food', 'Travel', 'Entertainment', 'Shopping', 'Healthcare', 'Bills', 'Education', 'Other']
        invalid_categories = ~df['category'].isin(valid_categories)
        df.loc[invalid_categories, 'category'] = 'Other'
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate transactions"""
        print("Removing duplicates...")
        
        initial_count = len(df)
        
        # Define duplicates as same date, amount, and description
        df = df.drop_duplicates(subset=['date', 'amount', 'description'], keep='first')
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate transactions")
        
        return df
    
    def validate_data_quality(self, df):
        """Generate data quality report"""
        report = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'amount_stats': {
                'total': df['amount'].sum(),
                'average': df['amount'].mean(),
                'median': df['amount'].median(),
                'min': df['amount'].min(),
                'max': df['amount'].max(),
                'std': df['amount'].std()
            },
            'categories': df['category'].value_counts().to_dict(),
            'outliers': df['is_outlier'].sum() if 'is_outlier' in df.columns else 0,
            'data_issues': {
                'missing_descriptions': (df['description'] == 'Unknown Expense').sum(),
                'uncategorized': (df['category'] == 'Other').sum()
            }
        }
        
        return report
    
    def get_cleaning_suggestions(self, df):
        """Provide suggestions for data improvement"""
        suggestions = []
        
        # Check for high number of "Other" categories
        other_ratio = (df['category'] == 'Other').sum() / len(df)
        if other_ratio > 0.3:
            suggestions.append(f"High percentage ({other_ratio:.1%}) of expenses categorized as 'Other'. Consider training a category prediction model.")
        
        # Check for missing descriptions
        unknown_desc_ratio = (df['description'] == 'Unknown Expense').sum() / len(df)
        if unknown_desc_ratio > 0.1:
            suggestions.append(f"{unknown_desc_ratio:.1%} of expenses have missing descriptions. This may affect category prediction accuracy.")
        
        # Check for outliers
        if 'is_outlier' in df.columns:
            outlier_ratio = df['is_outlier'].sum() / len(df)
            if outlier_ratio > 0.05:
                suggestions.append(f"{outlier_ratio:.1%} of expenses are statistical outliers. Review these for accuracy.")
        
        # Check date range
        date_span = (df['date'].max() - df['date'].min()).days
        if date_span < 30:
            suggestions.append("Dataset covers less than 30 days. More data would improve ML model accuracy.")
        
        # Check category distribution
        category_counts = df['category'].value_counts()
        if len(category_counts) < 3:
            suggestions.append("Limited category diversity. Consider adding more expense categories for better analysis.")
        
        return suggestions

class CSVProcessor:
    def __init__(self):
        self.cleaner = ExpenseDataCleaner()
    
    def process_uploaded_csv(self, file_path):
        """Process uploaded CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            print(f"Original CSV shape: {df.shape}")
            print(f"Original columns: {list(df.columns)}")
            
            # Clean the dataset
            cleaned_df = self.cleaner.clean_dataset(df)
            
            # Generate quality report
            quality_report = self.cleaner.validate_data_quality(cleaned_df)
            
            # Get suggestions
            suggestions = self.cleaner.get_cleaning_suggestions(cleaned_df)
            
            return {
                'success': True,
                'data': cleaned_df,
                'quality_report': quality_report,
                'suggestions': suggestions,
                'original_shape': df.shape,
                'cleaned_shape': cleaned_df.shape
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def export_cleaned_data(self, df, output_path):
        """Export cleaned data to CSV"""
        try:
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
# train_models.py - FIXED VERSION with proper anomaly model training
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import shutil
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def clean_model_directory(model_dir='ml_models'):
    """Clean and recreate model directory to ensure fresh training"""
    if os.path.exists(model_dir):
        print(f"🧹 Cleaning existing model directory: {model_dir}")
        shutil.rmtree(model_dir)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/preprocessors", exist_ok=True)
    print(f"✅ Created fresh model directory: {model_dir}")

def create_sample_training_data():
    """Create comprehensive sample training data with more examples"""
    
    # Expanded training data for better model performance
    food_descriptions = [
        'McDonald\'s burger meal', 'Restaurant dinner with friends', 'Grocery shopping vegetables',
        'Coffee shop latte', 'Pizza delivery order', 'Breakfast at cafe', 'Lunch takeaway',
        'Supermarket weekly shopping', 'Food court meal', 'Ice cream parlor',
        'Chinese restaurant dinner', 'Bakery fresh bread', 'Fast food combo meal',
        'Organic grocery store', 'Street food vendor', 'Starbucks coffee order',
        'KFC chicken bucket', 'Subway sandwich meal', 'Indian restaurant curry',
        'Thai food delivery', 'Italian pasta dinner', 'Sushi restaurant meal',
        'Dominos pizza order', 'Breakfast cereal purchase', 'Fresh fruit market',
        'Meat shop chicken', 'Dairy products milk', 'Snacks and chips',
        'Energy drinks purchase', 'Bottled water pack', 'Lunch box meal',
        'Cafeteria food payment', 'Room service hotel', 'Vending machine snack',
        'Food truck tacos', 'Buffet dinner payment', 'Cake shop dessert',
        'Juice bar smoothie', 'Tea shop chai', 'Restaurant bill tip'
    ]
    
    travel_descriptions = [
        'Uber ride to airport', 'Flight booking Mumbai', 'Train ticket booking',
        'Petrol pump fuel', 'Taxi fare downtown', 'Bus ticket interstate',
        'Hotel booking vacation', 'Car rental weekend', 'Metro card recharge',
        'Parking fee mall', 'Toll road charges', 'Airport shuttle service',
        'Long distance bus', 'Bike sharing rental', 'Fuel station payment',
        'Ola cab booking', 'Railway reservation', 'Flight cancellation fee',
        'Car insurance payment', 'Vehicle maintenance', 'Highway toll tax',
        'Auto rickshaw fare', 'Bike fuel filling', 'Car wash service',
        'Parking meter payment', 'Travel insurance buy', 'Luggage excess fee',
        'Train meal service', 'Airport lounge access', 'Rental car insurance',
        'Bus station snacks', 'Fuel efficiency check', 'Car tire replacement',
        'Vehicle registration', 'Driving license fee', 'Traffic fine payment',
        'GPS navigation app', 'Car accessories buy', 'Vehicle loan EMI',
        'Roadside assistance', 'Car battery replacement', 'Oil change service'
    ]
    
    entertainment_descriptions = [
        'Movie tickets IMAX', 'Netflix subscription', 'Gaming console purchase',
        'Concert tickets weekend', 'Spotify premium monthly', 'Theatre show booking',
        'Amusement park entry', 'Bowling alley games', 'Video game purchase',
        'Music streaming service', 'Cinema popcorn combo', 'Adventure sports',
        'Night club entry fee', 'Comedy show tickets', 'Sports event tickets',
        'Amazon Prime Video', 'YouTube premium sub', 'Disney plus hotstar',
        'PlayStation game buy', 'Xbox live membership', 'Steam game purchase',
        'Cricket match tickets', 'Football game entry', 'Tennis tournament',
        'Water park admission', 'Zoo entry tickets', 'Museum visit fee',
        'Art gallery entrance', 'Book fair tickets', 'Music festival pass',
        'Stand up comedy', 'Magic show tickets', 'Dance performance',
        'Opera house booking', 'Circus show entry', 'Planetarium visit',
        'Escape room game', 'Laser tag session', 'Mini golf game',
        'Arcade game tokens', 'Pool table rental', 'Karaoke room booking'
    ]
    
    healthcare_descriptions = [
        'Doctor consultation fee', 'Medical insurance premium', 'Spa massage therapy',
        'Pharmacy medicine purchase', 'Dental checkup cleaning', 'Eye exam optometrist',
        'Physical therapy session', 'Gym membership monthly', 'Yoga class package',
        'Health supplement vitamins', 'Medical test lab', 'Hospital emergency visit',
        'Prescription medication', 'Mental health counseling', 'Fitness tracker device',
        'Blood test checkup', 'X-ray scan fee', 'MRI scan payment',
        'Vaccination shot cost', 'Health insurance claim', 'Physiotherapy session',
        'Dermatologist visit', 'Orthopedic consultation', 'Cardiac checkup fee',
        'Diabetes test kit', 'Blood pressure monitor', 'Thermometer purchase',
        'First aid kit', 'Surgical mask buy', 'Hand sanitizer bottle',
        'Protein powder jar', 'Multivitamin tablets', 'Calcium supplements',
        'Gym equipment buy', 'Yoga mat purchase', 'Meditation app sub',
        'Health book purchase', 'Nutrition consultation', 'Diet plan fee',
        'Wellness retreat', 'Spa day package', 'Massage therapy'
    ]
    
    bills_descriptions = [
        'Electricity bill payment', 'Internet service monthly', 'Water bill quarterly',
        'Gas connection bill', 'Mobile phone plan', 'Cable TV subscription',
        'Home insurance premium', 'Property tax payment', 'Maintenance charges',
        'Utility bills combined', 'Broadband internet bill', 'Landline phone bill',
        'Home security system', 'Waste management fee', 'HOA monthly dues',
        'Society maintenance', 'Apartment rent payment', 'House loan EMI',
        'Property insurance', 'Home repair service', 'Plumber service call',
        'Electrician repair', 'AC maintenance fee', 'House cleaning service',
        'Garden maintenance', 'Pest control service', 'Water purifier service',
        'Solar panel payment', 'Generator maintenance', 'Inverter battery',
        'House painting cost', 'Roof repair charges', 'Window replacement',
        'Door lock repair', 'Ceiling fan service', 'Geyser maintenance',
        'Washing machine fix', 'Refrigerator repair', 'Microwave service',
        'Television repair', 'Computer maintenance', 'Printer ink refill'
    ]
    
    shopping_descriptions = [
        'Amazon shopping clothes', 'Electronics store laptop', 'Clothing brand outlet',
        'Shoes online purchase', 'Home decor items', 'Kitchen appliances buy',
        'Furniture store sofa', 'Jewelry store purchase', 'Bookstore novel buy',
        'Art supplies creative', 'Garden center plants', 'Sports equipment gear',
        'Beauty products cosmetics', 'Tech gadget accessories', 'Gift items purchase',
        'Flipkart mobile order', 'Fashion accessories', 'Watch brand store',
        'Perfume brand purchase', 'Sunglasses brand buy', 'Handbag designer',
        'Wallet leather goods', 'Belt accessories shop', 'Tie formal wear',
        'Shirt brand outlet', 'Jeans fashion store', 'Dress party wear',
        'Shoes sports brand', 'Sandals comfort wear', 'Socks undergarments',
        'Bedsheet home textile', 'Curtains window decor', 'Carpet floor covering',
        'Pillow comfort items', 'Blanket winter wear', 'Towel bathroom',
        'Kitchen utensils set', 'Dinner plate set', 'Glass tumbler set',
        'Spoon fork cutlery', 'Cooking pot steel', 'Non stick pan buy'
    ]
    
    education_descriptions = [
        'University course fee', 'Book purchase online', 'Online course subscription',
        'Library late fees', 'Educational workshop', 'Professional certification',
        'Language learning app', 'Skill development course', 'Academic conference',
        'Research materials buy', 'Educational software license', 'Tutoring sessions',
        'School supplies stationery', 'Scientific calculator', 'Art class materials',
        'Coaching class fees', 'Entrance exam form', 'Study guide purchase',
        'Educational toys kids', 'Learning tablet buy', 'School bag purchase',
        'Notebook stationery', 'Pen pencil buy', 'Eraser geometry box',
        'Drawing materials art', 'Craft supplies hobby', 'Science project kit',
        'Language dictionary', 'Reference book buy', 'Atlas geography',
        'Educational DVD set', 'Learning software', 'Online tutorial fee',
        'Webinar registration', 'Seminar attendance', 'Workshop materials',
        'Training program fee', 'Certification exam', 'Professional license',
        'Continuing education', 'Skill assessment', 'Career counseling'
    ]
    
    # Combine all data with multiple repetitions for better training
    all_descriptions = []
    all_categories = []
    
    categories_data = {
        'Food': food_descriptions,
        'Travel': travel_descriptions,
        'Entertainment': entertainment_descriptions,
        'Healthcare': healthcare_descriptions,
        'Bills': bills_descriptions,
        'Shopping': shopping_descriptions,
        'Education': education_descriptions
    }
    
    # Repeat data multiple times with slight variations
    for category, descriptions in categories_data.items():
        for desc in descriptions:
            all_descriptions.append(desc)
            all_categories.append(category)
            
            # Add variations to increase dataset
            if len(desc.split()) > 2:
                # Add shortened version
                short_desc = ' '.join(desc.split()[:3])
                all_descriptions.append(short_desc)
                all_categories.append(category)
    
    category_data = {
        'description': all_descriptions,
        'category': all_categories
    }
    
    print(f"Created {len(all_descriptions)} descriptions and {len(all_categories)} categories")
    
    # Sample expense data for anomaly detection and budget prediction
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    expense_data = []
    for date in dates[:150]:  # Use first 150 days for more data
        # Generate realistic expenses
        daily_expenses = np.random.randint(2, 6)  # 2-5 expenses per day
        for _ in range(daily_expenses):
            category = np.random.choice(['Food', 'Travel', 'Entertainment', 'Shopping', 'Healthcare', 'Bills', 'Education'])
            
            # More realistic amount ranges by category (in INR)
            amount_ranges = {
                'Food': (30, 1200),
                'Travel': (50, 3000),
                'Entertainment': (100, 2000),
                'Shopping': (200, 8000),
                'Healthcare': (150, 5000),
                'Bills': (300, 4000),
                'Education': (500, 20000)
            }
            
            min_amt, max_amt = amount_ranges[category]
            amount = np.random.randint(min_amt, max_amt)
            
            # Add seasonal variations
            month = date.month
            if category == 'Bills' and month in [6, 7, 8]:  # Summer months - higher electricity
                amount = int(amount * 1.3)
            elif category == 'Entertainment' and month in [11, 12]:  # Festival season
                amount = int(amount * 1.2)
            
            expense_data.append({
                'date': date,
                'amount': amount,
                'category': category,
                'description': f'{category} expense on {date.strftime("%Y-%m-%d")}'
            })
    
    return pd.DataFrame(category_data), pd.DataFrame(expense_data)

def train_category_model(df, model_dir='ml_models'):
    """Train category prediction model with better parameters"""
    print("Training category prediction model...")
    
    # Initialize vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8  # Ignore terms that appear in more than 80% of documents
    )
    
    model = LogisticRegression(
        random_state=42, 
        max_iter=2000,  # Increased iterations
        C=1.0,  # Regularization parameter
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Preprocess descriptions
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        import re
        # Keep more characters, including numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return ' '.join(text.split())
    
    df['description_clean'] = df['description'].apply(preprocess_text)
    df = df[df['description_clean'].str.len() > 0]
    
    print(f"Training on {len(df)} samples")
    
    # Vectorize and split
    X = vectorizer.fit_transform(df['description_clean'])
    y = df['category']
    
    # Use stratified split with larger test size for better evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Category Model Accuracy: {accuracy:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save model and vectorizer
    with open(f"{model_dir}/category_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/preprocessors/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Category model saved to {model_dir}/")
    return model, vectorizer

def train_anomaly_model(df, model_dir='ml_models'):
    """Train anomaly detection model - COMPLETELY FIXED VERSION"""
    print("Training anomaly detection model...")
    
    try:
        # Create a robust feature set
        df_copy = df.copy()
        
        # STEP 1: Prepare Label Encoder
        categories = ['Food', 'Travel', 'Entertainment', 'Shopping', 'Healthcare', 'Bills', 'Education']
        le = LabelEncoder()
        le.fit(categories)  # Fit on known categories first
        
        # STEP 2: Encode categories (handle unknown categories)
        def safe_encode_category(category):
            try:
                return le.transform([str(category)])[0]
            except ValueError:
                # Return 0 for unknown categories (Food as default)
                return 0
        
        df_copy['category_encoded'] = df_copy['category'].apply(safe_encode_category)
        
        # STEP 3: Create temporal features
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
        
        # STEP 4: Create feature matrix
        feature_columns = ['amount', 'category_encoded', 'day_of_week']
        X = df_copy[feature_columns].values
        
        # STEP 5: Validate data
        if X.shape[0] < 20:
            print("Warning: Limited data. Creating augmented dataset...")
            # Create additional synthetic data for better training
            np.random.seed(42)
            synthetic_data = []
            for _ in range(100):
                amount = np.random.randint(50, 5000)
                category_enc = np.random.randint(0, len(categories))
                day_of_week = np.random.randint(0, 7)
                synthetic_data.append([amount, category_enc, day_of_week])
            
            X = np.vstack([X, np.array(synthetic_data)])
        
        print(f"Training anomaly model on {X.shape[0]} samples with {X.shape[1]} features")
        
        # STEP 6: Scale features (CRITICAL - must not be None!)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # STEP 7: Verify scaler is properly fitted
        assert scaler.mean_ is not None, "Scaler fitting failed - mean_ is None"
        assert scaler.scale_ is not None, "Scaler fitting failed - scale_ is None"
        print(f"✅ Scaler fitted successfully: mean={scaler.mean_[:2]}, scale={scaler.scale_[:2]}")
        
        # STEP 8: Train Isolation Forest
        model = IsolationForest(
            contamination=0.05,  # Expect 5% anomalies
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False
        )
        
        model.fit(X_scaled)
        
        # STEP 9: Test predictions
        predictions = model.predict(X_scaled)
        decision_scores = model.decision_function(X_scaled)
        
        anomaly_count = np.sum(predictions == -1)
        normal_count = np.sum(predictions == 1)
        
        print(f"Anomaly model trained successfully!")
        print(f"  - Normal transactions: {normal_count}")
        print(f"  - Anomalous transactions: {anomaly_count}")
        print(f"  - Anomaly rate: {anomaly_count/len(predictions)*100:.2f}%")
        print(f"  - Decision score range: [{decision_scores.min():.3f}, {decision_scores.max():.3f}]")
        
        # STEP 10: Create comprehensive model package with VALIDATION
        anomaly_data = {
            'model': model,
            'scaler': scaler,
            'label_encoder': le,
            'feature_columns': feature_columns,
            'training_stats': {
                'n_samples': X_scaled.shape[0],
                'n_features': X_scaled.shape[1],
                'anomaly_rate': anomaly_count/len(predictions),
                'decision_score_range': [float(decision_scores.min()), float(decision_scores.max())]
            }
        }
        
        # STEP 11: Validate all components before saving
        assert anomaly_data['model'] is not None, "Model is None"
        assert anomaly_data['scaler'] is not None, "Scaler is None"
        assert anomaly_data['scaler'].mean_ is not None, "Scaler mean_ is None"
        assert anomaly_data['scaler'].scale_ is not None, "Scaler scale_ is None"
        assert anomaly_data['label_encoder'] is not None, "Label encoder is None"
        
        # STEP 12: Test the scaler with sample data
        test_sample = np.array([[500, 0, 1]])  # Test transformation
        test_scaled = anomaly_data['scaler'].transform(test_sample)
        test_prediction = anomaly_data['model'].predict(test_scaled)
        print(f"✅ Model validation test: Sample {test_sample[0]} -> Prediction {test_prediction[0]}")
        
        # STEP 13: Save with error handling
        model_path = f"{model_dir}/anomaly_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(anomaly_data, f)
        
        # STEP 14: Verify saved model
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
            assert loaded_data['scaler'] is not None, "Loaded scaler is None"
            assert loaded_data['scaler'].mean_ is not None, "Loaded scaler mean_ is None"
            
        print(f"✅ Anomaly model saved and verified at {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in train_anomaly_model: {e}")
        import traceback
        traceback.print_exc()
        raise e  # Re-raise to stop execution

def train_depletion_model(df, budget=50000, model_dir='ml_models'):
    """Train budget depletion prediction model with better features"""
    print("Training budget depletion model...")
    
    try:
        # Prepare data for time series prediction
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate daily and cumulative expenses
        daily_expenses = df.groupby(df['date'].dt.date)['amount'].sum()
        
        if len(daily_expenses) < 7:
            print("Warning: Not enough data for depletion model. Creating synthetic data.")
            # Create synthetic daily expenses
            dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
            daily_expenses = pd.Series(
                np.random.randint(500, 2000, size=60),
                index=dates.date
            )
        
        daily_expenses_cumsum = daily_expenses.cumsum()
        
        # Create more sophisticated features
        start_date = daily_expenses.index[0]
        X = []
        y = []
        
        for i, (date, cumulative) in enumerate(daily_expenses_cumsum.items()):
            days_since_start = (date - start_date).days + 1
            
            # Add more features
            day_of_week = date.weekday()  # 0 = Monday, 6 = Sunday
            day_of_month = date.day
            month = pd.to_datetime(date).month
            
            # Moving average of last 7 days (if available)
            if i >= 6:
                recent_avg = daily_expenses.iloc[i-6:i+1].mean()
            else:
                recent_avg = daily_expenses.iloc[:i+1].mean()
            
            X.append([days_since_start, day_of_week, day_of_month, month, recent_avg])
            y.append(cumulative)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training depletion model on {len(X)} data points")
        
        # Train model with regularization
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X, y)
        
        # Evaluate
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        
        print(f"Depletion model trained. RMSE: {rmse:.2f}")
        
        # Save model with additional info
        depletion_data = {
            'model': model,
            'budget': budget,
            'start_date': start_date,
            'feature_names': ['days_since_start', 'day_of_week', 'day_of_month', 'month', 'recent_avg'],
            'training_stats': {
                'rmse': rmse,
                'n_samples': len(X)
            }
        }
        
        with open(f"{model_dir}/depletion_model.pkl", 'wb') as f:
            pickle.dump(depletion_data, f)
        
        print(f"Depletion model saved to {model_dir}/")
        return model
        
    except Exception as e:
        print(f"Error in train_depletion_model: {e}")
        import traceback
        traceback.print_exc()
        raise e

def train_smart_allocation_model(df, model_dir='ml_models'):
    """Train Smart Allocation Planner with category-specific regressors"""
    print("Training Smart Allocation Planner...")
    
    try:
        # Prepare data
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Train per-category spending predictors
        category_regressors = {}
        
        for category in df['category'].unique():
            try:
                cat_data = df[df['category'] == category].copy()
                
                if len(cat_data) < 10:  # Skip categories with too little data
                    print(f"  Skipping {category}: insufficient data ({len(cat_data)} samples)")
                    continue
                    
                # Create features for category-specific model
                cat_data['day_num'] = (cat_data['date'] - cat_data['date'].min()).dt.days
                cat_data['day_of_week'] = cat_data['date'].dt.dayofweek
                cat_data['day_of_month'] = cat_data['date'].dt.day
                cat_data['month'] = cat_data['date'].dt.month
                
                # Group by day and sum amounts
                daily_cat_spending = cat_data.groupby(['day_num', 'day_of_week'])['amount'].sum().reset_index()
                
                if len(daily_cat_spending) < 5:
                    print(f"  Skipping {category}: insufficient daily data ({len(daily_cat_spending)} days)")
                    continue
                    
                # Features: day_num, day_of_week
                X_cat = daily_cat_spending[['day_num', 'day_of_week']].values
                y_cat = daily_cat_spending['amount'].values
                
                # Train simple linear regression for this category
                cat_model = LinearRegression()
                cat_model.fit(X_cat, y_cat)
                
                # Evaluate
                cat_pred = cat_model.predict(X_cat)
                cat_rmse = np.sqrt(mean_squared_error(y_cat, cat_pred))
                
                category_regressors[str(category)] = {
                    'model': cat_model,
                    'rmse': cat_rmse,
                    'n_samples': len(X_cat)
                }
                print(f"  {category} regressor: RMSE = {cat_rmse:.2f}, samples = {len(X_cat)}")
                
            except Exception as e:
                print(f"  Failed to train {category} regressor: {e}")
                continue
        
        # Train main allocation optimization model
        allocation_model = None
        try:
            # Calculate category ratios and trends
            category_totals = df.groupby('category')['amount'].sum()
            total_spending = category_totals.sum()
            category_ratios = category_totals / total_spending
            
            # Create features for allocation model
            features = []
            targets = []
            
            for category in category_ratios.index:
                cat_data = df[df['category'] == category]
                cat_std = cat_data['amount'].std()
                cat_mean = cat_data['amount'].mean()
                cat_count = len(cat_data)
                
                # Features: mean, std, count, day_of_week_mode
                mode_day = cat_data['date'].dt.dayofweek.mode().iloc[0] if len(cat_data) > 0 else 0
                
                features.append([cat_mean, cat_std, cat_count, mode_day])
                targets.append(category_ratios[category])
            
            if len(features) >= 3:  # Need at least 3 categories
                X_alloc = np.array(features)
                y_alloc = np.array(targets)
                
                # Train allocation model
                allocation_model = Ridge(alpha=0.5, random_state=42)
                allocation_model.fit(X_alloc, y_alloc)
                
                alloc_pred = allocation_model.predict(X_alloc)
                alloc_rmse = np.sqrt(mean_squared_error(y_alloc, alloc_pred))
                
                print(f"Allocation model trained. RMSE: {alloc_rmse:.4f}")
            else:
                print("Not enough categories for allocation model training")
                
        except Exception as e:
            print(f"Failed to train allocation model: {e}")
            allocation_model = None
        
        # Save smart allocation models
        smart_allocation_data = {
            'allocation_model': allocation_model,
            'category_regressors': category_regressors,
            'category_list': list(df['category'].unique()),
            'training_stats': {
                'n_categories': len(category_regressors),
                'total_samples': len(df)
            }
        }
        
        with open(f"{model_dir}/smart_allocation_model.pkl", 'wb') as f:
            pickle.dump(smart_allocation_data, f)
        
        print(f"Smart Allocation Planner saved to {model_dir}/")
        print(f"Trained regressors for {len(category_regressors)} categories")
        
        return allocation_model, category_regressors
        
    except Exception as e:
        print(f"Error in train_smart_allocation_model: {e}")
        import traceback
        traceback.print_exc()
        raise e

def validate_saved_models(model_dir='ml_models'):
    """Validate that all models were saved correctly and can be loaded"""
    print("\nValidating saved models...")
    
    model_files = [
        'category_model.pkl',
        'preprocessors/tfidf_vectorizer.pkl',
        'anomaly_model.pkl',
        'depletion_model.pkl',
        'smart_allocation_model.pkl'
    ]
    
    all_valid = True
    
    for model_file in model_files:
        file_path = os.path.join(model_dir, model_file)
        try:
            if not os.path.exists(file_path):
                print(f"❌ {model_file}: File not found")
                all_valid = False
                continue
                
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                file_size = os.path.getsize(file_path)
                
                # Additional validation for anomaly model
                if model_file == 'anomaly_model.pkl':
                    required_keys = ['model', 'scaler', 'label_encoder']
                    missing_keys = [key for key in required_keys if key not in model_data or model_data[key] is None]
                    if missing_keys:
                        print(f"❌ {model_file}: Missing components: {missing_keys}")
                        all_valid = False
                        continue
                    
                    # Validate scaler specifically
                    scaler = model_data['scaler']
                    if scaler.mean_ is None or scaler.scale_ is None:
                        print(f"❌ {model_file}: Scaler not properly fitted")
                        all_valid = False
                        continue
                
                print(f"✅ {model_file}: {file_size} bytes - OK")
                
        except Exception as e:
            print(f"❌ {model_file}: Error loading - {e}")
            all_valid = False
    
    if all_valid:
        print("✅ All models validated successfully!")
    else:
        print("❌ Some models failed validation!")
    
    return all_valid

def test_models_functionality(model_dir='ml_models'):
    """Test that models can make predictions"""
    print("\nTesting model functionality...")
    
    try:
        # Test category model
        print("Testing category model...")
        with open(f"{model_dir}/category_model.pkl", 'rb') as f:
            cat_model = pickle.load(f)
        with open(f"{model_dir}/preprocessors/tfidf_vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        
        test_desc = "McDonald's burger meal"
        test_vec = vectorizer.transform([test_desc])
        prediction = cat_model.predict(test_vec)[0]
        probability = cat_model.predict_proba(test_vec)[0].max()
        print(f"✅ Category model: '{test_desc}' -> {prediction} (confidence: {probability:.3f})")
        
        # Test anomaly model with PROPER validation
        print("Testing anomaly model...")
        with open(f"{model_dir}/anomaly_model.pkl", 'rb') as f:
            anomaly_data = pickle.load(f)
        
        anomaly_model = anomaly_data['model']
        scaler = anomaly_data['scaler']
        label_encoder = anomaly_data['label_encoder']
        
        # Validate scaler is properly loaded
        assert scaler is not None, "Scaler is None"
        assert scaler.mean_ is not None, "Scaler mean_ is None"
        assert scaler.scale_ is not None, "Scaler scale_ is None"
        print(f"✅ Scaler validation passed: mean shape {scaler.mean_.shape}, scaleShape {scaler.scale_.shape}")
        
        # Create test data that matches training format
        test_amount = 500
        test_category = 'Food'
        test_day_of_week = 1  # Tuesday
        
        # Encode category
        try:
            test_category_encoded = label_encoder.transform([test_category])[0]
        except:
            test_category_encoded = 0  # Fallback
        
        # Create feature vector
        test_features = np.array([[test_amount, test_category_encoded, test_day_of_week]])
        test_features_scaled = scaler.transform(test_features)
        
        anomaly_pred = anomaly_model.predict(test_features_scaled)[0]
        decision_score = anomaly_model.decision_function(test_features_scaled)[0]
        anomaly_status = "Normal" if anomaly_pred == 1 else "Anomaly"
        print(f"✅ Anomaly model: Amount {test_amount}, Category {test_category} -> {anomaly_status} (score: {decision_score:.3f})")
        
        # Test depletion model
        print("Testing depletion model...")
        with open(f"{model_dir}/depletion_model.pkl", 'rb') as f:
            depletion_data = pickle.load(f)
        
        depletion_model = depletion_data['model']
        test_features = np.array([[30, 1, 15, 6, 800]])  # day 30, Tuesday, 15th, June, 800 avg
        depletion_pred = depletion_model.predict(test_features)[0]
        print(f"✅ Depletion model: Predicted cumulative expense: ₹{depletion_pred:.0f}")
        
        # Test smart allocation model
        print("Testing smart allocation model...")
        with open(f"{model_dir}/smart_allocation_model.pkl", 'rb') as f:
            allocation_data = pickle.load(f)
        
        allocation_model = allocation_data['allocation_model']
        category_regressors = allocation_data['category_regressors']
        
        if allocation_model:
            test_alloc_features = np.array([[500, 200, 10, 1]])  # mean=500, std=200, count=10, mode_day=1
            alloc_pred = allocation_model.predict(test_alloc_features)[0]
            print(f"✅ Allocation model: Predicted ratio: {alloc_pred:.3f}")
        else:
            print("✅ Allocation model: No model trained (insufficient data)")
        
        if category_regressors:
            sample_category = list(category_regressors.keys())[0]
            sample_regressor = category_regressors[sample_category]['model']
            test_cat_features = np.array([[30, 1]])  # day 30, Monday
            cat_pred = sample_regressor.predict(test_cat_features)[0]
            print(f"✅ Category regressor ({sample_category}): Predicted spending: ₹{cat_pred:.0f}")
        else:
            print("✅ Category regressors: No regressors trained (insufficient data)")
        
        print("✅ All models are functional!")
        return True
        
    except Exception as e:
        print(f"❌ Model functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
def create_model_info_file(model_dir='ml_models'):
    """Create an info file with model details - FIXED VERSION"""
    info_content = f"""# ML Models Information
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models Trained:
1. **Category Prediction Model** (category_model.pkl)
   - Predicts expense category from description
   - Uses TF-IDF vectorization + Logistic Regression
   - Vectorizer saved in preprocessors/tfidf_vectorizer.pkl

2. **Anomaly Detection Model** (anomaly_model.pkl)
   - Detects unusual spending patterns
   - Uses Isolation Forest algorithm
   - Includes scaler and label encoder

3. **Budget Depletion Model** (depletion_model.pkl)
   - Predicts when budget will be exhausted
   - Uses Ridge regression with temporal features
   - Includes budget amount and start date

4. **Smart Allocation Planner** (smart_allocation_model.pkl)
   - Optimizes budget allocation across categories
   - Includes category-specific spending predictors
   - Uses multiple regression models

## Usage:
Load models using pickle:
```python
import pickle
with open('{model_dir}/category_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## Files Structure:
- ml_models/
  - category_model.pkl
  - anomaly_model.pkl
  - depletion_model.pkl
  - smart_allocation_model.pkl
  - preprocessors/
    - tfidf_vectorizer.pkl
  - model_info.txt
"""
    
    # FIX: Use UTF-8 encoding explicitly
    try:
        with open(f"{model_dir}/model_info.txt", 'w', encoding='utf-8') as f:
            f.write(info_content)
        print(f"Model information saved to {model_dir}/model_info.txt")
    except Exception as e:
        print(f"Warning: Could not save model info file: {e}")
        # Continue without failing the entire training process

def main():
    """Main training function with comprehensive error handling"""
    print("=" * 60)
    print("🚀 STARTING ML MODEL TRAINING - FIXED VERSION")
    print("=" * 60)
    
    # Create model directory (clean start)
    model_dir = 'ml_models'
    clean_model_directory(model_dir)
    
    try:
        # Generate training data
        print("\n📊 Generating comprehensive training data...")
        category_df, expense_df = create_sample_training_data()
        
        print(f"✅ Category training data: {len(category_df)} samples")
        print(f"✅ Expense training data: {len(expense_df)} samples")
        
        # Train all models with error handling
        models_trained = []
        
        # 1. Train category prediction model
        print("\n" + "="*50)
        print("🏷️  TRAINING CATEGORY PREDICTION MODEL")
        print("="*50)
        try:
            category_model, vectorizer = train_category_model(category_df, model_dir)
            models_trained.append("Category Model")
        except Exception as e:
            print(f"❌ Category model training failed: {e}")
            raise e
        
        # 2. Train anomaly detection model (CRITICAL FIX)
        print("\n" + "="*50)
        print("🔍 TRAINING ANOMALY DETECTION MODEL")
        print("="*50)
        try:
            anomaly_model = train_anomaly_model(expense_df, model_dir)
            models_trained.append("Anomaly Model")
        except Exception as e:
            print(f"❌ Anomaly model training failed: {e}")
            raise e
        
        # 3. Train budget depletion model
        print("\n" + "="*50)
        print("📈 TRAINING BUDGET DEPLETION MODEL")
        print("="*50)
        try:
            depletion_model = train_depletion_model(expense_df, budget=50000, model_dir=model_dir)
            models_trained.append("Depletion Model")
        except Exception as e:
            print(f"❌ Depletion model training failed: {e}")
            raise e
        
        # 4. Train smart allocation planner
        print("\n" + "="*50)
        print("🧠 TRAINING SMART ALLOCATION PLANNER")
        print("="*50)
        try:
            allocation_model, category_regressors = train_smart_allocation_model(expense_df, model_dir)
            models_trained.append("Smart Allocation Planner")
        except Exception as e:
            print(f"❌ Smart allocation model training failed: {e}")
            raise e
        
        # Summary
        print("\n" + "="*60)
        print("📋 TRAINING SUMMARY")
        print("="*60)
        print(f"✅ Successfully trained: {len(models_trained)} models")
        for model in models_trained:
            print(f"   • {model}")
        
        print(f"\n📁 Models saved in: {os.path.abspath(model_dir)}/")
        
        # Create info file
        create_model_info_file(model_dir)
        
        # Validate models
        print("\n" + "="*50)
        print("🔧 MODEL VALIDATION")
        print("="*50)
        validation_success = validate_saved_models(model_dir)
        functionality_success = test_models_functionality(model_dir)
        
        # Final status
        print("\n" + "="*60)
        if validation_success and functionality_success and len(models_trained) >= 4:
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("✅ All models are trained, validated, and functional!")
            print("🚀 Your Smart Expense Tracker ML backend is ready!")
        elif len(models_trained) >= 3:
            print("⚠️  TRAINING MOSTLY SUCCESSFUL")
            print("✅ Core models are working, some advanced features may be limited")
            print("🔧 Your expense tracker will work with most functionality")
        else:
            print("❌ TRAINING FAILED")
            print("🔧 Please check the errors above and retry")
        
        print("\n📖 Next Steps:")
        print("   1. The ml_models folder is ready in your current directory")
        print("   2. Run your expense tracker application")
        print("   3. The models will be automatically loaded when needed")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Please check your Python environment and dependencies")

if __name__ == "__main__":
    main()
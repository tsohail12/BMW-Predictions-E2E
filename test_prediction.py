"""
Test script to verify prediction pipeline works correctly
Run this before starting the web app
"""

import pandas as pd
import joblib
from pathlib import Path

print("="*70)
print("BMW PRICE PREDICTION - PIPELINE TEST")
print("="*70)

# Step 1: Check if artifacts exist
print("\n1. Checking if artifacts exist...")
artifacts_to_check = [
    ("Model", "artifacts/model_trainer/model.pkl"),
    ("Scaler", "artifacts/data_transformation/scaler.pkl"),
    ("Encoders", "artifacts/data_transformation/label_encoders.pkl"),
    ("Feature Name", "artifacts/data_transformation/feature_names.pkl")
]

all_exist = True
for name, path in artifacts_to_check:
    if Path(path).exists():
        print(f"   ✓ {name}: {path}")
    else:
        print(f"   ✗ {name} NOT FOUND: {path}")
        all_exist = False

if not all_exist:
    print("\n⚠️  ERROR: Some artifacts are missing!")
    print("   Run: python main.py")
    exit(1)

# Step 2: Load artifacts
print("\n2. Loading artifacts...")
try:
    model = joblib.load("artifacts/model_trainer/model.pkl")
    scaler = joblib.load("artifacts/data_transformation/scaler.pkl")
    label_encoders = joblib.load("artifacts/data_transformation/label_encoders.pkl")
    feature_names = joblib.load("artifacts/data_transformation/feature_names.pkl") 
    print("   ✓ All artifacts loaded successfully!")
except Exception as e:
    print(f"   ✗ Error loading artifacts: {e}")
    exit(1)

# Step 3: Test preprocessing
print("\n3. Testing preprocessing...")
test_data = {
    'model': '3 Series',
    'year': 2019,
    'transmission': 'Semi-Auto',
    'mileage': 15000,
    'fuelType': 'Diesel',
    'tax': 145,
    'mpg': 65.7,
    'engineSize': 2.0
}

print(f"   Input: {test_data}")

try:
    # Create DataFrame
    df = pd.DataFrame([test_data])
    print(f"   ✓ DataFrame created: {df.shape}")
    
    # Feature engineering
    df['car_age'] = 2026 - df['year']
    print(f"   ✓ car_age created: {df['car_age'].values[0]}")
    
    # Encode categorical
    for col in ['model', 'transmission', 'fuelType']:
        le = label_encoders[col]
        df[f'{col}_encoded'] = le.transform(df[col])
    print(f"   ✓ Categorical features encoded")
    
    # Drop original categorical
    df = df.drop(columns=['model', 'transmission', 'fuelType'])
    
    # Reorder columns to match training
    # Check what columns scaler expects
    if hasattr(scaler, 'feature_names_in_'):
        expected_columns = list(scaler.feature_names_in_)
        print(f"   Scaler expects: {expected_columns}")
    else:
        # Fallback to default order
        expected_columns = [
            'year', 'mileage', 'mpg', 'engineSize', 'tax', 'car_age',
            'model_encoded', 'transmission_encoded', 'fuelType_encoded'
        ]
    
    df = df[expected_columns]
    print(f"   ✓ Columns reordered: {list(df.columns)}")
    
    # Scale - ensure DataFrame is passed to preserve column names
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=expected_columns
    )
    print(f"   ✓ Features scaled: {df_scaled.shape}")
    
except Exception as e:
    print(f"   ✗ Preprocessing error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Test prediction
print("\n4. Testing prediction...")
try:
    prediction = model.predict(df_scaled)
    predicted_price = float(prediction[0])
    print(f"   ✓ Prediction successful: £{predicted_price:,.2f}")
    
    # Confidence interval
    lower = predicted_price * 0.92
    upper = predicted_price * 1.08
    print(f"   ✓ Confidence interval: £{lower:,.2f} - £{upper:,.2f}")
    
except Exception as e:
    print(f"   ✗ Prediction error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Test multiple cars
print("\n5. Testing batch prediction...")
batch_data = pd.DataFrame([
    {
        'model': '1 Series',
        'year': 2018,
        'transmission': 'Manual',
        'mileage': 25000,
        'fuelType': 'Petrol',
        'tax': 30,
        'mpg': 54.3,
        'engineSize': 1.5
    },
    {
        'model': '5 Series',
        'year': 2020,
        'transmission': 'Automatic',
        'mileage': 10000,
        'fuelType': 'Diesel',
        'tax': 150,
        'mpg': 60.1,
        'engineSize': 3.0
    }
])

try:
    # Preprocess batch
    batch_df = batch_data.copy()
    batch_df['car_age'] = 2026 - batch_df['year']
    
    for col in ['model', 'transmission', 'fuelType']:
        batch_df[f'{col}_encoded'] = label_encoders[col].transform(batch_df[col])
        batch_df = batch_df.drop(columns=[col])
    
    batch_df = batch_df[expected_columns]
    batch_scaled = scaler.transform(batch_df)
    
    # Predict
    predictions = model.predict(batch_scaled)
    
    print(f"   ✓ Batch prediction successful:")
    for i, (idx, row) in enumerate(batch_data.iterrows()):
        print(f"      Car {i+1} ({row['model']}, {row['year']}): £{predictions[i]:,.2f}")
    
except Exception as e:
    print(f"   ✗ Batch prediction error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Success!
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour prediction pipeline is working correctly!")
print("You can now run: python app.py")
print("="*70 + "\n")
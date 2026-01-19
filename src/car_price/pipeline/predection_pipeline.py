import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from car_price import logger


class PredictionPipeline:
    def __init__(self):
        # Load the trained model
        self.model_path = Path("artifacts/model_trainer/model.pkl")
        self.scaler_path = Path("artifacts/data_transformation/scaler.pkl")
        self.encoder_path = Path("artifacts/data_transformation/label_encoders.pkl")
        self.feature_names_path = Path("artifacts/data_transformation/feature_names.pkl")
        
        logger.info("Initializing Prediction Pipeline")
        self.load_artifacts()
    
    
    def load_artifacts(self):
        """Load model, scaler, and encoders"""
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
            
            # Load scaler
            logger.info(f"Loading scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Scaler loaded successfully")
            
            # Load encoders
            logger.info(f"Loading encoders from {self.encoder_path}")
            self.label_encoders = joblib.load(self.encoder_path)
            logger.info("Encoders loaded successfully")
            
            # Load feature names
            logger.info(f"Loading feature names from {self.feature_names_path}")
            self.feature_names = joblib.load(self.feature_names_path)
            logger.info(f"Feature names loaded: {self.feature_names}")

            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise e
    
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        
        Args:
            input_data: dict or DataFrame with features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()
            
            logger.info(f"Preprocessing input data with shape: {df.shape}")
            
            # Feature engineering - Create car_age
            df['car_age'] = 2026 - df['year']
            logger.info("Created 'car_age' feature")
            
            # Encode categorical features
            categorical_features = ['model', 'transmission', 'fuelType']
            
            for col in categorical_features:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            
            for col in categorical_features:
                if col in df.columns:
                    le = self.label_encoders[col]
                    df[f'{col}_encoded'] = le.transform(df[col])
                    logger.info(f"Encoded '{col}'")
            
            # Drop original categorical columns
            df = df.drop(columns=categorical_features, errors='ignore')
            
            # Ensure correct column order (same as training)
            # Align columns EXACTLY as training
            df = df[self.feature_names]

            # Scale features
            df_scaled = self.scaler.transform(df)
            df_scaled = pd.DataFrame(df_scaled, columns=self.feature_names)

            
            logger.info("Input preprocessing completed")
            
            return df_scaled
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise e
    
    
    def predict(self, input_data):
        """
        Make price prediction
        
        Args:
            input_data: dict or DataFrame with car features
            
        Returns:
            Predicted price(s)
        """
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            logger.info("Making prediction")
            prediction = self.model.predict(processed_data)
            
            logger.info(f"Prediction completed: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e
    
    
    def predict_with_details(self, input_data):
        """
        Make prediction and return detailed results
        
        Args:
            input_data: dict with car features
            
        Returns:
            Dictionary with prediction and input details
        """
        try:
            # Make prediction
            prediction = self.predict(input_data)
            
            # Prepare result
            result = {
                'predicted_price': float(prediction[0]),
                'predicted_price_formatted': f"£{prediction[0]:,.2f}",
                'input_features': input_data,
                'confidence_interval': {
                    'lower': float(prediction[0] * 0.92),  # Approximate 8% margin
                    'upper': float(prediction[0] * 1.08)
                }
            }
            
            logger.info(f"Prediction result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detailed prediction: {e}")
            raise e


class CustomData:
    """
    Class for handling custom input data from user
    """
    def __init__(self,
                model: str,
                year: int,
                transmission: str,
                mileage: int,
                fuelType: str,
                tax: int,
                mpg: float,
                engineSize: float):
        
        self.model = model
        self.year = year
        self.transmission = transmission
        self.mileage = mileage
        self.fuelType = fuelType
        self.tax = tax
        self.mpg = mpg
        self.engineSize = engineSize
    
    
    def get_data_as_dict(self):
        """Convert custom data to dictionary"""
        try:
            custom_data_input_dict = {
                "model": self.model,
                "year": self.year,
                "transmission": self.transmission,
                "mileage": self.mileage,
                "fuelType": self.fuelType,
                "tax": self.tax,
                "mpg": self.mpg,
                "engineSize": self.engineSize
            }
            
            return custom_data_input_dict
            
        except Exception as e:
            logger.error(f"Error creating data dictionary: {e}")
            raise e
    
    
    def get_data_as_dataframe(self):
        """Convert custom data to DataFrame"""
        try:
            data_dict = {
                "model": [self.model],
                "year": [self.year],
                "transmission": [self.transmission],
                "mileage": [self.mileage],
                "fuelType": [self.fuelType],
                "tax": [self.tax],
                "mpg": [self.mpg],
                "engineSize": [self.engineSize]
            }
            return pd.DataFrame(data_dict)
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            raise e


# Example usage
if __name__ == "__main__":
    # Example 1: Single prediction with dictionary
    print("="*60)
    print("Example 1: Single Prediction")
    print("="*60)
    
    sample_data = {
        'model': ' 3 Series',
        'year': 2019,
        'transmission': 'Semi-Auto',
        'mileage': 15000,
        'fuelType': 'Diesel',
        'tax': 145,
        'mpg': 65.7,
        'engineSize': 2.0
    }
    
    pipeline = PredictionPipeline()
    result = pipeline.predict_with_details(sample_data)
    
    print(f"\nInput Features:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nPredicted Price: {result['predicted_price_formatted']}")
    print(f"Confidence Interval: £{result['confidence_interval']['lower']:,.2f} - £{result['confidence_interval']['upper']:,.2f}")
    
    
    # Example 2: Using CustomData class
    print("\n" + "="*60)
    print("Example 2: Using CustomData Class")
    print("="*60)
    
    custom_car = CustomData(
        model=' 5 Series',
        year=2020,
        transmission='Automatic',
        mileage=10000,
        fuelType='Petrol',
        tax=150,
        mpg=50.4,
        engineSize=3.0
    )
    
    df = custom_car.get_data_as_dataframe()
    prediction = pipeline.predict(df)
    
    print(f"\nPredicted Price: £{prediction[0]:,.2f}")
    
    
    # Example 3: Batch prediction
    print("\n" + "="*60)
    print("Example 3: Batch Prediction")
    print("="*60)
    
    batch_data = pd.DataFrame({
        'model': [' 1 Series', ' X3', ' 4 Series'],
        'year': [2018, 2019, 2020],
        'transmission': ['Manual', 'Semi-Auto', 'Automatic'],
        'mileage': [25000, 12000, 8000],
        'fuelType': ['Petrol', 'Diesel', 'Hybrid'],
        'tax': [30, 145, 140],
        'mpg': [54.3, 60.1, 62.8],
        'engineSize': [1.5, 2.0, 2.0]
    })
    
    predictions = pipeline.predict(batch_data)
    
    batch_data['predicted_price'] = predictions
    batch_data['predicted_price_formatted'] = batch_data['predicted_price'].apply(lambda x: f"£{x:,.2f}")
    
    print("\nBatch Predictions:")
    print(batch_data[['model', 'year', 'mileage', 'predicted_price_formatted']])
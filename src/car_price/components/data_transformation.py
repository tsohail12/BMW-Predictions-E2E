import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from car_price import logger
from car_price.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = None


    def load_data(self):
        """Load the validated dataset"""
        logger.info(f"Loading data from {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df


    def feature_engineering(self, df):
        """Create new features"""
        logger.info("Starting feature engineering")
        
        # Create car age feature
        df['car_age'] = 2026 - df['year']
        logger.info("Created 'car_age' feature")
        
        return df


    def encode_categorical_features(self, df, is_training=True):
        """Encode categorical features using Label Encoding"""
        logger.info(f"Encoding categorical features: {self.config.categorical_features}")
        
        df_encoded = df.copy()
        
        for col in self.config.categorical_features:
            df_encoded[col] = df_encoded[col].astype(str).str.strip()
        
        for col in self.config.categorical_features:
            if is_training:
                # Fit and transform during training
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                logger.info(f"Encoded '{col}' with {len(le.classes_)} unique values")
            else:
                # Only transform during inference
                le = self.label_encoders[col]
                df_encoded[f'{col}_encoded'] = le.transform(df_encoded[col])
        
        # Drop original categorical columns
        df_encoded = df_encoded.drop(columns=self.config.categorical_features)
        
        return df_encoded


    def scale_features(self, X_train, X_test):
        """Scale numerical features using StandardScaler"""
        if self.config.scaling:
            logger.info("Scaling features using StandardScaler")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrame to preserve column names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            logger.info("Feature scaling completed")
            return X_train_scaled, X_test_scaled
        else:
            logger.info("Scaling is disabled")
            return X_train, X_test


    def split_data(self, df):
        """Split data into train and test sets"""
        logger.info("Splitting data into train and test sets")
        
        # Separate features and target
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        logger.info(f"Train set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test


    def save_data(self, X_train, X_test, y_train, y_test):
        """Save processed train and test data"""
        logger.info("Saving processed data")
        
        # Combine features and target
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save to CSV
        train_path = os.path.join(self.config.root_dir, "train.csv")
        test_path = os.path.join(self.config.root_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Train data saved to: {train_path}")
        logger.info(f"Test data saved to: {test_path}")
        
        # Save encoders and scaler
        if self.label_encoders:
            encoder_path = os.path.join(self.config.root_dir, "label_encoders.pkl")
            joblib.dump(self.label_encoders, encoder_path)
            logger.info(f"Label encoders saved to: {encoder_path}")
        
        if self.scaler:
            scaler_path = os.path.join(self.config.root_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")


    def transform(self):
        """Main transformation pipeline"""
        try:
            # Load data
            df = self.load_data()
            
            # Feature engineering
            df = self.feature_engineering(df)
            
            # Encode categorical features
            df_encoded = self.encode_categorical_features(df, is_training=True)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(df_encoded)
            
            # Scale features
            X_train, X_test = self.scale_features(X_train, X_test)
            
            # Save processed data
            self.save_data(X_train, X_test, y_train, y_test)
            
            logger.info("Data transformation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e
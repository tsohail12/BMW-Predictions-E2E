import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from car_price import logger
from car_price.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def load_data(self):
        """Load train and test data"""
        logger.info(f"Loading training data from {self.config.train_data_path}")
        train_df = pd.read_csv(self.config.train_data_path)
        
        logger.info(f"Loading test data from {self.config.test_data_path}")
        test_df = pd.read_csv(self.config.test_data_path)
        
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df


    def prepare_data(self, train_df, test_df):
        """Prepare features and target variables"""
        logger.info("Preparing features and target variables")
        
        # Separate features and target
        X_train = train_df.drop(columns=[self.config.target_column])
        y_train = train_df[self.config.target_column]
        
        X_test = test_df.drop(columns=[self.config.target_column])
        y_test = test_df[self.config.target_column]
        
        logger.info(f"Feature columns: {list(X_train.columns)}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        return X_train, y_train, X_test, y_test


    def train_model(self, X_train, y_train):
        """Train Random Forest model with specified hyperparameters"""
        logger.info("="*60)
        logger.info("Training Random Forest Regressor")
        logger.info("="*60)
        
        # Log hyperparameters
        logger.info("Hyperparameters:")
        logger.info(f"  n_estimators: {self.config.n_estimators}")
        logger.info(f"  max_depth: {self.config.max_depth}")
        logger.info(f"  min_samples_split: {self.config.min_samples_split}")
        logger.info(f"  min_samples_leaf: {self.config.min_samples_leaf}")
        logger.info(f"  max_features: {self.config.max_features}")
        logger.info(f"  random_state: {self.config.random_state}")
        logger.info(f"  n_jobs: {self.config.n_jobs}")
        
        # Initialize model with hyperparameters from params.yaml
        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        # Train the model
        logger.info("Starting model training...")
        import time
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return model


    def save_model(self, model):
        """Save trained model"""
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")


    def train(self):
        """Main training pipeline"""
        try:
            # Load data
            train_df, test_df = self.load_data()
            
            # Prepare data
            X_train, y_train, X_test, y_test = self.prepare_data(train_df, test_df)
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Save model
            self.save_model(model)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise e
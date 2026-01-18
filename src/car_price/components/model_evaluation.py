import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from car_price import logger
from car_price.entity.config_entity import ModelEvaluationConfig
from car_price.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def load_data(self):
        """Load test data"""
        logger.info(f"Loading test data from {self.config.test_data_path}")
        test_df = pd.read_csv(self.config.test_data_path)
        logger.info(f"Test data loaded with shape: {test_df.shape}")
        return test_df


    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from {self.config.model_path}")
        model = joblib.load(self.config.model_path)
        logger.info("Model loaded successfully")
        return model


    def prepare_data(self, test_df):
        """Prepare test features and target"""
        X_test = test_df.drop(columns=[self.config.target_column])
        y_test = test_df[self.config.target_column]
        
        logger.info(f"Test features shape: {X_test.shape}")
        logger.info(f"Test target shape: {y_test.shape}")
        
        return X_test, y_test


    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        logger.info("Calculating evaluation metrics")
        
        metrics = {}
        
        # R² Score
        if 'r2_score' in self.config.metrics:
            r2 = r2_score(y_true, y_pred)
            metrics['r2_score'] = float(r2)
            logger.info(f"R² Score: {r2:.4f}")
        
        # RMSE
        if 'rmse' in self.config.metrics:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(rmse)
            logger.info(f"RMSE: £{rmse:.2f}")
        
        # MAE
        if 'mae' in self.config.metrics:
            mae = mean_absolute_error(y_true, y_pred)
            metrics['mae'] = float(mae)
            logger.info(f"MAE: £{mae:.2f}")
        
        # MAPE
        if 'mape' in self.config.metrics:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = float(mape)
            logger.info(f"MAPE: {mape:.2f}%")
        
        return metrics


    def save_metrics(self, metrics):
        """Save evaluation metrics to JSON file"""
        # Add model parameters to metrics
        evaluation_results = {
            "metrics": metrics,
            "model_parameters": {
                "n_estimators": self.config.all_params.RandomForest.n_estimators,
                "max_depth": self.config.all_params.RandomForest.max_depth,
                "min_samples_split": self.config.all_params.RandomForest.min_samples_split,
                "min_samples_leaf": self.config.all_params.RandomForest.min_samples_leaf,
                "max_features": self.config.all_params.RandomForest.max_features,
                "random_state": self.config.all_params.RandomForest.random_state
            }
        }
        
        # Save to JSON
        save_json(path=Path(self.config.metric_file_name), data=evaluation_results)
        logger.info(f"Metrics saved to: {self.config.metric_file_name}")


    def log_results(self, metrics):
        """Log evaluation results"""
        logger.info("="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        
        logger.info("\nPerformance Metrics:")
        for metric_name, metric_value in metrics.items():
            if metric_name == 'r2_score':
                logger.info(f"  R² Score: {metric_value:.4f}")
            elif metric_name == 'rmse':
                logger.info(f"  RMSE: £{metric_value:.2f}")
            elif metric_name == 'mae':
                logger.info(f"  MAE: £{metric_value:.2f}")
            elif metric_name == 'mape':
                logger.info(f"  MAPE: {metric_value:.2f}%")
        
        logger.info("\nModel Parameters:")
        logger.info(f"  n_estimators: {self.config.all_params.RandomForest.n_estimators}")
        logger.info(f"  max_depth: {self.config.all_params.RandomForest.max_depth}")
        logger.info(f"  min_samples_split: {self.config.all_params.RandomForest.min_samples_split}")
        logger.info(f"  min_samples_leaf: {self.config.all_params.RandomForest.min_samples_leaf}")
        logger.info(f"  max_features: {self.config.all_params.RandomForest.max_features}")
        
        logger.info("="*60)


    def evaluate(self):
        """Main evaluation pipeline"""
        try:
            # Load test data
            test_df = self.load_data()
            
            # Load model
            model = self.load_model()
            
            # Prepare data
            X_test, y_test = self.prepare_data(test_df)
            
            # Make predictions
            logger.info("Making predictions on test data")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Save metrics
            self.save_metrics(metrics)
            
            # Log results
            self.log_results(metrics)
            
            logger.info("Model evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise e
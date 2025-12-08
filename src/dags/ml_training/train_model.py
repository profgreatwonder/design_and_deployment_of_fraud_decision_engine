"""
GRU Model Training Script for Fraud Detection
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

from pathlib import Path
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


from data_preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    def __init__(self, config_path="/opt/airflow/config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.features_config = self.config['features']
        
        # Set MLflow tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def prepare_features(self, df):
        """Prepare features and target"""
        logger.info("Preparing features...")
        
        numerical_features = self.features_config['numerical']
        categorical_features = self.features_config['categorical']
        features = numerical_features + categorical_features
        
        X = df[features].copy()
        y = df['is_fraud'].copy()
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Fraud rate: {y.mean():.4f}")
        
        return X, y, numerical_features, categorical_features
    
    def encode_and_scale(self, X, numerical_features, categorical_features, fit=True):
        """Encode categorical and scale numerical features"""
        logger.info("Encoding and scaling features...")
        
        if fit:
            # Fit encoder
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_encoded = self.encoder.fit_transform(X[categorical_features])
            
            # Save encoder
            with open(self.model_config['encoder_path'], 'wb') as f:
                pickle.dump(self.encoder, f)
            logger.info(f"Encoder saved to {self.model_config['encoder_path']}")
        else:
            X_encoded = self.encoder.transform(X[categorical_features])
        
        # Create encoded dataframe
        X_encoded_df = pd.DataFrame(
            X_encoded, 
            columns=self.encoder.get_feature_names_out(categorical_features),
            index=X.index
        )
        
        # Combine with numerical features
        X = pd.concat([X.drop(columns=categorical_features), X_encoded_df], axis=1)
        
        if fit:
            # Fit scaler
            self.scaler = StandardScaler()
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
            
            # Save scaler
            with open(self.model_config['scaler_path'], 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {self.model_config['scaler_path']}")
        else:
            X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        logger.info(f"Final feature matrix shape after encoding: {X.shape}")
        
        return X
    
    def handle_imbalance(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        logger.info("Handling class imbalance with SMOTE...")
        logger.info(f"Before SMOTE - Class distribution:\n{y.value_counts()}")
        
        smote = SMOTE(
            random_state=self.model_config['random_state'],
            k_neighbors=self.model_config['smote_k_neighbors']
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"After SMOTE - Class distribution:\n{pd.Series(y_resampled).value_counts()}")
        logger.info(f"Resampled data shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def build_model(self, input_shape):
        """Build GRU model"""
        logger.info("Building GRU model...")
        
        model = Sequential([
            GRU(
                units=self.model_config['gru_units'],
                activation='relu',
                input_shape=input_shape
            ),
            Dropout(self.model_config['dropout_rate']),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )
        
        logger.info("Model built successfully")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self, X_train, y_train):
        """Train the model"""
        logger.info("Training model...")
        
        # Reshape for GRU (samples, timesteps, features)
        n_samples, n_features = X_train.shape
        X_train_reshaped = X_train.values.reshape(n_samples, 1, n_features)
        
        # Build model
        input_shape = (1, n_features)
        self.model = self.build_model(input_shape)
        
        # Train
        history = self.model.fit(
            X_train_reshaped,
            y_train,
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            validation_split=self.model_config['validation_split'],
            verbose=1
        )
        
        logger.info("Training completed")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        logger.info("Evaluating model...")
        
        # Reshape for GRU
        n_samples, n_features = X_test.shape
        X_test_reshaped = X_test.values.reshape(n_samples, 1, n_features)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test_reshaped)
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        loss, accuracy, recall, precision = self.model.evaluate(X_test_reshaped, y_test, verbose=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Test Loss: {loss:.4f}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred_binary))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'auc_roc': auc_roc
        }
    
    def save_model(self):
        """Save trained model"""
        self.model.save(self.model_config['save_path'])
        logger.info(f"Model saved to {self.model_config['save_path']}")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        with mlflow.start_run():
            # Load and preprocess data
            preprocessor = DataPreprocessor()
            df = preprocessor.process_all()
            
            # Prepare features
            X, y, numerical_features, categorical_features = self.prepare_features(df)
            
            # Encode and scale
            X_processed = self.encode_and_scale(X, numerical_features, categorical_features, fit=True)
            
            # Handle imbalance
            X_resampled, y_resampled = self.handle_imbalance(X_processed, y)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled,
                test_size=self.model_config['test_size'],
                random_state=self.model_config['random_state']
            )
            
            logger.info(f"Training set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")
            
            # Log parameters
            mlflow.log_params(self.model_config)
            
            # Train model
            history = self.train(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate(X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save model
            self.save_model()
            mlflow.keras.log_model(self.model, "model")
            
            logger.info("\n=== Training pipeline completed successfully ===")
            
            return self.model, metrics


if __name__ == "__main__":
    trainer = FraudDetectionTrainer()
    model, metrics = trainer.run_training_pipeline()
    
    print("\n=== Final Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
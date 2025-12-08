"""
Real-time Fraud Decisioning Engine
Returns: ALLOW, REJECT, or PEND decisions
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging

from datetime import datetime
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDecisionEngine:
    def __init__(self, config_path="/opt/airflow/config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.features_config = self.config['features']
        self.decision_config = self.config['decision']
        self.high_risk_mccs = self.config['high_risk_mccs']
        
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model, scaler, and encoder"""
        logger.info("Loading model artifacts...")
        
        # Load model
        self.model = keras.models.load_model(self.model_config['save_path'])
        logger.info(f"Model loaded from {self.model_config['save_path']}")
        
        # Load scaler
        with open(self.model_config['scaler_path'], 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {self.model_config['scaler_path']}")
        
        # Load encoder
        with open(self.model_config['encoder_path'], 'rb') as f:
            self.encoder = pickle.load(f)
        logger.info(f"Encoder loaded from {self.model_config['encoder_path']}")
    
    def preprocess_transaction(self, transaction: Dict) -> pd.DataFrame:
        """Preprocess a single transaction for prediction"""
        
        # Create dataframe
        df = pd.DataFrame([transaction])
        
        # Extract datetime features if timestamp is present
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
        
        # Engineer features (simplified for real-time)
        # In production, these would come from a feature store or cache
        
        # Credit utilization
        if 'credit_limit' in df.columns and 'amount' in df.columns:
            df['credit_utilization'] = np.where(
                df['credit_limit'] != 0,
                df['amount'] / df['credit_limit'],
                0
            )
        else:
            df['credit_utilization'] = 0
        
        # High-risk MCC flag
        if 'mcc' in df.columns:
            df['is_high_risk_mcc'] = df['mcc'].isin(self.high_risk_mccs).astype(int)
        else:
            df['is_high_risk_mcc'] = 0
        
        # Set default values for missing engineered features
        # In production, these would be retrieved from feature store
        default_features = {
            'transactions_per_day_past': 0,
            'transactions_per_week_past': 0,
            'rolling_mean_3day': 0,
            'rolling_std_3day': 0,
            'age_at_acct_open': 0,
            'card_age': 0
        }
        
        for feature, default_value in default_features.items():
            if feature not in df.columns:
                df[feature] = default_value
        
        # Fill missing values
        numerical_features = self.features_config['numerical']
        categorical_features = self.features_config['categorical']
        
        for feature in numerical_features:
            if feature not in df.columns:
                df[feature] = 0
        
        for feature in categorical_features:
            if feature not in df.columns:
                df[feature] = 'Unknown'
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and process features for prediction"""
        
        numerical_features = self.features_config['numerical']
        categorical_features = self.features_config['categorical']
        
        # Ensure all required features exist
        for feature in numerical_features + categorical_features:
            if feature not in df.columns:
                if feature in numerical_features:
                    df[feature] = 0
                else:
                    df[feature] = 'Unknown'
        
        # Encode categorical features
        X_encoded = self.encoder.transform(df[categorical_features])
        X_encoded_df = pd.DataFrame(
            X_encoded,
            columns=self.encoder.get_feature_names_out(categorical_features),
            index=df.index
        )
        
        # Combine with numerical features
        X = pd.concat([df[numerical_features], X_encoded_df], axis=1)
        
        # Scale numerical features
        X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        # Reshape for GRU (samples, timesteps, features)
        n_samples, n_features = X.shape
        X_reshaped = X.values.reshape(n_samples, 1, n_features)
        
        return X_reshaped
    
    def predict_fraud_probability(self, transaction: Dict) -> float:
        """Predict fraud probability for a transaction"""
        
        # Preprocess
        df = self.preprocess_transaction(transaction)
        
        # Extract features
        X = self.extract_features(df)
        
        # Predict
        fraud_probability = float(self.model.predict(X, verbose=0)[0][0])
        
        return fraud_probability
    
    def make_decision(self, fraud_probability: float) -> str:
        """Make decision based on fraud probability"""
        
        reject_threshold = self.decision_config['reject_threshold']
        pend_threshold = self.decision_config['pend_threshold']
        
        if fraud_probability >= reject_threshold:
            return "REJECT"
        elif fraud_probability >= pend_threshold:
            return "PEND"
        else:
            return "ALLOW"
    
    def process_transaction(self, transaction: Dict) -> Dict:
        """Process transaction and return decision"""
        
        start_time = datetime.now()
        
        try:
            # Predict fraud probability
            fraud_probability = self.predict_fraud_probability(transaction)
            
            # Make decision
            decision = self.make_decision(fraud_probability)
            
            # Calculate latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'fraud_probability': round(fraud_probability, 4),
                'decision': decision,
                'latency_ms': round(latency_ms, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                f"Transaction {result['transaction_id']}: "
                f"{decision} (prob={fraud_probability:.4f}, latency={latency_ms:.2f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            return {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'decision': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


if __name__ == "__main__":
    # Test the decision engine
    engine = FraudDecisionEngine()
    
    # Sample transaction
    test_transaction = {
        'transaction_id': 'test_001',
        'user_id': 1234,
        'amount': 1500.00,
        'currency': 'USD',
        'merchant': 'Test Merchant',
        'timestamp': datetime.now().isoformat(),
        'location': 'US',
        'mcc': 5411,
        'credit_limit': 5000,
        'card_brand': 'Visa',
        'card_type': 'Credit',
        'use_chip': 'Swipe Transaction',
        'has_chip': 'YES',
        'gender': 'M',
        'card_on_dark_web': 'No'
    }
    
    result = engine.process_transaction(test_transaction)
    print(f"\n=== Decision Result ===")
    print(f"Decision: {result['decision']}")
    print(f"Fraud Probability: {result['fraud_probability']}")
    print(f"Latency: {result['latency_ms']}ms")
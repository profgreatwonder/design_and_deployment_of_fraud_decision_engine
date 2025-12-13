# """
# Real-time Fraud Decisioning Engine
# Returns: ALLOW, REJECT, or PEND decisions
# """

# import pandas as pd
# import numpy as np
# import yaml
# import pickle
# import logging

# from datetime import datetime
# from typing import Dict, Tuple

# import tensorflow as tf
# from tensorflow import keras

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class FraudDecisionEngine:
#     def __init__(self, config_path="/opt/airflow/config/config.yaml"):
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         self.model_config = self.config['model']
#         self.features_config = self.config['features']
#         self.decision_config = self.config['decision']
#         self.high_risk_mccs = self.config['high_risk_mccs']
        
#         self.load_artifacts()
    
#     def load_artifacts(self):
#         """Load model, scaler, and encoder"""
#         logger.info("Loading model artifacts...")
        
#         # Load model
#         self.model = keras.models.load_model(self.model_config['save_path'])
#         logger.info(f"Model loaded from {self.model_config['save_path']}")
        
#         # Load scaler
#         with open(self.model_config['scaler_path'], 'rb') as f:
#             self.scaler = pickle.load(f)
#         logger.info(f"Scaler loaded from {self.model_config['scaler_path']}")
        
#         # Load encoder
#         with open(self.model_config['encoder_path'], 'rb') as f:
#             self.encoder = pickle.load(f)
#         logger.info(f"Encoder loaded from {self.model_config['encoder_path']}")
    
#     def preprocess_transaction(self, transaction: Dict) -> pd.DataFrame:
#         """Preprocess a single transaction for prediction"""
        
#         # Create dataframe
#         df = pd.DataFrame([transaction])
        
#         # Extract datetime features if timestamp is present
#         if 'timestamp' in df.columns:
#             df['date'] = pd.to_datetime(df['timestamp'])
#             df['hour'] = df['date'].dt.hour
#             df['day_of_week'] = df['date'].dt.dayofweek
#             df['month'] = df['date'].dt.month
        
#         # Engineer features (simplified for real-time)
#         # In production, these would come from a feature store or cache
        
#         # Credit utilization
#         if 'credit_limit' in df.columns and 'amount' in df.columns:
#             df['credit_utilization'] = np.where(
#                 df['credit_limit'] != 0,
#                 df['amount'] / df['credit_limit'],
#                 0
#             )
#         else:
#             df['credit_utilization'] = 0
        
#         # High-risk MCC flag
#         if 'mcc' in df.columns:
#             df['is_high_risk_mcc'] = df['mcc'].isin(self.high_risk_mccs).astype(int)
#         else:
#             df['is_high_risk_mcc'] = 0
        
#         # Set default values for missing engineered features
#         # In production, these would be retrieved from feature store
#         default_features = {
#             'transactions_per_day_past': 0,
#             'transactions_per_week_past': 0,
#             'rolling_mean_3day': 0,
#             'rolling_std_3day': 0,
#             'age_at_acct_open': 0,
#             'card_age': 0
#         }
        
#         for feature, default_value in default_features.items():
#             if feature not in df.columns:
#                 df[feature] = default_value
        
#         # Fill missing values
#         numerical_features = self.features_config['numerical']
#         categorical_features = self.features_config['categorical']
        
#         for feature in numerical_features:
#             if feature not in df.columns:
#                 df[feature] = 0
        
#         for feature in categorical_features:
#             if feature not in df.columns:
#                 df[feature] = 'Unknown'
        
#         return df
    
#     def extract_features(self, df: pd.DataFrame) -> np.ndarray:
#         """Extract and process features for prediction"""
        
#         numerical_features = self.features_config['numerical']
#         categorical_features = self.features_config['categorical']
        
#         # Ensure all required features exist
#         for feature in numerical_features + categorical_features:
#             if feature not in df.columns:
#                 if feature in numerical_features:
#                     df[feature] = 0
#                 else:
#                     df[feature] = 'Unknown'
        
#         # Encode categorical features
#         X_encoded = self.encoder.transform(df[categorical_features])
#         X_encoded_df = pd.DataFrame(
#             X_encoded,
#             columns=self.encoder.get_feature_names_out(categorical_features),
#             index=df.index
#         )
        
#         # Combine with numerical features
#         X = pd.concat([df[numerical_features], X_encoded_df], axis=1)
        
#         # Scale numerical features
#         X[numerical_features] = self.scaler.transform(X[numerical_features])
        
#         # Reshape for GRU (samples, timesteps, features)
#         n_samples, n_features = X.shape
#         X_reshaped = X.values.reshape(n_samples, 1, n_features)
        
#         return X_reshaped
    
#     def predict_fraud_probability(self, transaction: Dict) -> float:
#         """Predict fraud probability for a transaction"""
        
#         # Preprocess
#         df = self.preprocess_transaction(transaction)
        
#         # Extract features
#         X = self.extract_features(df)
        
#         # Predict
#         fraud_probability = float(self.model.predict(X, verbose=0)[0][0])
        
#         return fraud_probability
    
#     def make_decision(self, fraud_probability: float) -> str:
#         """Make decision based on fraud probability"""
        
#         reject_threshold = self.decision_config['reject_threshold']
#         pend_threshold = self.decision_config['pend_threshold']
        
#         if fraud_probability >= reject_threshold:
#             return "REJECT"
#         elif fraud_probability >= pend_threshold:
#             return "PEND"
#         else:
#             return "ALLOW"
    
#     def process_transaction(self, transaction: Dict) -> Dict:
#         """Process transaction and return decision"""
        
#         start_time = datetime.now()
        
#         try:
#             # Predict fraud probability
#             fraud_probability = self.predict_fraud_probability(transaction)
            
#             # Make decision
#             decision = self.make_decision(fraud_probability)
            
#             # Calculate latency
#             latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
#             result = {
#                 'transaction_id': transaction.get('transaction_id', 'unknown'),
#                 'fraud_probability': round(fraud_probability, 4),
#                 'decision': decision,
#                 'latency_ms': round(latency_ms, 2),
#                 'timestamp': datetime.now().isoformat()
#             }
            
#             logger.info(
#                 f"Transaction {result['transaction_id']}: "
#                 f"{decision} (prob={fraud_probability:.4f}, latency={latency_ms:.2f}ms)"
#             )
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Error processing transaction: {str(e)}")
#             return {
#                 'transaction_id': transaction.get('transaction_id', 'unknown'),
#                 'decision': 'ERROR',
#                 'error': str(e),
#                 'timestamp': datetime.now().isoformat()
#             }


# if __name__ == "__main__":
#     # Test the decision engine
#     engine = FraudDecisionEngine()
    
#     # Sample transaction
#     test_transaction = {
#         'transaction_id': 'test_001',
#         'user_id': 1234,
#         'amount': 1500.00,
#         'currency': 'USD',
#         'merchant': 'Test Merchant',
#         'timestamp': datetime.now().isoformat(),
#         'location': 'US',
#         'mcc': 5411,
#         'credit_limit': 5000,
#         'card_brand': 'Visa',
#         'card_type': 'Credit',
#         'use_chip': 'Swipe Transaction',
#         'has_chip': 'YES',
#         'gender': 'M',
#         'card_on_dark_web': 'No'
#     }
    
#     result = engine.process_transaction(test_transaction)
#     print(f"\n=== Decision Result ===")
#     print(f"Decision: {result['decision']}")
#     print(f"Fraud Probability: {result['fraud_probability']}")
#     print(f"Latency: {result['latency_ms']}ms")



# """
# Decision Engine for Fraud Detection
# Updated to support Isolation Forest model
# """

# import pickle
# import numpy as np
# import pandas as pd
# import yaml
# import logging
# from pathlib import Path

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class DecisionEngine:
#     def __init__(self, config_path="/app/src/config/config.yaml"):
#         """Initialize decision engine with configuration"""
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         self.model_config = self.config['model']
#         self.features_config = self.config['features']
        
#         # Load model artifacts
#         self.load_model_artifacts()
    
#     def load_model_artifacts(self):
#         """Load trained model and preprocessing objects"""
#         logger.info("Loading model artifacts...")
        
#         try:
#             # Load model (Isolation Forest - .pkl file)
#             model_path = self.model_config['save_path'].replace('.h5', '.pkl')
#             with open(model_path, 'rb') as f:
#                 self.model = pickle.load(f)
#             logger.info(f"✅ Model loaded from {model_path}")
            
#             # Load scaler
#             scaler_path = self.model_config['scaler_path']
#             with open(scaler_path, 'rb') as f:
#                 self.scaler = pickle.load(f)
#             logger.info(f"✅ Scaler loaded from {scaler_path}")
            
#             # No encoder needed for Isolation Forest (only numerical features)
#             self.encoder = None
            
#             logger.info("✅ All model artifacts loaded successfully")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to load model artifacts: {e}")
#             raise
    
#     def preprocess_transaction(self, transaction_data):
#         """Preprocess a single transaction for prediction"""
#         # Extract only numerical features
#         numerical_features = self.features_config['numerical']
        
#         # Create feature dict
#         features = {}
#         for feature in numerical_features:
#             features[feature] = transaction_data.get(feature, 0)
        
#         # Convert to DataFrame
#         df = pd.DataFrame([features])
        
#         # Handle missing values
#         df = df.fillna(0)
        
#         # Scale features
#         df_scaled = pd.DataFrame(
#             self.scaler.transform(df),
#             columns=df.columns
#         )
        
#         return df_scaled
    
#     def predict(self, transaction_data):
#         """
#         Predict if transaction is fraudulent
        
#         Returns:
#             dict: {
#                 'is_fraud': bool,
#                 'fraud_score': float,
#                 'risk_level': str
#             }
#         """
#         try:
#             # Preprocess
#             X = self.preprocess_transaction(transaction_data)
            
#             # Predict
#             prediction = self.model.predict(X)[0]  # -1 for anomaly, 1 for normal
#             anomaly_score = self.model.decision_function(X)[0]
            
#             # Convert to fraud prediction
#             is_fraud = (prediction == -1)
            
#             # Convert score to 0-1 range (lower score = more anomalous)
#             # Normalize anomaly score to fraud probability
#             fraud_score = float(1 / (1 + np.exp(anomaly_score)))  # Sigmoid-like transformation
            
#             # Determine risk level
#             if fraud_score > 0.7:
#                 risk_level = 'HIGH'
#             elif fraud_score > 0.5:
#                 risk_level = 'MEDIUM'
#             else:
#                 risk_level = 'LOW'
            
#             return {
#                 'is_fraud': bool(is_fraud),
#                 'fraud_score': fraud_score,
#                 'risk_level': risk_level,
#                 'raw_anomaly_score': float(anomaly_score)
#             }
            
#         except Exception as e:
#             logger.error(f"Prediction error: {e}")
#             raise
    
#     def predict_batch(self, transactions):
#         """Predict for multiple transactions"""
#         results = []
#         for txn in transactions:
#             try:
#                 result = self.predict(txn)
#                 results.append(result)
#             except Exception as e:
#                 logger.error(f"Error processing transaction: {e}")
#                 results.append({
#                     'is_fraud': False,
#                     'fraud_score': 0.0,
#                     'risk_level': 'UNKNOWN',
#                     'error': str(e)
#                 })
        
#         return results


# # Test function
# if __name__ == "__main__":
#     # Test with sample transaction
#     sample_transaction = {
#         'amount': 150.0,
#         'credit_limit': 5000.0,
#         'credit_utilization': 0.03,
#         'rolling_mean_3day': 100.0,
#         'rolling_std_3day': 50.0,
#         'transactions_per_day_past': 2,
#         'transactions_per_week_past': 10,
#         'age_at_acct_open': 35,
#         'card_age': 3.5,
#         'hour': 14,
#         'day_of_week': 3,
#         'month': 6,
#         'is_high_risk_mcc': 0
#     }
    
#     engine = DecisionEngine()
#     result = engine.predict(sample_transaction)
    
#     print("\n=== Prediction Result ===")
#     print(f"Is Fraud: {result['is_fraud']}")
#     print(f"Fraud Score: {result['fraud_score']:.4f}")
#     print(f"Risk Level: {result['risk_level']}")
#     print(f"Raw Anomaly Score: {result['raw_anomaly_score']:.4f}")




"""
Decision Engine for Fraud Detection
Updated to support Isolation Forest model with correct Consumer interface
"""

import pickle
import numpy as np
import pandas as pd
import yaml
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDecisionEngine:
    def __init__(self, config_path=None):
        """Initialize decision engine with configuration"""
        if config_path is None:
             config_path = "/app/src/config/config.yaml"
             
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.features_config = self.config['features']
        self.decision_config = self.config['decision']
        
        # Load model artifacts
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load trained model and preprocessing objects"""
        logger.info("Loading model artifacts...")
        
        try:
            # Load model (Isolation Forest - .pkl file)
            # Ensure we look for .pkl even if config says .h5 momentarily
            model_path = self.model_config['save_path'].replace('.h5', '.pkl')
            
            if not Path(model_path).exists():
                logger.warning(f"Model file not found at {model_path}. Please run training pipeline.")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded from {model_path}")
            
            # Load scaler
            scaler_path = self.model_config['scaler_path']
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"✅ Scaler loaded from {scaler_path}")
            
            logger.info("✅ All model artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model artifacts: {e}")
            raise
    
    def preprocess_transaction(self, transaction_data):
        """Preprocess a single transaction for prediction"""
        # Extract only numerical features defined in config
        numerical_features = self.features_config['numerical']
        
        # Create feature dict, defaulting missing keys to 0
        features = {}
        for feature in numerical_features:
            features[feature] = transaction_data.get(feature, 0)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Handle missing values (simple fill with 0 for inference)
        df = df.fillna(0)
        
        # Scale features using the loaded scaler
        try:
            df_scaled = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns
            )
            return df_scaled
        except Exception as e:
            logger.error(f"Scaling error: {e}")
            # Fallback to unscaled if scaler fails (not ideal but prevents crash)
            return df
    
    def predict(self, transaction_data):
        """
        Internal prediction logic
        Returns dict with score and risk level
        """
        try:
            # Preprocess
            X = self.preprocess_transaction(transaction_data)
            
            # Predict
            # Isolation Forest returns -1 for anomaly, 1 for normal
            prediction = self.model.predict(X)[0] 
            
            # decision_function returns anomaly score (lower is more anomalous)
            raw_anomaly_score = self.model.decision_function(X)[0]
            
            # Convert to a probability-like fraud score (0 to 1)
            # We invert logic: lower raw score -> higher fraud probability
            fraud_score = float(1 / (1 + np.exp(raw_anomaly_score)))
            
            is_anomaly = (prediction == -1)
            
            return {
                'is_fraud': bool(is_anomaly),
                'fraud_score': fraud_score,
                'raw_anomaly_score': float(raw_anomaly_score)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Default safe response
            return {
                'is_fraud': False,
                'fraud_score': 0.0,
                'raw_anomaly_score': 0.0
            }

    def process_transaction(self, transaction):
        """
        Main interface called by the Consumer.
        Wraps predict() and applies decision logic (ALLOW/PEND/REJECT).
        """
        start_time = time.time()
        
        # Get model prediction
        result = self.predict(transaction)
        fraud_prob = result['fraud_score']
        
        # Apply Decision Logic based on Config Thresholds
        reject_thr = self.decision_config.get('reject_threshold', 0.75)
        pend_thr = self.decision_config.get('pend_threshold', 0.45)
        
        if fraud_prob >= reject_thr:
            decision = "REJECT"
        elif fraud_prob >= pend_thr:
            decision = "PEND"
        else:
            decision = "ALLOW"
            
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'transaction_id': transaction.get('transaction_id'),
            'fraud_probability': fraud_prob,
            'decision': decision,
            'latency_ms': latency_ms,
            'raw_score': result['raw_anomaly_score']
        }
# # """
# # GRU Model Training Script for Fraud Detection
# # """

# # import pandas as pd
# # import numpy as np
# # import yaml
# # import pickle
# # import logging

# # import tensorflow as tf
# # from tensorflow import keras
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import GRU, Dense, Dropout

# # from pathlib import Path
# # from imblearn.over_sampling import SMOTE

# # import mlflow
# # import mlflow.keras

# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# # from data_preprocessor import DataPreprocessor

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)


# # class FraudDetectionTrainer:
# #     def __init__(self, config_path="/opt/airflow/config/config.yaml"):
# #         with open(config_path, 'r') as f:
# #             self.config = yaml.safe_load(f)
        
# #         self.model_config = self.config['model']
# #         self.features_config = self.config['features']
        
# #         # Set MLflow tracking
# #         mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
# #         mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
# #     def prepare_features(self, df):
# #         """Prepare features and target"""
# #         logger.info("Preparing features...")
        
# #         numerical_features = self.features_config['numerical']
# #         categorical_features = self.features_config['categorical']
# #         features = numerical_features + categorical_features
        
# #         X = df[features].copy()
# #         y = df['is_fraud'].copy()
        
# #         logger.info(f"Feature matrix shape: {X.shape}")
# #         logger.info(f"Target shape: {y.shape}")
# #         logger.info(f"Fraud rate: {y.mean():.4f}")
        
# #         return X, y, numerical_features, categorical_features
    
# #     def encode_and_scale(self, X, numerical_features, categorical_features, fit=True):
# #         """Encode categorical and scale numerical features"""
# #         logger.info("Encoding and scaling features...")
        
# #         if fit:
# #             # Fit encoder
# #             self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# #             X_encoded = self.encoder.fit_transform(X[categorical_features])
            
# #             # Save encoder
# #             with open(self.model_config['encoder_path'], 'wb') as f:
# #                 pickle.dump(self.encoder, f)
# #             logger.info(f"Encoder saved to {self.model_config['encoder_path']}")
# #         else:
# #             X_encoded = self.encoder.transform(X[categorical_features])
        
# #         # Create encoded dataframe
# #         X_encoded_df = pd.DataFrame(
# #             X_encoded, 
# #             columns=self.encoder.get_feature_names_out(categorical_features),
# #             index=X.index
# #         )
        
# #         # Combine with numerical features
# #         X = pd.concat([X.drop(columns=categorical_features), X_encoded_df], axis=1)
        
# #         if fit:
# #             # Fit scaler
# #             self.scaler = StandardScaler()
# #             X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
            
# #             # Save scaler
# #             with open(self.model_config['scaler_path'], 'wb') as f:
# #                 pickle.dump(self.scaler, f)
# #             logger.info(f"Scaler saved to {self.model_config['scaler_path']}")
# #         else:
# #             X[numerical_features] = self.scaler.transform(X[numerical_features])
        
# #         logger.info(f"Final feature matrix shape after encoding: {X.shape}")
        
# #         return X
    
# #     def handle_imbalance(self, X, y):
# #         """Apply SMOTE to handle class imbalance"""
# #         logger.info("Handling class imbalance with SMOTE...")
# #         logger.info(f"Before SMOTE - Class distribution:\n{y.value_counts()}")
        
# #         smote = SMOTE(
# #             random_state=self.model_config['random_state'],
# #             k_neighbors=self.model_config['smote_k_neighbors']
# #         )
        
# #         X_resampled, y_resampled = smote.fit_resample(X, y)
        
# #         logger.info(f"After SMOTE - Class distribution:\n{pd.Series(y_resampled).value_counts()}")
# #         logger.info(f"Resampled data shape: {X_resampled.shape}")
        
# #         return X_resampled, y_resampled
    
# #     def build_model(self, input_shape):
# #         """Build GRU model"""
# #         logger.info("Building GRU model...")
        
# #         model = Sequential([
# #             GRU(
# #                 units=self.model_config['gru_units'],
# #                 activation='relu',
# #                 input_shape=input_shape
# #             ),
# #             Dropout(self.model_config['dropout_rate']),
# #             Dense(1, activation='sigmoid')
# #         ])
        
# #         model.compile(
# #             optimizer='adam',
# #             loss='binary_crossentropy',
# #             metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
# #         )
        
# #         logger.info("Model built successfully")
# #         model.summary(print_fn=logger.info)
        
# #         return model
    
# #     def train(self, X_train, y_train):
# #         """Train the model"""
# #         logger.info("Training model...")
        
# #         # Reshape for GRU (samples, timesteps, features)
# #         n_samples, n_features = X_train.shape
# #         X_train_reshaped = X_train.values.reshape(n_samples, 1, n_features)
        
# #         # Build model
# #         input_shape = (1, n_features)
# #         self.model = self.build_model(input_shape)
        
# #         # Train
# #         history = self.model.fit(
# #             X_train_reshaped,
# #             y_train,
# #             epochs=self.model_config['epochs'],
# #             batch_size=self.model_config['batch_size'],
# #             validation_split=self.model_config['validation_split'],
# #             verbose=1
# #         )
        
# #         logger.info("Training completed")
        
# #         return history
    
# #     def evaluate(self, X_test, y_test):
# #         """Evaluate the model"""
# #         logger.info("Evaluating model...")
        
# #         # Reshape for GRU
# #         n_samples, n_features = X_test.shape
# #         X_test_reshaped = X_test.values.reshape(n_samples, 1, n_features)
        
# #         # Predictions
# #         y_pred_proba = self.model.predict(X_test_reshaped)
# #         y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
# #         # Metrics
# #         loss, accuracy, recall, precision = self.model.evaluate(X_test_reshaped, y_test, verbose=0)
# #         auc_roc = roc_auc_score(y_test, y_pred_proba)
        
# #         logger.info(f"Test Loss: {loss:.4f}")
# #         logger.info(f"Test Accuracy: {accuracy:.4f}")
# #         logger.info(f"Test Recall: {recall:.4f}")
# #         logger.info(f"Test Precision: {precision:.4f}")
# #         logger.info(f"AUC-ROC: {auc_roc:.4f}")
        
# #         # Classification report
# #         logger.info("\nClassification Report:")
# #         logger.info("\n" + classification_report(y_test, y_pred_binary))
        
# #         # Confusion matrix
# #         cm = confusion_matrix(y_test, y_pred_binary)
# #         logger.info(f"\nConfusion Matrix:\n{cm}")
        
# #         return {
# #             'loss': loss,
# #             'accuracy': accuracy,
# #             'recall': recall,
# #             'precision': precision,
# #             'auc_roc': auc_roc
# #         }
    
# #     def save_model(self):
# #         """Save trained model"""
# #         self.model.save(self.model_config['save_path'])
# #         logger.info(f"Model saved to {self.model_config['save_path']}")
    
# #     def run_training_pipeline(self):
# #         """Run complete training pipeline"""
# #         with mlflow.start_run():
# #             # Load and preprocess data
# #             preprocessor = DataPreprocessor()
# #             df = preprocessor.process_all()
            
# #             # Prepare features
# #             X, y, numerical_features, categorical_features = self.prepare_features(df)
            
# #             # Encode and scale
# #             X_processed = self.encode_and_scale(X, numerical_features, categorical_features, fit=True)
            
# #             # Handle imbalance
# #             X_resampled, y_resampled = self.handle_imbalance(X_processed, y)
            
# #             # Train-test split
# #             X_train, X_test, y_train, y_test = train_test_split(
# #                 X_resampled, y_resampled,
# #                 test_size=self.model_config['test_size'],
# #                 random_state=self.model_config['random_state']
# #             )
            
# #             logger.info(f"Training set size: {X_train.shape}")
# #             logger.info(f"Test set size: {X_test.shape}")
            
# #             # Log parameters
# #             mlflow.log_params(self.model_config)
            
# #             # Train model
# #             history = self.train(X_train, y_train)
            
# #             # Evaluate
# #             metrics = self.evaluate(X_test, y_test)
            
# #             # Log metrics
# #             mlflow.log_metrics(metrics)
            
# #             # Save model
# #             self.save_model()
# #             mlflow.keras.log_model(self.model, "model")
            
# #             logger.info("\n=== Training pipeline completed successfully ===")
            
# #             return self.model, metrics


# # if __name__ == "__main__":
# #     trainer = FraudDetectionTrainer()
# #     model, metrics = trainer.run_training_pipeline()
    
# #     print("\n=== Final Metrics ===")
# #     for key, value in metrics.items():
# #         print(f"{key}: {value:.4f}")




# """
# Isolation Forest Model Training Script for Fraud Detection
# Simpler alternative to GRU - no deep learning required
# """

# import pandas as pd
# import numpy as np
# import yaml
# import pickle
# import logging
# from pathlib import Path

# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# import mlflow
# import mlflow.sklearn

# from data_preprocessor import DataPreprocessor

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class IsolationForestTrainer:
#     def __init__(self, config_path="/opt/airflow/config/config.yaml"):
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         self.model_config = self.config['model']
#         self.features_config = self.config['features']
        
#         # Set MLflow tracking
#         mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
#         mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
#     def prepare_features(self, df):
#         """Prepare numerical features for Isolation Forest"""
#         logger.info("Preparing features for Isolation Forest...")
        
#         # Only use numerical features for Isolation Forest
#         numerical_features = self.features_config['numerical']
        
#         X = df[numerical_features].copy()
#         y = df['is_fraud'].copy() if 'is_fraud' in df.columns else None
        
#         # Handle missing values
#         X = X.fillna(0)
        
#         logger.info(f"Feature matrix shape: {X.shape}")
#         if y is not None:
#             logger.info(f"Target shape: {y.shape}")
#             logger.info(f"Fraud rate: {y.mean():.4f}")
        
#         return X, y, numerical_features
    
#     def scale_features(self, X, fit=True):
#         """Scale numerical features"""
#         logger.info("Scaling features...")
        
#         if fit:
#             self.scaler = StandardScaler()
#             X_scaled = self.scaler.fit_transform(X)
            
#             # Save scaler
#             scaler_path = self.model_config['scaler_path']
#             with open(scaler_path, 'wb') as f:
#                 pickle.dump(self.scaler, f)
#             logger.info(f"Scaler saved to {scaler_path}")
#         else:
#             X_scaled = self.scaler.transform(X)
        
#         return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
#     def train(self, X_train):
#         """Train Isolation Forest model"""
#         logger.info("Training Isolation Forest model...")
        
#         # Create model
#         self.model = IsolationForest(
#             n_estimators=100,
#             contamination=0.01,  # Expect 1% anomalies
#             random_state=42,
#             n_jobs=-1,
#             verbose=1
#         )
        
#         # Fit model
#         self.model.fit(X_train)
        
#         logger.info("Training completed")
        
#         return self.model
    
#     def evaluate(self, X_test, y_test=None):
#         """Evaluate the model"""
#         logger.info("Evaluating model...")
        
#         # Predict (-1 for anomalies, 1 for normal)
#         y_pred_labels = self.model.predict(X_test)
        
#         # Get anomaly scores
#         anomaly_scores = self.model.decision_function(X_test)
        
#         # Convert to binary (0 = normal, 1 = fraud)
#         y_pred_binary = (y_pred_labels == -1).astype(int)
        
#         metrics = {
#             'anomaly_count': y_pred_binary.sum(),
#             'anomaly_rate': y_pred_binary.mean()
#         }
        
#         logger.info(f"Anomalies detected: {metrics['anomaly_count']}")
#         logger.info(f"Anomaly rate: {metrics['anomaly_rate']:.4f}")
        
#         # If we have labels, calculate performance metrics
#         if y_test is not None:
#             accuracy = (y_pred_binary == y_test).mean()
#             auc_roc = roc_auc_score(y_test, -anomaly_scores)  # Negative because lower scores = anomalies
            
#             metrics['accuracy'] = accuracy
#             metrics['auc_roc'] = auc_roc
            
#             logger.info(f"Accuracy: {accuracy:.4f}")
#             logger.info(f"AUC-ROC: {auc_roc:.4f}")
            
#             logger.info("\nClassification Report:")
#             logger.info("\n" + classification_report(y_test, y_pred_binary))
            
#             cm = confusion_matrix(y_test, y_pred_binary)
#             logger.info(f"\nConfusion Matrix:\n{cm}")
        
#         return metrics
    
#     def save_model(self):
#         """Save trained model"""
#         model_path = self.model_config['save_path']
        
#         # Change extension to .pkl for sklearn model
#         model_path = model_path.replace('.h5', '.pkl')
        
#         with open(model_path, 'wb') as f:
#             pickle.dump(self.model, f)
        
#         logger.info(f"Model saved to {model_path}")
        
#         return model_path
    
#     def run_training_pipeline(self):
#         """Run complete training pipeline"""
#         with mlflow.start_run():
#             # Load and preprocess data
#             preprocessor = DataPreprocessor()
#             df = preprocessor.process_all()
            
#             # Prepare features
#             X, y, numerical_features = self.prepare_features(df)
            
#             # Scale features
#             X_scaled = self.scale_features(X, fit=True)
            
#             # Split data (80/20)
#             split_idx = int(len(X_scaled) * 0.8)
#             X_train = X_scaled.iloc[:split_idx]
#             X_test = X_scaled.iloc[split_idx:]
            
#             if y is not None:
#                 y_train = y.iloc[:split_idx]
#                 y_test = y.iloc[split_idx:]
#             else:
#                 y_train = None
#                 y_test = None
            
#             logger.info(f"Training set size: {X_train.shape}")
#             logger.info(f"Test set size: {X_test.shape}")
            
#             # Log parameters
#             mlflow.log_params({
#                 'model_type': 'IsolationForest',
#                 'n_estimators': 100,
#                 'contamination': 0.01,
#                 'features_count': len(numerical_features)
#             })
            
#             # Train model
#             self.train(X_train)
            
#             # Evaluate
#             metrics = self.evaluate(X_test, y_test)
            
#             # Log metrics
#             mlflow.log_metrics(metrics)
            
#             # Save model
#             model_path = self.save_model()
#             mlflow.sklearn.log_model(self.model, "model")
            
#             logger.info("\n=== Training pipeline completed successfully ===")
#             logger.info(f"Model saved to: {model_path}")
            
#             return self.model, metrics


# if __name__ == "__main__":
#     trainer = IsolationForestTrainer()
#     model, metrics = trainer.run_training_pipeline()
    
#     print("\n=== Final Metrics ===")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")



# works


# """
# Isolation Forest Model Training Script
# Trains the model and saves it as a .pkl file.
# """

# import pandas as pd
# import numpy as np
# import yaml
# import pickle
# import logging
# import os
# import sys
# from pathlib import Path

# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report

# # Add the project root to path so we can import modules
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # Handle different import locations for Airflow vs Local
# try:
#     from src.dags.ml_training.data_preprocessor import DataPreprocessor
# except ImportError:
#     try:
#         from dags.ml_training.data_preprocessor import DataPreprocessor
#     except ImportError:
#         # Fallback if running directly from the folder
#         sys.path.append(os.path.join(os.path.dirname(__file__)))
#         from data_preprocessor import DataPreprocessor

# import mlflow
# import mlflow.sklearn

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class IsolationForestTrainer:
#     def __init__(self, config_path=None):
#         self.paths_tried = []
        
#         # Locate config file if not provided
#         if config_path is None:
#             # Default locations to try
#             possible_paths = [
#                 "/app/src/config/config.yaml",  # Docker absolute path
#                 "src/config/config.yaml",       # From project root
#                 "config/config.yaml",           # From src folder
#                 "../config/config.yaml",        # Relative
#                 "../../config/config.yaml"      # Relative
#             ]
            
#             for p in possible_paths:
#                 self.paths_tried.append(p)
#                 if os.path.exists(p):
#                     config_path = p
#                     break
        
#         # Validate config path exists
#         if not config_path or not os.path.exists(config_path):
#             error_msg = f"Config file not found: {config_path}. Tried: {self.paths_tried}"
#             logger.error(error_msg)
#             raise FileNotFoundError(error_msg)

#         # FIX: Save the resolved config path so it can be passed to other classes
#         self.config_path = os.path.abspath(config_path)

#         logger.info(f"Loading config from: {self.config_path}")
#         with open(self.config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         self.model_config = self.config['model']
#         self.features_config = self.config['features']
        
#         # Set MLflow tracking (Optional)
#         self.use_mlflow = False
#         try:
#             # Only try MLflow if we can resolve the host (avoids long hang on local run)
#             tracking_uri = self.config['mlflow']['tracking_uri']
#             mlflow.set_tracking_uri(tracking_uri)
#             mlflow.set_experiment(self.config['mlflow']['experiment_name'])
#             self.use_mlflow = True
#         except Exception as e:
#             logger.warning(f"MLflow setup failed ({str(e)}). Proceeding without experiment tracking.")
#             self.use_mlflow = False
    
#     def prepare_features(self, df):
#         """Prepare numerical features for Isolation Forest"""
#         logger.info("Preparing features for Isolation Forest...")
        
#         numerical_features = self.features_config['numerical']
        
#         # Ensure all features exist in DF
#         missing_cols = [col for col in numerical_features if col not in df.columns]
#         if missing_cols:
#             logger.warning(f"Missing columns in dataset: {missing_cols}")
#             for col in missing_cols:
#                 df[col] = 0
                
#         X = df[numerical_features].copy()
#         X = X.fillna(0)
        
#         y = df['is_fraud'].copy() if 'is_fraud' in df.columns else None
        
#         logger.info(f"Feature matrix shape: {X.shape}")
#         return X, y, numerical_features
    
#     def scale_features(self, X, fit=True):
#         """Scale numerical features"""
#         logger.info("Scaling features...")
        
#         if fit:
#             self.scaler = StandardScaler()
#             X_scaled = self.scaler.fit_transform(X)
            
#             # Save scaler
#             scaler_path = self.model_config['scaler_path']
#             # Ensure path is absolute or correct relative to execution
#             if not os.path.isabs(scaler_path) and os.getenv('AIRFLOW_HOME'):
#                 scaler_path = os.path.join(os.getenv('AIRFLOW_HOME'), scaler_path)
            
#             os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            
#             with open(scaler_path, 'wb') as f:
#                 pickle.dump(self.scaler, f)
#             logger.info(f"Scaler saved to {scaler_path}")
#         else:
#             X_scaled = self.scaler.transform(X)
        
#         return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
#     def train(self, X_train):
#         """Train Isolation Forest model"""
#         logger.info("Training Isolation Forest model...")
        
#         self.model = IsolationForest(
#             n_estimators=self.model_config.get('n_estimators', 100),
#             contamination=self.model_config.get('contamination', 0.01),
#             random_state=self.model_config.get('random_state', 42),
#             n_jobs=self.model_config.get('n_jobs', -1),
#             verbose=1
#         )
        
#         self.model.fit(X_train)
#         logger.info("Training completed")
#         return self.model
    
#     def save_model(self):
#         """Save trained model to disk"""
#         save_path = self.model_config['save_path']
        
#         # Handle file extension
#         if save_path.endswith('.h5'):
#             save_path = save_path.replace('.h5', '.pkl')
            
#         # Ensure path is handled correctly in Airflow vs Local
#         if not os.path.isabs(save_path) and os.getenv('AIRFLOW_HOME'):
#             save_path = os.path.join(os.getenv('AIRFLOW_HOME'), save_path)
            
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
#         with open(save_path, 'wb') as f:
#             pickle.dump(self.model, f)
        
#         logger.info(f"✅ Model saved to {save_path}")
#         return save_path

#     def run_training_pipeline(self):
#         """Run complete training pipeline"""
#         if self.use_mlflow:
#             try:
#                 mlflow.start_run()
#             except Exception:
#                 logger.warning("Could not start MLflow run. Continuing locally.")
#                 self.use_mlflow = False
            
#         try:
#             # 1. Load and preprocess data
#             # Use self.config_path which is now guaranteed to be set
#             preprocessor = DataPreprocessor(config_path=self.config_path)
#             df = preprocessor.process_all()
            
#             # 2. Prepare features
#             X, y, numerical_features = self.prepare_features(df)
            
#             # 3. Scale features
#             X_scaled = self.scale_features(X, fit=True)
            
#             # 4. Train model
#             self.train(X_scaled)
            
#             # 5. Evaluate
#             if y is not None:
#                 preds = self.model.predict(X_scaled)
#                 preds_binary = (preds == -1).astype(int)
#                 logger.info("\nClassification Report:")
#                 logger.info(classification_report(y, preds_binary))
            
#             # 6. Save model
#             self.save_model()
            
#             if self.use_mlflow:
#                 mlflow.sklearn.log_model(self.model, "model")
#                 mlflow.end_run()
                
#             return self.model, {}
                
#         except Exception as e:
#             logger.error(f"Training failed: {e}")
#             if self.use_mlflow:
#                 mlflow.end_run()
#             raise

# if __name__ == "__main__":
#     # Robust Config Detection
#     cfg_path = None
    
#     # Priority 1: Check standard locations relative to current working directory
#     potential_configs = [
#         "src/config/config.yaml",
#         "config/config.yaml",
#         "../config/config.yaml",
#         "../../config/config.yaml",
#         "/app/src/config/config.yaml"
#     ]
    
#     print(f"Current working directory: {os.getcwd()}")
    
#     for path in potential_configs:
#         if os.path.exists(path):
#             cfg_path = os.path.abspath(path)
#             print(f"Found config at: {cfg_path}")
#             break
            
#     if cfg_path is None:
#         print("❌ Could not find config.yaml in standard locations.")
#         # Fallback to try and find it based on script location
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
#         fallback_path = os.path.join(project_root, 'src', 'config', 'config.yaml')
#         if os.path.exists(fallback_path):
#             cfg_path = fallback_path
#             print(f"Found config via script location: {cfg_path}")
    
#     # Run pipeline
#     trainer = IsolationForestTrainer(config_path=cfg_path)
#     trainer.run_training_pipeline()





# Claude
"""
Isolation Forest Model Training Script
Trains the model and saves it as a .pkl file.
FIXED VERSION - Better MLflow connectivity, error handling, and Airflow compatibility
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
import os
import sys
import socket
from pathlib import Path
from urllib.parse import urlparse

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Add the project root to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Handle different import locations for Airflow vs Local
try:
    from src.dags.ml_training.data_preprocessor import DataPreprocessor
except ImportError:
    try:
        from dags.ml_training.data_preprocessor import DataPreprocessor
    except ImportError:
        # Fallback if running directly from the folder
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from data_preprocessor import DataPreprocessor

import mlflow
import mlflow.sklearn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IsolationForestTrainer:
    def __init__(self, config_path=None):
        self.paths_tried = []
        self.use_mlflow = False
        self.model = None
        self.scaler = None
        
        # Locate config file if not provided
        if config_path is None:
            config_path = self._find_config()
        
        # Validate config path exists
        if not config_path or not os.path.exists(config_path):
            error_msg = f"Config file not found: {config_path}. Tried: {self.paths_tried}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Save the resolved config path
        self.config_path = os.path.abspath(config_path)
        logger.info(f"✅ Using config: {self.config_path}")

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.features_config = self.config['features']
        
        # Setup MLflow with better error handling
        self._setup_mlflow()
    
    def _find_config(self):
        """Find config file in standard locations"""
        possible_paths = [
            "/opt/airflow/config/config.yaml",  # Your Docker setup
            "/app/src/config/config.yaml",      # Alternative Docker
            "config/config.yaml",               # From AIRFLOW_HOME
            "src/config/config.yaml",           # From project root
            "../config/config.yaml",            # Relative up one
            "../../config/config.yaml",         # Relative up two
        ]
        
        # If in Airflow, also try AIRFLOW_HOME based path
        if os.getenv('AIRFLOW_HOME'):
            airflow_path = os.path.join(os.getenv('AIRFLOW_HOME'), 'config', 'config.yaml')
            possible_paths.insert(0, airflow_path)
        
        for path in possible_paths:
            self.paths_tried.append(path)
            if os.path.exists(path):
                return path
        
        return None
    
    def _setup_mlflow(self):
        """Setup MLflow with proper error handling and connectivity check"""
        try:
            mlflow_config = self.config.get('mlflow', {})
            tracking_uri = mlflow_config.get('tracking_uri')
            experiment_name = mlflow_config.get('experiment_name', 'fraud_detection')
            
            if not tracking_uri:
                logger.info("ℹ️  MLflow tracking URI not configured in config.yaml")
                self.use_mlflow = False
                return
            
            # FIX: Test if MLflow server is actually reachable (with timeout)
            logger.info(f"Testing MLflow connectivity to {tracking_uri}...")
            
            try:
                parsed = urlparse(tracking_uri)
                host = parsed.hostname or 'localhost'
                port = parsed.port or 5000
                
                # Quick connectivity check with 2 second timeout
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    # MLflow is reachable
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_experiment(experiment_name)
                    self.use_mlflow = True
                    logger.info(f"✅ MLflow connected: {tracking_uri}")
                else:
                    logger.warning(f"⚠️  MLflow server not reachable at {tracking_uri}:{port}")
                    logger.warning(f"⚠️  Training will continue without MLflow tracking")
                    self.use_mlflow = False
                    
            except Exception as conn_error:
                logger.warning(f"⚠️  MLflow connectivity test failed: {conn_error}")
                self.use_mlflow = False
                
        except Exception as e:
            logger.warning(f"⚠️  MLflow setup failed: {str(e)}")
            logger.warning(f"⚠️  Training will continue without experiment tracking")
            self.use_mlflow = False
    
    def prepare_features(self, df):
        """Prepare numerical features for Isolation Forest"""
        logger.info("Preparing features for Isolation Forest...")
        
        numerical_features = self.features_config['numerical']
        
        # Ensure all features exist in DF
        missing_cols = [col for col in numerical_features if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️  Missing columns: {missing_cols}. Creating with zeros.")
            for col in missing_cols:
                df[col] = 0
        
        # Create feature matrix
        X = df[numerical_features].copy()
        X = X.fillna(0)
        
        # Handle infinite values that might cause issues
        X = X.replace([np.inf, -np.inf], 0)
        
        y = df['is_fraud'].copy() if 'is_fraud' in df.columns else None
        
        logger.info(f"✅ Feature matrix shape: {X.shape}")
        if y is not None:
            logger.info(f"✅ Fraud rate in training data: {y.mean():.4f}")
        
        return X, y, numerical_features
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        logger.info("Scaling features...")
        
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Save scaler with proper path resolution
            scaler_path = self._resolve_path(self.model_config['scaler_path'])
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"✅ Scaler saved to {scaler_path}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted yet. Call scale_features with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def _resolve_path(self, path):
        """Resolve path for Airflow or local execution"""
        if os.path.isabs(path):
            return path
        
        # In Airflow environment
        airflow_home = os.getenv('AIRFLOW_HOME')
        if airflow_home:
            return os.path.join(airflow_home, path)
        
        # Local execution - relative to project root
        config_dir = os.path.dirname(self.config_path)
        project_root = os.path.abspath(os.path.join(config_dir, '..'))
        return os.path.join(project_root, path)
    
    def train(self, X_train):
        """Train Isolation Forest model"""
        logger.info("="*60)
        logger.info("TRAINING ISOLATION FOREST MODEL")
        logger.info("="*60)
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Parameters:")
        logger.info(f"  - n_estimators: {self.model_config.get('n_estimators', 100)}")
        logger.info(f"  - contamination: {self.model_config.get('contamination', 0.01)}")
        logger.info(f"  - random_state: {self.model_config.get('random_state', 42)}")
        logger.info(f"  - n_jobs: {self.model_config.get('n_jobs', -1)}")
        
        self.model = IsolationForest(
            n_estimators=self.model_config.get('n_estimators', 100),
            contamination=self.model_config.get('contamination', 0.01),
            random_state=self.model_config.get('random_state', 42),
            n_jobs=self.model_config.get('n_jobs', -1),
            verbose=1
        )
        
        self.model.fit(X_train)
        logger.info("✅ Training completed")
        logger.info("="*60)
        return self.model
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if y is None:
            logger.warning("⚠️  No labels available for evaluation")
            return {}
        
        logger.info("="*60)
        logger.info("EVALUATING MODEL")
        logger.info("="*60)
        
        preds = self.model.predict(X)
        preds_binary = (preds == -1).astype(int)
        
        report = classification_report(y, preds_binary, output_dict=True)
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y, preds_binary))
        logger.info("="*60)
        
        return report
    
    def save_model(self):
        """Save trained model to disk"""
        save_path = self.model_config['save_path']
        
        # Handle file extension
        if save_path.endswith('.h5'):
            save_path = save_path.replace('.h5', '.pkl')
        
        save_path = self._resolve_path(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"✅ Model saved to {save_path}")
        return save_path

    def run_training_pipeline(self):
        """Run complete training pipeline"""
        mlflow_run = None
        
        # Start MLflow run if available
        if self.use_mlflow:
            try:
                mlflow_run = mlflow.start_run()
                logger.info(f"✅ MLflow run started: {mlflow_run.info.run_id}")
            except Exception as e:
                logger.warning(f"⚠️  Could not start MLflow run: {e}")
                self.use_mlflow = False
        
        try:
            logger.info("\n" + "="*60)
            logger.info("STARTING TRAINING PIPELINE")
            logger.info("="*60)
            
            # 1. Load and preprocess data
            logger.info("\n[1/6] Loading and preprocessing data...")
            preprocessor = DataPreprocessor(config_path=self.config_path)
            df = preprocessor.process_all()
            
            # 2. Prepare features
            logger.info("\n[2/6] Preparing features...")
            X, y, numerical_features = self.prepare_features(df)
            
            # 3. Scale features
            logger.info("\n[3/6] Scaling features...")
            X_scaled = self.scale_features(X, fit=True)
            
            # 4. Train model
            logger.info("\n[4/6] Training model...")
            self.train(X_scaled)
            
            # 5. Evaluate
            logger.info("\n[5/6] Evaluating model...")
            report = self.evaluate(X_scaled, y)
            
            # 6. Save model
            logger.info("\n[6/6] Saving model...")
            model_path = self.save_model()
            
            # Log to MLflow if available
            if self.use_mlflow and report:
                try:
                    # Log parameters
                    mlflow.log_params(self.model_config)
                    
                    # Log metrics
                    if '1' in report:
                        mlflow.log_metrics({
                            'precision': report['1']['precision'],
                            'recall': report['1']['recall'],
                            'f1-score': report['1']['f1-score'],
                            'accuracy': report['accuracy']
                        })
                    
                    # Log model
                    mlflow.sklearn.log_model(self.model, "model")
                    logger.info("✅ Metrics logged to MLflow")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to log to MLflow: {e}")
            
            logger.info("\n" + "="*60)
            logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Model saved to: {model_path}")
            
            return self.model, report
            
        except Exception as e:
            logger.error(f"\n❌ TRAINING FAILED: {e}")
            logger.exception("Full traceback:")
            raise
            
        finally:
            # End MLflow run if it was started
            if self.use_mlflow and mlflow_run:
                try:
                    mlflow.end_run()
                    logger.info("MLflow run ended")
                except:
                    pass


def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("="*60)
    
    # Robust Config Detection
    cfg_path = None
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"AIRFLOW_HOME: {os.getenv('AIRFLOW_HOME', 'Not set')}")
    
    # Try to find config
    potential_configs = [
        "/opt/airflow/config/config.yaml",
        "config/config.yaml",
        "src/config/config.yaml",
        "../config/config.yaml",
        "../../config/config.yaml",
        "/app/src/config/config.yaml"
    ]
    
    for path in potential_configs:
        if os.path.exists(path):
            cfg_path = os.path.abspath(path)
            logger.info(f"✅ Found config at: {cfg_path}")
            break
    
    if cfg_path is None:
        logger.error("⚠️  Could not find config.yaml in standard locations.")
        # Fallback to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        fallback_path = os.path.join(project_root, 'src', 'config', 'config.yaml')
        if os.path.exists(fallback_path):
            cfg_path = fallback_path
            logger.info(f"✅ Found config via script location: {cfg_path}")
        else:
            logger.error(f"❌ Config not found at fallback: {fallback_path}")
            logger.error(f"Tried paths: {potential_configs + [fallback_path]}")
            sys.exit(1)
    
    # Run pipeline
    try:
        trainer = IsolationForestTrainer(config_path=cfg_path)
        model, metrics = trainer.run_training_pipeline()
        logger.info("✅ Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
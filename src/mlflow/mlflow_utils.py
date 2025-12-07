"""
MLflow utilities for fraud detection system
Provides easy integration with MLflow tracking server
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import json
from datetime import datetime


class FraudMLflowTracker:
    """
    MLflow tracker for fraud detection models
    Handles experiment tracking, model logging, and artifact management
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "fraud_detection",
        minio_endpoint: Optional[str] = None,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None
    ):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: MLflow tracking server URI (defaults to env var)
            experiment_name: Name of the experiment
            minio_endpoint: MinIO endpoint URL (defaults to env var)
            minio_access_key: MinIO access key (defaults to env var)
            minio_secret_key: MinIO secret key (defaults to env var)
        """
        # Set tracking URI
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Configure MinIO/S3 credentials
        os.environ['AWS_ACCESS_KEY_ID'] = minio_access_key or os.getenv('MINIO_ROOT_USER', 'minioadmin')
        os.environ['AWS_SECRET_ACCESS_KEY'] = minio_secret_key or os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = minio_endpoint or os.getenv('MINIO_ENDPOINT_URL', 'http://localhost:9000')
        
        # Set or create experiment
        self.experiment_name = experiment_name
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")
            self.experiment_id = None
        
        self.client = MlflowClient()
        self.current_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags
        """
        if self.current_run is not None:
            print("Warning: A run is already active. Ending it first.")
            self.end_run()
        
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        return self.current_run
    
    def end_run(self):
        """End the current MLflow run"""
        if self.current_run is not None:
            mlflow.end_run()
            self.current_run = None
    
    def log_model_params(self, params: Dict[str, Any]):
        """
        Log model parameters
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_model_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log model metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number for tracking metric evolution
        """
        for key, value in metrics.items():
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
    
    def log_fraud_detection_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        auc_roc: float,
        false_positive_rate: float,
        false_negative_rate: float,
        step: Optional[int] = None
    ):
        """
        Log standard fraud detection metrics
        
        Args:
            All fraud detection relevant metrics
            step: Optional step number
        """
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_roc': auc_roc,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }
        self.log_model_metrics(metrics, step)
    
    def log_confusion_matrix(self, confusion_matrix: Any, artifact_name: str = "confusion_matrix.json"):
        """
        Log confusion matrix as artifact
        
        Args:
            confusion_matrix: Confusion matrix (numpy array or dict)
            artifact_name: Name for the artifact file
        """
        import numpy as np
        
        # Convert to dict if numpy array
        if isinstance(confusion_matrix, np.ndarray):
            cm_dict = {
                'matrix': confusion_matrix.tolist(),
                'shape': confusion_matrix.shape
            }
        else:
            cm_dict = confusion_matrix
        
        # Save as JSON
        with open(artifact_name, 'w') as f:
            json.dump(cm_dict, f, indent=2)
        
        mlflow.log_artifact(artifact_name)
        os.remove(artifact_name)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log sklearn or any compatible model
        
        Args:
            model: The trained model
            artifact_path: Path within the run's artifact directory
            registered_model_name: If provided, register model in model registry
            signature: Model signature
            input_example: Example input for the model
        """
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        except Exception as e:
            print(f"Warning: Could not log model with sklearn flavor: {e}")
            # Fallback to generic Python model
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=registered_model_name
            )
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log information about the training dataset
        
        Args:
            dataset_info: Dictionary with dataset information
        """
        for key, value in dataset_info.items():
            if isinstance(value, (int, float, str)):
                mlflow.log_param(f"dataset_{key}", value)
            else:
                mlflow.log_param(f"dataset_{key}", str(value))
    
    def log_feature_importance(self, feature_names: list, importance_values: list):
        """
        Log feature importance as artifact
        
        Args:
            feature_names: List of feature names
            importance_values: List of importance values
        """
        import pandas as pd
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        # Save as CSV
        filename = 'feature_importance.csv'
        df.to_csv(filename, index=False)
        mlflow.log_artifact(filename)
        os.remove(filename)
    
    def log_training_metadata(self, metadata: Dict[str, Any]):
        """
        Log training metadata (timestamps, duration, etc.)
        
        Args:
            metadata: Dictionary of metadata
        """
        metadata_file = 'training_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        mlflow.log_artifact(metadata_file)
        os.remove(metadata_file)
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run
        
        Args:
            tags: Dictionary of tags
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def log_artifact_from_file(self, filepath: str):
        """
        Log an artifact from a file path
        
        Args:
            filepath: Path to the file to log
        """
        mlflow.log_artifact(filepath)
    
    def get_run_info(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a run
        
        Args:
            run_id: Run ID (defaults to current run)
        
        Returns:
            Dictionary with run information
        """
        if run_id is None and self.current_run is not None:
            run_id = self.current_run.info.run_id
        
        if run_id is None:
            raise ValueError("No run ID provided and no active run")
        
        run = self.client.get_run(run_id)
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }
    
    def load_model(self, run_id: str, artifact_path: str = "model"):
        """
        Load a model from a previous run
        
        Args:
            run_id: Run ID containing the model
            artifact_path: Path to the model within the run
        
        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.sklearn.load_model(model_uri)
    
    def search_runs(
        self,
        filter_string: str = "",
        order_by: list = None,
        max_results: int = 100
    ):
        """
        Search for runs in the experiment
        
        Args:
            filter_string: Filter query string
            order_by: List of columns to order by
            max_results: Maximum number of results
        
        Returns:
            List of runs
        """
        return self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["start_time DESC"],
            max_results=max_results
        )


# Convenience function for quick tracking
def track_fraud_model_training(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_params: Dict[str, Any],
    run_name: str = None,
    experiment_name: str = "fraud_detection"
):
    """
    Quick function to track a complete fraud model training session
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_params: Model parameters
        run_name: Name for the run
        experiment_name: Name of the experiment
    
    Returns:
        run_id: The MLflow run ID
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    import numpy as np
    
    tracker = FraudMLflowTracker(experiment_name=experiment_name)
    
    with tracker.start_run(run_name=run_name) as run:
        # Log parameters
        tracker.log_model_params(model_params)
        
        # Log dataset info
        tracker.log_dataset_info({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
            'train_fraud_ratio': float(np.mean(y_train)) if isinstance(y_train, np.ndarray) else sum(y_train) / len(y_train)
        })
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Log metrics
        tracker.log_fraud_detection_metrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
        
        # Log confusion matrix
        tracker.log_confusion_matrix(cm)
        
        # Log model
        tracker.log_model(model, registered_model_name="fraud_detection_model")
        
        # Set tags
        tracker.set_tags({
            'model_type': type(model).__name__,
            'task': 'fraud_detection',
            'timestamp': datetime.now().isoformat()
        })
        
        return run.info.run_id
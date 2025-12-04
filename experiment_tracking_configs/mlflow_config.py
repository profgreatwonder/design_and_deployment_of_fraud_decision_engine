"""
MLflow Configuration Management for Fraud Detection System
Handles loading configuration from environment variables
"""

import os
from dotenv import load_dotenv
from typing import Optional


class MLflowConfig:
    """
    Configuration manager for MLflow integration
    Loads settings from environment variables or .env file
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            env_file: Path to .env file (default: .env in current directory)
        """
        # Load environment variables from .env file if it exists
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # MLflow settings
        self.mlflow_tracking_uri = os.getenv(
            'MLFLOW_TRACKING_URI',
            'http://localhost:5000'
        )
        self.mlflow_experiment_name = os.getenv(
            'MLFLOW_EXPERIMENT_NAME',
            'fraud_detection'
        )
        
        # MinIO settings
        self.minio_endpoint_url = os.getenv(
            'MINIO_ENDPOINT_URL',
            'http://localhost:9000'
        )
        self.minio_root_user = os.getenv('MINIO_ROOT_USER', 'minioadmin')
        self.minio_root_password = os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
        self.minio_bucket = os.getenv('MINIO_BUCKET', 'mlflow')
        
        # MongoDB settings
        self.mongodb_uri = os.getenv('MONGODB_URI', '')
        
        # Validate critical settings
        self._validate()
    
    def _validate(self):
        """Validate that critical settings are present"""
        if not self.mongodb_uri and 'localhost' not in self.mlflow_tracking_uri:
            print("WARNING: MONGODB_URI not set. Make sure MLflow server has access to MongoDB.")
        
        if 'localhost' not in self.minio_endpoint_url and \
           (self.minio_root_user == 'minioadmin' or self.minio_root_password == 'minioadmin'):
            print("WARNING: Using default MinIO credentials. Set MINIO_ROOT_USER and MINIO_ROOT_PASSWORD.")
    
    def get_mlflow_config(self) -> dict:
        """Get MLflow configuration as dictionary"""
        return {
            'tracking_uri': self.mlflow_tracking_uri,
            'experiment_name': self.mlflow_experiment_name
        }
    
    def get_minio_config(self) -> dict:
        """Get MinIO configuration as dictionary"""
        return {
            'endpoint_url': self.minio_endpoint_url,
            'access_key': self.minio_root_user,
            'secret_key': self.minio_root_password,
            'bucket': self.minio_bucket
        }
    
    def set_environment_variables(self):
        """
        Set MLflow-related environment variables
        Call this before using MLflow to ensure proper configuration
        """
        os.environ['MLFLOW_TRACKING_URI'] = self.mlflow_tracking_uri
        os.environ['AWS_ACCESS_KEY_ID'] = self.minio_root_user
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.minio_root_password
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = self.minio_endpoint_url
    
    def print_config(self):
        """Print current configuration (without secrets)"""
        print("=" * 50)
        print("MLflow Configuration")
        print("=" * 50)
        print(f"Tracking URI: {self.mlflow_tracking_uri}")
        print(f"Experiment: {self.mlflow_experiment_name}")
        print(f"MinIO Endpoint: {self.minio_endpoint_url}")
        print(f"MinIO User: {self.minio_root_user}")
        print(f"MinIO Password: {'*' * len(self.minio_root_password)}")
        print(f"MinIO Bucket: {self.minio_bucket}")
        print(f"MongoDB URI: {'Set' if self.mongodb_uri else 'Not set'}")
        print("=" * 50)
    
    @staticmethod
    def is_configured() -> bool:
        """Check if MLflow is configured (not using default localhost)"""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        return 'localhost' not in tracking_uri and '127.0.0.1' not in tracking_uri


# Singleton instance
_config_instance = None


def get_config(env_file: Optional[str] = None) -> MLflowConfig:
    """
    Get or create MLflow configuration singleton
    
    Args:
        env_file: Path to .env file
    
    Returns:
        MLflowConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = MLflowConfig(env_file)
    return _config_instance


def setup_mlflow():
    """
    Convenience function to setup MLflow with configuration
    Call this at the start of your training scripts
    """
    config = get_config()
    config.set_environment_variables()
    config.print_config()
    return config


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = setup_mlflow()
    
    # Check if properly configured
    if MLflowConfig.is_configured():
        print("\n✓ MLflow is configured for remote tracking")
    else:
        print("\n⚠ MLflow is using local/default configuration")
        print("Set MLFLOW_TRACKING_URI environment variable for remote tracking")
# Install required packages
pip install mlflow pymongo boto3 python-dotenv

# Set environment variables
export MLFLOW_TRACKING_URI="https://fraud-detection-mlflow.onrender.com"
export MINIO_ENDPOINT_URL="https://fraud-detection-minio.onrender.com"
export MINIO_ROOT_USER="fraudshield"
export MINIO_ROOT_PASSWORD="fraudshield_password"

# Run test
python "design_and_deployment_of_fraud_decision_engine/experiment_tracking_configs/test-mlflow.py"
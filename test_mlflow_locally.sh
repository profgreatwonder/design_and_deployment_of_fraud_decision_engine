# Install required packages
pip install mlflow pymongo boto3 python-dotenv

# Set environment variables
export MLFLOW_TRACKING_URI="https://fraud-detection-mlflow.onrender.com"
export MINIO_ENDPOINT_URL="https://fraud-detection-minio.onrender.com"
export MINIO_ROOT_USER="your_username"
export MINIO_ROOT_PASSWORD="your_password"

# Run test
python test-mlflow.py
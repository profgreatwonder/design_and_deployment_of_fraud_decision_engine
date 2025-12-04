# Install minio package
pip install minio

# Set environment variables with your Render URL
export MINIO_ENDPOINT="https://fraud-detection-minio.onrender.com"
export MINIO_ROOT_USER="fraudshield"
export MINIO_ROOT_PASSWORD="fraudshielded_password"
export MINIO_BUCKET="mlflow"

# Run setup
python setup-minio.py
